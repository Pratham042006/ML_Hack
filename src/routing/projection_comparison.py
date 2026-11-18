"""Compare midpoint and projected Viterbi on one permitted development trip."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from src.data.alignment import align_reference, assess_candidate_coverage
from src.data.pipeline import generate_candidates, load_trip
from src.models.inference import score_candidates
from src.routing.continuity import audit_continuity
from src.routing.decoder import (
    DEFAULT_MAX_GAP_SECONDS, DEFAULT_MAX_SPEED_MPS,
    DirectedRoadGraph, decode_classifier_only, decode_with_diagnostics,
)
from src.routing.integration_demo import validate_pipeline_outputs
from src.routing.map_diagnostics import matched_blocks
from src.routing.projected import (
    BACKWARD_JITTER_M, BACKWARD_JITTER_MAX_GAP_SECONDS,
    ProjectedDistance, candidate_fractions, decode_projected,
)
from src.routing.proxy_evaluation import summarize_proxy


def compare_trip(
    path, *, with_proxy=False, model_artifact=None,
    candidate_generator=generate_candidates, position_provider=candidate_fractions,
):
    """Compare scores on shared candidates, with optional teammate providers."""
    path = Path(path)
    if path.resolve().parent.name not in {"train", "student_test"}:
        raise ValueError("expected a permitted trip under the train or student_test split")
    artifact_path = Path(model_artifact) if model_artifact is not None else None
    if artifact_path is not None and not artifact_path.is_file():
        raise ValueError(f"model artifact is not a file: {artifact_path}")

    started = perf_counter()
    trip = load_trip(path, include_reference=False)
    observations, network = trip["observations"], trip["network"]
    candidates = candidate_generator(trip)
    if artifact_path is not None:
        from src.models.inference import _load_artifact
        artifact = _load_artifact(artifact_path)
        if "heading_agreement" in artifact["feature_names"]:
            from src.data.heading import add_heading_features
            candidates = add_heading_features(trip, candidates)
    scored = score_candidates(candidates)
    shared_seconds = perf_counter() - started

    started = perf_counter()
    fractions = position_provider(trip, candidates)
    distance = ProjectedDistance(fractions)
    projection_seconds = perf_counter() - started
    graph = DirectedRoadGraph(network)
    baseline, baseline_predictions = _run_decoders(trip, candidates, scored, distance, graph)
    report = {
        "trip_id": trip["trip_id"],
        "observations": len(observations),
        "candidates": len(candidates),
        "scorer": "distance baseline",
        "null_projections": sum(f is None for f in fractions.values()),
        "backward_jitter_metres": BACKWARD_JITTER_M,
        "backward_jitter_max_gap_seconds": BACKWARD_JITTER_MAX_GAP_SECONDS,
        "max_gap_seconds": DEFAULT_MAX_GAP_SECONDS,
        "max_speed_mps": DEFAULT_MAX_SPEED_MPS,
        "load_generate_score_seconds": round(shared_seconds, 4),
        "projection_preparation_seconds": round(projection_seconds, 4),
        **baseline,
    }
    trained_predictions = None
    if artifact_path is not None:
        with artifact_path.open("rb") as stream:
            digest = sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            artifact_hash = digest.hexdigest()
        started = perf_counter()
        artifact_scored = score_candidates(candidates, model_artifact=artifact_path)
        artifact_seconds = perf_counter() - started
        trained, trained_predictions = _run_decoders(
            trip, candidates, artifact_scored, distance, graph,
        )
        report["model_comparison"] = {
            "artifact_filename": artifact_path.name,
            "artifact_sha256": artifact_hash,
            "score_seconds_including_artifact_load": round(artifact_seconds, 4),
            **trained,
            "changed_matches_vs_distance_baseline": {
                name: sum(a["arc_id"] != b["arc_id"] for a, b in zip(
                    baseline_predictions[name], trained_predictions[name],
                ))
                for name in trained_predictions
            },
        }
    if with_proxy:
        # Every decoder has already returned. Reference data is used only here.
        reference_trip = load_trip(path, include_reference=True)
        ledger = assess_candidate_coverage(align_reference(reference_trip), candidates)
        report["offline_proxy"] = {
            "provenance": ledger["provenance"],
            "label_counts": ledger["counts"],
            "exclusion_reason_counts": dict(Counter(
                reason for label in ledger["labels"] if label["status"] != "confident"
                for reason in label["reasons"]
            )),
            "agreement": {
                name: summarize_proxy(ledger, predictions)
                for name, predictions in baseline_predictions.items()
            },
        }
        if trained_predictions is not None:
            report["offline_proxy"]["model_agreement"] = {
                name: summarize_proxy(ledger, predictions)
                for name, predictions in trained_predictions.items()
            }
    return report


def _run_decoders(trip, candidates, scored, distance, graph):
    observations, network = trip["observations"], trip["network"]
    decoders = {
        "classifier_only": lambda: (decode_classifier_only(observations, scored, network), []),
        "midpoint": lambda: decode_with_diagnostics(observations, scored, network),
        "projected": lambda: decode_projected(observations, scored, network, distance=distance),
    }
    predictions_by_method, pair_details, runs = {}, {}, {}
    for name, decode in decoders.items():
        started = perf_counter()
        predictions, events = decode()
        decode_seconds = perf_counter() - started
        validation = validate_pipeline_outputs(observations, candidates, predictions)
        if not validation["all_checks_passed"]:
            raise ValueError(f"{name} predictions failed routing validation: {validation['errors']}")
        midpoint_counts, midpoint_pairs = audit_continuity(observations, predictions, graph)
        projected_counts, projected_pairs = audit_continuity(
            observations, predictions, graph, travel_distance=distance,
        )
        matched = sum(p["arc_id"] is not None for p in predictions)
        predictions_by_method[name] = predictions
        pair_details[name] = midpoint_pairs, projected_pairs
        runs[name] = {
            "validation": validation,
            "matched": matched,
            "unmatched": len(observations) - matched,
            "decode_seconds": round(decode_seconds, 4),
            "midpoint_audit": midpoint_counts,
            "projected_audit": projected_counts,
        }
        if name != "classifier_only":
            runs[name]["diagnostics"] = {
                "event_counts": dict(Counter(e["event"] for e in events)),
                "events": events,
                "matched_blocks": matched_blocks(predictions, events),
            }

    examples, changed = _compare_pairs(
        trip, predictions_by_method, pair_details, scored, distance.fractions,
    )
    return {
        "runs": runs,
        "midpoint_vs_projected_changed_matches": changed,
        "examples": examples,
    }, predictions_by_method


def _compare_pairs(trip, predictions, details, scored, fractions):
    observations = trip["observations"]
    gps_distances = {(c["observation_index"], c["arc_id"]): c["features"]["distance_m"] for c in scored}
    examples = {"reinspection": [], "plausibility_gain": [], "plausibility_regression": []}
    changed = 0
    blocked = {"disconnected", "over_speed_cap"}
    for position, observation in enumerate(observations):
        old = predictions["midpoint"][position]["arc_id"]
        new = predictions["projected"][position]["arc_id"]
        changed += old != new
        if position == 0:
            continue
        previous_index = observations[position - 1]["observation_index"]
        index = observation["observation_index"]
        old_status = details["midpoint"][1][index]["status"]
        new_status = details["projected"][1][index]["status"]
        targets = []
        if old_status in blocked and new_status == "plausible":
            targets.append("plausibility_gain")
        if old_status == "plausible" and new_status in blocked:
            targets.append("plausibility_regression")
        if trip["trip_id"] == "00000005" and index in {2, 25, 26}:
            targets.append("reinspection")
        targets = [name for name in targets if len(examples[name]) < 3]
        if not targets:
            continue
        example = {"from_observation_index": previous_index, "to_observation_index": index}
        for name, records in predictions.items():
            first, second = records[position - 1]["arc_id"], records[position]["arc_id"]
            selected = ((previous_index, first), (index, second))
            example[name] = {
                "arc_ids": [first, second],
                "projection_fractions": [fractions.get(key) for key in selected],
                "gps_distances_m": [gps_distances.get(key) for key in selected],
                "midpoint_audit": details[name][0][index],
                "projected_audit": details[name][1][index],
            }
        for name in targets:
            examples[name].append(example)
    return examples, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip", type=Path)
    parser.add_argument("--with-proxy", action="store_true", help="Read development reference after inference")
    parser.add_argument("--model-artifact", type=Path, help="Trusted PathSnap artifact for a paired score comparison")
    args = parser.parse_args()
    if args.trip.resolve().parent.name not in {"train", "student_test"}:
        parser.error("expected a permitted trip under the train or student_test split")
    print(json.dumps(compare_trip(
        args.trip, with_proxy=args.with_proxy, model_artifact=args.model_artifact,
    ), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
