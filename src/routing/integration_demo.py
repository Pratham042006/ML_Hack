"""Real-trip integration demo and contract validation for the routing decoder.

Runnable from repository root as:
    python -m src.routing.integration_demo "PATH_TO_TRIP"

Directly imports teammate interfaces per docs/INTEGRATION_CONTRACT.md:
    - load_trip, generate_candidates from src.data.pipeline
    - score_candidates from src.models.inference
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any

from src.data.pipeline import generate_candidates, load_trip
from src.models.inference import score_candidates
from src.routing.continuity import audit_continuity
from src.routing.map_diagnostics import matched_blocks
from src.routing.decoder import (
    DEFAULT_MAX_GAP_SECONDS,
    DEFAULT_MAX_SPEED_MPS,
    DirectedRoadGraph,
    decode_classifier_only,
    decode_with_diagnostics,
)


def validate_pipeline_outputs(
    observations: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate all contract guarantees and invariants."""
    checks = {
        "exactly_one_prediction_per_observation": True,
        "input_order_preserved": True,
        "selected_arc_belongs_to_candidate_set": True,
        "unmatched_outputs_have_null_arc_and_empty_geometry": True,
        "matched_outputs_have_nonempty_geometry": True,
    }
    errors: list[str] = []

    if len(predictions) != len(observations):
        checks["exactly_one_prediction_per_observation"] = False
        errors.append(
            f"Prediction count ({len(predictions)}) does not match observation count ({len(observations)})"
        )

    obs_candidates: dict[int, set[int]] = defaultdict(set)
    for c in candidates:
        obs_idx = c.get("observation_index")
        arc_id = c.get("arc_id")
        if obs_idx is not None and arc_id is not None:
            obs_candidates[obs_idx].add(arc_id)

    for pos, (obs, pred) in enumerate(zip(observations, predictions)):
        obs_idx = obs.get("observation_index")
        pred_idx = pred.get("observation_index")
        if obs_idx != pred_idx:
            checks["input_order_preserved"] = False
            errors.append(
                f"Position {pos}: observation index mismatch (expected {obs_idx}, got {pred_idx})"
            )
            break

    for pred in predictions:
        obs_idx = pred.get("observation_index")
        arc_id = pred.get("arc_id")
        geometry = pred.get("geometry")

        if arc_id is None:
            if "arc_id" not in pred or geometry != []:
                checks["unmatched_outputs_have_null_arc_and_empty_geometry"] = False
                errors.append(
                    f"Observation {obs_idx}: unmatched prediction must have geometry=[], got {geometry!r}"
                )
        else:
            valid_arcs = obs_candidates.get(obs_idx, set())
            if arc_id not in valid_arcs:
                checks["selected_arc_belongs_to_candidate_set"] = False
                errors.append(
                    f"Observation {obs_idx}: selected arc_id {arc_id} not in candidate set {sorted(valid_arcs)}"
                )
            if not isinstance(geometry, list) or len(geometry) < 2:
                checks["matched_outputs_have_nonempty_geometry"] = False
                errors.append(
                    f"Observation {obs_idx}: matched prediction has invalid geometry: {geometry!r}"
                )

    return {
        "all_checks_passed": all(checks.values()) and len(errors) == 0,
        "checks": checks,
        "error_count": len(errors),
        "errors": errors[:10],
    }


def build_integration_report(
    trip: dict[str, Any],
    candidates: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    classifier_predictions: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    timings: dict[str, float],
    trip_path: str | Path,
) -> dict[str, Any]:
    """Assemble structured, concise, machine-readable validation report."""
    observations = trip["observations"]
    total_observations = len(observations)

    validation_results = validate_pipeline_outputs(observations, candidates, predictions)
    classifier_validation = validate_pipeline_outputs(
        observations, candidates, classifier_predictions
    )
    validation_results["classifier_only"] = classifier_validation
    validation_results["all_checks_passed"] &= classifier_validation["all_checks_passed"]

    matched_count = sum(1 for p in predictions if p.get("arc_id") is not None)
    unmatched_count = total_observations - matched_count
    coverage_rate = matched_count / total_observations if total_observations > 0 else 0.0

    obs_cand_counts = defaultdict(int)
    for c in candidates:
        obs_cand_counts[c["observation_index"]] += 1
    obs_with_cands = sum(1 for count in obs_cand_counts.values() if count > 0)
    obs_without_cands = total_observations - obs_with_cands
    all_counts = [obs_cand_counts[obs["observation_index"]] for obs in observations]

    differences: list[dict[str, Any]] = []
    agreement_count = 0
    for obs, c_pred, v_pred in zip(observations, classifier_predictions, predictions):
        obs_idx = obs["observation_index"]
        c_arc = c_pred.get("arc_id")
        v_arc = v_pred.get("arc_id")
        if c_arc == v_arc:
            agreement_count += 1
        else:
            differences.append({
                "observation_index": obs_idx,
                "classifier_arc_id": c_arc,
                "viterbi_arc_id": v_arc,
            })
    difference_count = len(differences)
    agreement_rate = agreement_count / total_observations if total_observations > 0 else 0.0

    classifier_matched = sum(p.get("arc_id") is not None for p in classifier_predictions)
    continuity = None
    if validation_results["all_checks_passed"]:
        graph = DirectedRoadGraph(trip["network"])
        baseline_counts, baseline_pairs = audit_continuity(observations, classifier_predictions, graph)
        viterbi_counts, viterbi_pairs = audit_continuity(observations, predictions, graph)
        continuity = {
            "max_gap_seconds": DEFAULT_MAX_GAP_SECONDS,
            "max_speed_mps": DEFAULT_MAX_SPEED_MPS,
            "classifier_only": baseline_counts,
            "viterbi": viterbi_counts,
        }
        scored_by_key = {(c["observation_index"], c["arc_id"]): c for c in scored}
        following = dict(zip(
            (o["observation_index"] for o in observations),
            (o["observation_index"] for o in observations[1:]),
        ))
        for difference in differences[:10]:
            index = difference["observation_index"]
            for name, pairs in (("classifier", baseline_pairs), ("viterbi", viterbi_pairs)):
                candidate = scored_by_key.get((index, difference[f"{name}_arc_id"]))
                difference[name] = {
                    "score": candidate["score"] if candidate else None,
                    "gps_distance_m": candidate["features"]["distance_m"] if candidate else None,
                    "incoming": pairs.get(index),
                    "outgoing": pairs.get(following.get(index)),
                }

    event_counts: dict[str, int] = {
        "missing_candidates": 0,
        "large_gap_reset": 0,
        "unreachable_transition_reset": 0,
        "nonpositive_timestamp_reset": 0,
        "all_zero_scores_uniform": 0,
    }
    for d in diagnostics:
        event = d.get("event")
        if event in event_counts:
            event_counts[event] += 1
        elif event:
            event_counts[event] = event_counts.get(event, 0) + 1

    return {
        "trip_id": trip.get("trip_id", ""),
        "trip_path": str(trip_path),
        "disclaimers": {
            "scorer": "distance baseline (exp(-distance_m / 20.0)), not a trained model",
            "reference_labels": "not loaded; load_trip was called with include_reference=False; .route files are never read",
            "metrics_interpretation": "matched/unmatched counts represent pipeline coverage only, NOT ground-truth route accuracy",
        },
        "validation": validation_results,
        "coverage": {
            "total_observations": total_observations,
            "matched_observations": matched_count,
            "unmatched_observations": unmatched_count,
            "coverage_rate": round(coverage_rate, 4),
            "note": "Coverage measures observations with an assigned arc candidate; does not reflect ground-truth accuracy",
        },
        "candidate_counts": {
            "total_candidates": len(candidates),
            "total_scored_candidates": len(scored),
            "observations_with_candidates": obs_with_cands,
            "observations_without_candidates": obs_without_cands,
            "mean_candidates_per_observation": round(
                sum(all_counts) / total_observations if total_observations > 0 else 0.0,
                2,
            ),
            "min_candidates_per_observation": min(all_counts) if all_counts else 0,
            "max_candidates_per_observation": max(all_counts) if all_counts else 0,
        },
        "timings_seconds": {
            "load_trip_s": round(timings.get("load_trip_s", 0.0), 4),
            "generate_candidates_s": round(timings.get("generate_candidates_s", 0.0), 4),
            "score_candidates_s": round(timings.get("score_candidates_s", 0.0), 4),
            "decode_s": round(timings.get("decode_s", 0.0), 4),
            "classifier_only_decode_s": round(timings.get("classifier_only_decode_s", 0.0), 4),
            "total_pipeline_s": round(timings.get("total_pipeline_s", 0.0), 4),
        },
        "classifier_vs_viterbi": {
            "classifier_only_matched_observations": classifier_matched,
            "classifier_only_unmatched_observations": total_observations - classifier_matched,
            "total_observations": total_observations,
            "agreement_count": agreement_count,
            "difference_count": difference_count,
            "agreement_rate": round(agreement_rate, 4),
            "differing_observations_count": difference_count,
            "sample_differences": differences[:10],
        },
        "diagnostics": {
            "total_diagnostic_events": len(diagnostics),
            "event_breakdown": event_counts,
            "events": diagnostics,
            "matched_blocks": (
                matched_blocks(predictions, diagnostics)
                if validation_results["all_checks_passed"] else []
            ),
        },
        "continuity": continuity,
    }


def run_integration_demo(trip_path: str | Path) -> dict[str, Any]:
    """Execute end-to-end integration demo for one trip directory."""
    path = Path(trip_path)
    t0 = time.perf_counter()

    t_load_start = time.perf_counter()
    trip = load_trip(path, include_reference=False)
    t_load = time.perf_counter() - t_load_start

    t_cands_start = time.perf_counter()
    candidates = generate_candidates(trip)
    t_cands = time.perf_counter() - t_cands_start

    t_score_start = time.perf_counter()
    scored = score_candidates(candidates)
    t_score = time.perf_counter() - t_score_start

    t_decode_start = time.perf_counter()
    predictions, diagnostics = decode_with_diagnostics(
        trip["observations"], scored, trip["network"]
    )
    t_decode = time.perf_counter() - t_decode_start

    t_classifier_start = time.perf_counter()
    classifier_predictions = decode_classifier_only(
        trip["observations"], scored, trip["network"]
    )
    t_classifier = time.perf_counter() - t_classifier_start
    t_total = time.perf_counter() - t0

    timings = {
        "load_trip_s": t_load,
        "generate_candidates_s": t_cands,
        "score_candidates_s": t_score,
        "decode_s": t_decode,
        "classifier_only_decode_s": t_classifier,
        "total_pipeline_s": t_total,
    }

    return build_integration_report(
        trip=trip,
        candidates=candidates,
        scored=scored,
        predictions=predictions,
        classifier_predictions=classifier_predictions,
        diagnostics=diagnostics,
        timings=timings,
        trip_path=path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real-trip routing integration demo and contract validation."
    )
    parser.add_argument(
        "trip_path",
        help="Path to trip directory containing <trip_id>.track, .nodes, and .arcs",
    )
    args = parser.parse_args()

    try:
        report = run_integration_demo(args.trip_path)
        print(json.dumps(report, indent=2))
        if not report["validation"]["all_checks_passed"]:
            sys.exit(1)
    except Exception as exc:
        error_output = {
            "error": str(exc),
            "trip_path": args.trip_path,
            "validation": {"all_checks_passed": False},
        }
        print(json.dumps(error_output, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
