"""Prepare and diagnose the permitted student-test proxy evaluation ledger."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from src.data.align_trips import _sha256
from src.data.batch_prepare import prepare
from src.data.heading import HeadingConfig, TripHeading
from src.data.pipeline import (
    CANDIDATE_RADIUS_M, MAX_CANDIDATES_PER_OBSERVATION,
    _point_segment_distances, _project_local, generate_candidates, load_trip,
)
from src.data.review_alignment import _reference_breaks


VERSION = "student-evaluation-v1"


def _diagnose(trip_dir, trip_output, summary):
    labels = [json.loads(line) for line in (trip_output / "labels.jsonl").read_text(
        encoding="utf-8").splitlines()]

    # Complete the inference-side calculations before opening the offline reference.
    trip = load_trip(trip_dir)
    candidates = generate_candidates(trip)
    heading = TripHeading(trip)
    observations = trip["observations"]
    arcs, nodes = trip["network"]["arcs"], trip["network"]["nodes"]
    xy = _project_local(nodes + [[o["longitude"], o["latitude"]] for o in observations])
    points = {o["observation_index"]: point for o, point in zip(observations, xy[len(nodes):])}
    arc_nodes = np.asarray(arcs)
    starts, ends = xy[arc_nodes[:, 0]], xy[arc_nodes[:, 1]]
    by_index = defaultdict(set)
    candidate_heading = Counter()
    for candidate in candidates:
        idx, arc = candidate["observation_index"], candidate["arc_id"]
        by_index[idx].add(arc)
        _, reason = heading.agreement(idx, arc)
        candidate_heading[reason or "available"] += 1
    observation_heading = Counter(reason or "available" for _, reason in heading.observations.values())
    examples = {}
    for idx, (_, reason) in heading.observations.items():
        if reason and reason not in examples:
            step, gap = heading.steps[idx]
            examples[reason] = {"observation_index": idx, "step_m": step, "gap_seconds": gap}
    if [r["observation_index"] for r in labels] != list(points):
        raise ValueError(f"Ledger does not cover the original observation sequence for {trip['trip_id']}")

    route_trip = load_trip(trip_dir, include_reference=True)
    route = route_trip["reference_route"]
    breaks = []
    for position in _reference_breaks(route_trip):
        previous, current = route[position - 1:position + 1]
        breaks.append({"route_position": position,
                       "share_any_node": bool(set(arcs[previous]) & set(arcs[current])),
                       "directed_endpoint_gap_m": float(np.linalg.norm(ends[previous] - starts[current]))})
    reference_ids = np.array(route, dtype=int)
    misses, distant = [], []
    for label in labels:
        idx = label["observation_index"]
        if len(by_index[idx]) != label["candidate_count"]:
            raise ValueError(f"Candidate count differs from ledger at {trip['trip_id']}:{idx}")
        arc = label["arc_id"]
        if label["candidate_miss"] is True:
            if arc in by_index[idx]:
                raise ValueError("Reported candidate miss is present after regeneration")
            distances = _point_segment_distances(points[idx], starts, ends)
            # Rank exactly as generate_candidates does, including equal-distance arc ID ties.
            radius = float(distances[arc])
            rank = 1 + int(np.count_nonzero((distances < radius) |
                                           ((distances == radius) & (np.arange(len(arcs)) < arc))))
            if radius > CANDIDATE_RADIUS_M:
                reason = "outside_candidate_radius"
            elif rank > MAX_CANDIDATES_PER_OBSERVATION:
                reason = "below_candidate_rank_cutoff"
            else:
                reason = "unexplained_candidate_miss"
            misses.append({"observation_index": idx, "inferred_arc_id": arc,
                           "inferred_arc_distance_m": radius, "nearest_network_distance_m": float(distances.min()),
                           "inferred_arc_rank": rank, "reason": reason})
        if "distant_from_aligned_route_segment" in label["reasons"]:
            distances = _point_segment_distances(points[idx], starts, ends)
            distant.append({"observation_index": idx,
                            "selected_reference_distance_m": float(distances[arc]),
                            "nearest_reference_distance_m": float(distances[reference_ids].min()),
                            "nearest_network_distance_m": float(distances.min())})
    if len(misses) != summary["counts"]["candidate_misses"]:
        raise ValueError("Diagnosis does not account for every confident label miss")
    return {
        "input_sha256": summary["input_sha256"],
        "files_sha256": {name: _sha256(trip_output / name) for name in
                         ("labels.jsonl", "summary.json", "provenance.json")},
        "candidate_misses": misses, "distant_observations": distant,
        "disconnected_reference_transitions": breaks,
        "gps_heading_status": dict(observation_heading),
        "candidate_heading_status": dict(candidate_heading),
        "heading_unavailable_examples": examples,
    }


def evaluate(student_test_root, output_dir):
    root, output = Path(student_test_root).resolve(), Path(output_dir).resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError("Use a new output directory outside the authorized student-test data")
    if sum(path.is_dir() for path in root.iterdir()) != 10:
        raise ValueError("Expected exactly the ten authorized student-test trip folders")
    prepared = output / "ledgers"
    manifest = prepare(root, prepared, "student-test")
    if manifest["failed_trips"]:
        raise ValueError("Student-test ledger is incomplete. See the batch manifest.")
    trips = {summary["trip_id"]: _diagnose(root / summary["trip_id"], prepared / summary["trip_id"], summary)
             for summary in manifest["trips"]}
    misses = Counter(miss["reason"] for trip in trips.values() for miss in trip["candidate_misses"])
    gps_heading, candidate_heading = Counter(), Counter()
    for trip in trips.values():
        gps_heading.update(trip["gps_heading_status"])
        candidate_heading.update(trip["candidate_heading_status"])
    report = {
        "version": VERSION, "scope": "permitted student-test; offline proxy evaluation only",
        "verified_ground_truth": False, "dataset_sha256": manifest["dataset_sha256"],
        "ledger_manifest_sha256": _sha256(prepared / "manifest.json"),
        "method": manifest["method"], "counts": manifest["totals"],
        "miss_causes": dict(misses), "heading_config": asdict(HeadingConfig()),
        "gps_heading_status": dict(gps_heading), "candidate_heading_status": dict(candidate_heading),
        "diagnostic_code_sha256": {name: _sha256(Path(__file__).with_name(name)) for name in
                                   ("student_evaluation.py", "heading.py")},
        "trips": trips,
    }
    (output / "diagnostics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student_test_root", help="The explicitly authorized ten-trip student-test folder")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = evaluate(args.student_test_root, args.output_dir)
    print(json.dumps({key: value for key, value in report.items() if key != "trips"}, indent=2))
