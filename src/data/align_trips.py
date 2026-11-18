"""Export approximate reference labels for explicitly permitted development trips."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from src.data.alignment import align_reference, assess_candidate_coverage
from src.data.pipeline import (
    load_trip, generate_candidates, CANDIDATE_RADIUS_M, MAX_CANDIDATES_PER_OBSERVATION,
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_trip(trip_dir, output_dir, partition, export_training=False):
    if partition not in {"training", "student-test"}:
        raise ValueError("Only explicitly permitted development partitions are supported")
    if export_training and partition != "training":
        raise ValueError("Training CSV export is restricted to the training partition")
    directory = Path(trip_dir).resolve()
    destination = Path(output_dir).resolve() / directory.name
    if destination == directory or directory in destination.parents:
        raise ValueError("Write sidecars outside the raw trip directory")
    trip = load_trip(directory, include_reference=True)
    alignment = align_reference(trip)
    candidates = generate_candidates(trip)  # Independent inference pipeline.
    assessed = assess_candidate_coverage(alignment, candidates)
    provenance = {
        **assessed["provenance"], "trip_id": trip["trip_id"], "partition": partition,
        "partition_authorization": "Explicit caller declaration; not inferred from folder name",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": {
            suffix: _sha256(directory / f"{trip['trip_id']}.{suffix}")
            for suffix in ["track", "nodes", "arcs", "route"]
        },
        "source_sha256": {
            name: _sha256(Path(__file__).parent / name)
            for name in ["alignment.py", "pipeline.py", "align_trips.py"]
        },
        "best_path_cost": assessed["best_path_cost"],
        "candidate_policy": {"radius_m": CANDIDATE_RADIUS_M,
                             "max_per_observation": MAX_CANDIDATES_PER_OBSERVATION},
        "warning": "Approximate inferred labels, not organizer ground truth or calibrated confidence",
    }
    labels = [{"trip_id": trip["trip_id"], **record} for record in assessed["labels"]]
    reason_counts = Counter(reason for record in labels for reason in record["reasons"])
    summary = {"trip_id": trip["trip_id"], "partition": partition,
               "counts": assessed["counts"], "reason_counts": dict(reason_counts),
               "reference_entries": len(trip["reference_route"]),
               "best_path_cost": assessed["best_path_cost"],
               "label_source": "approximate_reference_alignment",
               "verified_ground_truth": False}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "labels.jsonl").write_text(
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in labels), encoding="utf-8")
    for name, content in [("provenance.json", provenance), ("summary.json", summary)]:
        (destination / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if export_training:
        eligible = {record["observation_index"]: record["arc_id"] for record in labels
                    if record["training_eligible"]}
        with (destination / "approximate_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "trip_id", "observation_index", "arc_id", "distance_m", "label"])
            writer.writeheader()
            for candidate in candidates:
                index = candidate["observation_index"]
                if index in eligible:
                    writer.writerow({"trip_id": trip["trip_id"], "observation_index": index,
                                     "arc_id": candidate["arc_id"],
                                     "distance_m": candidate["features"]["distance_m"],
                                     "label": int(candidate["arc_id"] == eligible[index])})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip_dirs", nargs="+")
    parser.add_argument("--partition", required=True, choices=["training", "student-test"])
    parser.add_argument("--output-dir", default="outputs/alignment")
    parser.add_argument("--export-training", action="store_true",
                        help="Also export confident, candidate-present observations for Pratham's CSV format")
    args = parser.parse_args()
    for trip_dir in args.trip_dirs:
        print(json.dumps(export_trip(trip_dir, args.output_dir, args.partition, args.export_training), indent=2))
