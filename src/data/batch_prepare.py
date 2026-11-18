"""Prepare a versioned development-data export and account for every trip."""

import argparse
from collections import Counter
from contextlib import nullcontext
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from src.data.align_trips import export_trip, _sha256


COUNTS = (
    "total_observations", "confident", "ambiguous", "unavailable",
    "candidate_misses", "candidate_miss_denominator", "no_candidates_all_observations",
    "training_eligible",
)
CSV_FIELDS = ["trip_id", "observation_index", "arc_id", "distance_m", "label"]


def _validated_rows(trip_dir, summary):
    labels = {}
    observations = set()
    counts = Counter()
    with (trip_dir / "labels.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            index = record["observation_index"]
            if record["trip_id"] != summary["trip_id"] or index in observations:
                raise ValueError("Ledger contains incorrect trip IDs or duplicate observations")
            observations.add(index)
            counts[record["status"]] += 1
            counts["no_candidates_all_observations"] += record["candidate_count"] == 0
            counts["candidate_misses"] += record["candidate_miss"] is True
            if record["training_eligible"]:
                labels[index] = record["arc_id"]
    counts.update(total_observations=len(observations), training_eligible=len(labels),
                  candidate_miss_denominator=counts["confident"])
    if any(counts[key] != summary["counts"][key] for key in COUNTS):
        raise ValueError("Ledger and summary counts disagree")
    if summary["partition"] != "training":
        return []
    with (trip_dir / "approximate_candidates.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError("Unexpected training CSV schema")
        rows = list(reader)
    seen = set()
    positives = Counter()
    for row in rows:
        index, arc_id = int(row["observation_index"]), int(row["arc_id"])
        pair = index, arc_id
        if row["trip_id"] != summary["trip_id"] or pair in seen or index not in labels:
            raise ValueError("Training rows do not match the eligible observation ledger")
        seen.add(pair)
        positive = arc_id == labels[index]
        if row["label"] != str(int(positive)):
            raise ValueError("Training row label disagrees with ledger")
        if positive:
            positives[index] += 1
    if positives != Counter({index: 1 for index in labels}):
        raise ValueError("Expected exactly one positive per eligible observation")
    return rows


def prepare(trip_root, output_dir, partition):
    if partition not in {"training", "student-test"}:
        raise ValueError("Use an authorized training or student-test partition")
    root, destination = Path(trip_root).resolve(), Path(output_dir).resolve()
    if root == destination or root in destination.parents:
        raise ValueError("Output directory must be outside the input partition")
    trips = sorted(path for path in root.iterdir() if path.is_dir())
    if not trips:
        raise ValueError(f"No trip folders found in {root}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    training = partition == "training"
    totals, reasons = Counter(), Counter()
    manifest = {"partition": partition, "trip_count": len(trips), "trips": [],
                "verified_ground_truth": False,
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "batch_script_sha256": _sha256(Path(__file__))}
    csv_path = destination / "training_candidates.csv"
    with (csv_path.open("w", newline="", encoding="utf-8") if training else nullcontext()) as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS) if training else None
        if writer:
            writer.writeheader()
        candidate_rows = 0
        for trip in trips:
            try:
                with TemporaryDirectory(dir=destination) as staging:
                    summary = export_trip(trip, staging, partition, export_training=training)
                    trip_output = Path(staging) / trip.name
                    rows = _validated_rows(trip_output, summary)
                    provenance = json.loads((trip_output / "provenance.json").read_text(encoding="utf-8"))
                    trip_output.rename(destination / trip.name)
            except (OSError, ValueError, MemoryError) as exc:
                manifest["trips"].append({"trip_id": trip.name, "status": "failed",
                                           "observation_count": None,
                                           "error": f"{type(exc).__name__}: {exc}"})
                print(f"{trip.name}: failed: {exc}", flush=True)
                continue
            status = "processed" if summary["counts"]["confident"] else "no_confident_labels"
            manifest["trips"].append({**summary, "status": status,
                                      "artifact_directory": trip.name,
                                      "input_sha256": provenance["input_sha256"]})
            totals.update({key: summary["counts"][key] for key in COUNTS})
            reasons.update(summary["reason_counts"])
            if writer:
                writer.writerows(rows)
                candidate_rows += len(rows)
            print(f"{trip.name}: {status}", flush=True)
    failed = sum(trip["status"] == "failed" for trip in manifest["trips"])
    manifest.update(processed_trips=len(trips) - failed, failed_trips=failed,
                    totals={key: totals[key] for key in COUNTS}, reason_counts=dict(reasons),
                    candidate_miss_rate=(totals["candidate_misses"] / totals["candidate_miss_denominator"]
                                         if totals["candidate_miss_denominator"] else None))
    if training:
        manifest["training_csv"] = {"path": csv_path.name, "rows": candidate_rows,
                                    "sha256": _sha256(csv_path)}
    fingerprint = {
        "partition": partition,
        "batch_script_sha256": manifest["batch_script_sha256"],
        "trips": [(trip["trip_id"], trip.get("input_sha256"), trip["status"])
                  for trip in manifest["trips"]],
        "training_csv": manifest.get("training_csv"),
    }
    if len(trips) > failed:
        manifest["method"] = {key: provenance[key] for key in
                              ("method_version", "config", "candidate_policy", "source_sha256")}
        fingerprint["method"] = manifest["method"]
    manifest["dataset_sha256"] = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "trips"}, indent=2),
          flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip_root")
    parser.add_argument("--partition", required=True, choices=["training", "student-test"])
    parser.add_argument("--output-dir", required=True, help="New directory for this run")
    args = parser.parse_args()
    result = prepare(args.trip_root, args.output_dir, args.partition)
    raise SystemExit(1 if result["failed_trips"] else 0)
