"""Append an experimental heading feature to a verified training snapshot."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data.align_trips import _sha256
from src.data.batch_prepare import CSV_FIELDS
from src.data.heading import HeadingConfig, TripHeading
from src.data.pipeline import load_trip


VERSION = "heading-v1"
FIELD = "heading_agreement"


def _source_manifest(path, csv_hash):
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if manifest.get("partition") != "training" or manifest.get("verified_ground_truth") is not False:
        raise ValueError("Expected a training manifest for approximate reference labels")
    expected = manifest.get("combined_sha256") or manifest.get("training_csv", {}).get("sha256")
    if expected != csv_hash:
        raise ValueError(f"Training CSV hash is {csv_hash}, but the source manifest expects {expected}")
    if "trips" in manifest:
        trips = {row["trip_id"]: row for row in manifest["trips"]}
        if any(row["status"] == "failed" for row in trips.values()):
            raise ValueError("Source manifest includes failed trips with unverified inputs")
    else:
        trips = {trip: {"counts": counts} for trip, counts in manifest["trip_counts"].items()}
    return manifest, trips


def _read_candidates(path, trips):
    rows, positives = defaultdict(list), Counter()
    ordered_rows, seen_keys, observations = [], set(), set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if list(reader.fieldnames or []) != CSV_FIELDS:
            raise ValueError(f"Expected CSV columns {CSV_FIELDS}")
        for row in reader:
            trip_id = row["trip_id"]
            observation, arc = int(row["observation_index"]), int(row["arc_id"])
            key = trip_id, observation, arc
            if (trip_id not in trips or not trip_id.isascii() or not trip_id.isdecimal()
                    or min(observation, arc) < 0 or key in seen_keys
                    or row["label"] not in ("0", "1") or None in row):
                raise ValueError(f"Invalid or duplicate candidate at CSV line {reader.line_num}")
            seen_keys.add(key)
            observations.add(key[:2])
            positives[key[:2]] += row["label"] == "1"
            rows[trip_id].append((observation, arc, row))
            ordered_rows.append(row)
    if not ordered_rows or any(positives[key] != 1 for key in observations):
        raise ValueError("Expected exactly one positive row per included observation")
    return rows, ordered_rows


def export(training_root, input_csv, source_manifest, output_dir, config=HeadingConfig()):
    root, source, provenance, destination = map(
        lambda path: Path(path).resolve(), (training_root, input_csv, source_manifest, output_dir))
    if destination.exists() or destination.is_relative_to(root):
        raise ValueError("Use a new export directory outside the raw training root")
    csv_hash, manifest_hash = _sha256(source), _sha256(provenance)
    source_data, trips = _source_manifest(provenance, csv_hash)
    trip_rows, ordered_rows = _read_candidates(source, trips)
    observation_reasons, candidate_reasons, thresholds = Counter(), Counter(), Counter()
    trip_summaries, examples = {}, {}
    for trip_id, metadata in sorted(trips.items()):
        if not trip_id.isascii() or not trip_id.isdecimal():
            raise ValueError(f"Invalid trip directory ID in manifest: {trip_id}")
        trip_dir = root / trip_id
        hashes = {suffix: _sha256(trip_dir / f"{trip_id}.{suffix}")
                  for suffix in ("track", "nodes", "arcs")}
        known_hashes = metadata.get("input_sha256")
        if known_hashes and any(hashes[suffix] != known_hashes.get(suffix) for suffix in hashes):
            raise ValueError(f"Inference inputs for {trip_id} differ from the source manifest")
        heading = TripHeading(load_trip(trip_dir), config)
        all_observations = Counter(reason or "available" for _, reason in heading.observations.values())
        for distance, seconds in heading.steps.values():
            if seconds is None or seconds <= 0:
                continue
            if seconds <= config.max_gap_seconds:
                for limit in (3, 5, 10):
                    thresholds[f"displacement_under_{limit}m_at_configured_gap"] += distance < limit
            for limit in (5, 10, 20):
                thresholds[f"gap_over_{limit}s"] += seconds > limit
        eligible = defaultdict(bool)
        trip_reasons = Counter()
        for obs, arc, row in trip_rows[trip_id]:
            agreement, reason = heading.agreement(obs, arc)
            eligible[obs] |= agreement is not None
            trip_reasons[reason or "available"] += 1
            row[FIELD] = agreement
            if agreement is None:
                example_type = reason
            elif agreement > 0.9:
                example_type = "aligned"
            elif agreement < -0.9:
                example_type = "opposite"
            else:
                continue
            examples.setdefault(example_type, {"trip_id": trip_id, "observation_index": obs,
                                                "arc_id": arc, FIELD: agreement,
                                                "step_m": heading.steps[obs][0],
                                                "step_seconds": heading.steps[obs][1]})
        expected = metadata["counts"]
        if (len(heading.observations) != expected["total_observations"]
                or len(eligible) != expected["training_eligible"]):
            raise ValueError(f"Observation counts for {trip_id} disagree with source manifest")
        observation_reasons.update(all_observations)
        candidate_reasons.update(trip_reasons)
        trip_summaries[trip_id] = {
            "gps_observation_status": dict(all_observations),
            "candidate_heading_status": dict(trip_reasons),
            "eligible_observations": len(eligible),
            "eligible_observations_all_headings_missing": sum(not value for value in eligible.values()),
            "eligible_observations_gps_heading_missing": sum(
                heading.observations[index][0] is None for index in eligible),
            "inference_input_sha256": hashes,
        }
    if _sha256(source) != csv_hash or _sha256(provenance) != manifest_hash:
        raise ValueError("Source snapshot changed during feature export")
    report = {
        "version": VERSION, "features": ["distance_m", FIELD], "config": asdict(config),
        "partition": "training", "verified_ground_truth": False,
        "missing_encoding": {"csv": "empty field", "python": None},
        "source_snapshot": {
            "csv_sha256": csv_hash, "manifest_sha256": manifest_hash,
            "dataset_sha256": source_data.get("dataset_sha256"),
            "method_version": source_data.get("method_version") or source_data["method"]["method_version"],
            "inference_inputs_checked_against_source_manifest": all(
                bool(metadata.get("input_sha256")) for metadata in trips.values()),
        },
        "feature_code_sha256": {name: _sha256(Path(__file__).with_name(name))
                                for name in ("heading.py", "export_heading.py", "pipeline.py")},
        "gps_observation_status": dict(observation_reasons),
        "candidate_heading_status": dict(candidate_reasons),
        "candidate_rows": len(ordered_rows), "trips": trip_summaries,
        "eligible_observations": sum(row["eligible_observations"] for row in trip_summaries.values()),
        "eligible_observations_all_headings_missing": sum(
            row["eligible_observations_all_headings_missing"] for row in trip_summaries.values()),
        "eligible_observations_gps_heading_missing": sum(
            row["eligible_observations_gps_heading_missing"] for row in trip_summaries.values()),
        "eligibility_change": 0,
        "threshold_diagnostics": dict(thresholds), "example_rows": examples,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temp:
        staged = Path(temp) / VERSION
        staged.mkdir()
        csv_path = staged / "training_candidates_heading_v1.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS + [FIELD])
            writer.writeheader()
            writer.writerows(ordered_rows)
        report["export_csv_sha256"] = _sha256(csv_path)
        (staged / "manifest.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.rename(destination)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_root")
    parser.add_argument("input_csv")
    parser.add_argument("source_manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-displacement-m", type=float, default=5.0)
    parser.add_argument("--max-gap-seconds", type=float, default=10.0)
    args = parser.parse_args()
    config = HeadingConfig(args.min_displacement_m, args.max_gap_seconds)
    result = export(args.training_root, args.input_csv, args.source_manifest, args.output_dir, config)
    print(json.dumps({key: value for key, value in result.items() if key != "trips"}, indent=2))
