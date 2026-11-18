"""Compare two PathSnap candidate exports without rewriting either snapshot."""

import argparse
from collections import Counter
import csv
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

from src.data.align_trips import _sha256
from src.data.batch_prepare import CSV_FIELDS


def _read_export(path):
    rows = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if sorted(reader.fieldnames or []) != sorted(CSV_FIELDS):
            raise ValueError(f"{path}: expected exactly the fields {CSV_FIELDS}")
        for row in reader:
            try:
                trip = row["trip_id"]
                key = (trip, int(row["observation_index"]), int(row["arc_id"]))
                distance, label = Decimal(row["distance_m"]), int(row["label"])
                if (not trip or min(key[1:]) < 0 or not distance.is_finite()
                        or distance < 0 or label not in (0, 1) or None in row):
                    raise ValueError("invalid key, distance or label")
            except (TypeError, ValueError, InvalidOperation) as exc:
                raise ValueError(f"{path}:{reader.line_num}: {exc}") from exc
            if key in rows:
                raise ValueError(f"{path}:{reader.line_num}: duplicate candidate {key}")
            rows[key] = distance, label
    return rows


def compare_exports(left_path, right_path, expected_right_sha256=None):
    """Classify byte, row-order, key, exact numeric, and label differences."""
    left_hash, right_hash = _sha256(Path(left_path)), _sha256(Path(right_path))
    if expected_right_sha256 and right_hash != expected_right_sha256.lower():
        raise ValueError(f"Right file SHA-256 is {right_hash}, expected {expected_right_sha256}")
    left, right = _read_export(left_path), _read_export(right_path)
    left_only, right_only = left.keys() - right.keys(), right.keys() - left.keys()
    affected = {name: Counter() for name in ("left_only", "right_only", "distance", "label")}
    affected["left_only"].update(key[0] for key in left_only)
    affected["right_only"].update(key[0] for key in right_only)
    distance_changes = label_changes = 0
    max_distance_delta = Decimal(0)
    changed_examples = []
    for key, (left_distance, left_label) in left.items():
        if key not in right:
            continue
        right_distance, right_label = right[key]
        distance_changed, label_changed = left_distance != right_distance, left_label != right_label
        distance_changes += distance_changed
        label_changes += label_changed
        if distance_changed:
            affected["distance"][key[0]] += 1
            max_distance_delta = max(max_distance_delta, abs(left_distance - right_distance))
        if label_changed:
            affected["label"][key[0]] += 1
        if (distance_changed or label_changed) and len(changed_examples) < 10:
            changed_examples.append({"key": key, "left_distance_m": str(left_distance),
                                     "right_distance_m": str(right_distance),
                                     "left_label": left_label, "right_label": right_label})
    same_values = not (left_only or right_only or distance_changes or label_changes)
    return {
        "left_sha256": left_hash, "right_sha256": right_hash,
        "same_bytes": left_hash == right_hash, "same_candidate_rows": same_values,
        "same_key_order": left.keys() == right.keys() and all(a == b for a, b in zip(left, right)),
        "left_rows": len(left), "right_rows": len(right),
        "left_only": len(left_only), "right_only": len(right_only),
        "distance_changes": distance_changes, "label_changes": label_changes,
        "max_distance_delta_m": str(max_distance_delta),
        "differences_by_type_and_trip": affected,
        "left_only_examples": sorted(left_only)[:10],
        "right_only_examples": sorted(right_only)[:10],
        "changed_examples": changed_examples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--expected-right-sha256")
    args = parser.parse_args()
    report = compare_exports(args.left, args.right, args.expected_right_sha256)
    print(json.dumps(report, indent=2))
