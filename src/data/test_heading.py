"""Check heading geometry and that a heading export preserves eligibility."""

import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.data.align_trips import _sha256
from src.data.export_heading import export
from src.data.heading import HeadingConfig, TripHeading


def observation(index, longitude, latitude, timestamp):
    return dict(observation_index=index, longitude=longitude, latitude=latitude, timestamp=timestamp)


class HeadingTests(unittest.TestCase):
    def test_direction_and_wrap_around(self):
        trip = {
            "observations": [observation(4, 0, 0, 0), observation(9, 0.0000175, 0.001, 1)],
            "network": {"nodes": [[0, 0], [-0.0000175, 0.001], [0, 0.001], [0.001, 0.001]],
                        "arcs": [[0, 1], [1, 0], [2, 3], [0, 0]]},
        }
        heading = TripHeading(trip)
        agreement, reason = heading.agreement(9, 0)
        self.assertIsNone(reason)
        self.assertAlmostEqual(agreement, math.cos(math.radians(2)), delta=0.00005)
        self.assertAlmostEqual(heading.agreement(9, 1)[0], -agreement)
        self.assertEqual(heading.agreement(9, 3), (None, "zero_length_arc"))
        self.assertEqual(heading.agreement(4, 0), (None, "first_observation"))

        trip["observations"] = [observation(0, 0, 0, 0), observation(1, 0, .001, 1)]
        self.assertAlmostEqual(TripHeading(trip).agreement(1, 2)[0], 0)

    def test_missing_movement_and_time(self):
        trip = {
            "observations": [observation(i, lon, 0, time) for i, (lon, time) in enumerate([
                (0, 0), (0, 1), (0.00001, 2), (0.001, 2), (0.002, 13), (0.003, 23),
            ])],
            "network": {"nodes": [[0, 0], [0.004, 0]], "arcs": [[0, 1]]},
        }
        heading = TripHeading(trip)
        self.assertEqual([heading.agreement(i, 0)[1] for i in range(6)], [
            "first_observation", "insufficient_displacement", "insufficient_displacement",
            "nonpositive_or_invalid_time_gap", "large_time_gap", None,
        ])
        self.assertEqual(heading.agreement(5, 0), (1., None))
        with self.assertRaises(ValueError):
            HeadingConfig(min_displacement_m=0)

    def test_export_keeps_unavailable_rows_and_reads_no_route(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            training = root / "train"
            trip = training / "00000001"
            trip.mkdir(parents=True)
            for suffix, contents in {
                "track": "0 0 0\n0.001 0 1\n", "nodes": "0 0\n0.002 0\n", "arcs": "0 1\n1 0\n",
            }.items():
                (trip / f"00000001.{suffix}").write_text(contents, encoding="utf-8")
            old = root / "distance.csv"
            old.write_text("trip_id,observation_index,arc_id,distance_m,label\n"
                           "00000001,0,0,0.000,1\n00000001,1,0,0.0,1\n00000001,1,1,0,0\n",
                           encoding="utf-8")
            original = old.read_bytes()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "partition": "training", "verified_ground_truth": False,
                "combined_sha256": _sha256(old), "method_version": "synthetic",
                "trip_counts": {"00000001": {"total_observations": 2, "training_eligible": 2}},
            }), encoding="utf-8")
            output = root / "heading-v1"
            report = export(training, old, manifest, output)
            self.assertEqual(old.read_bytes(), original)
            self.assertEqual(report["eligibility_change"], 0)
            self.assertEqual(report["eligible_observations_all_headings_missing"], 1)
            with (output / "training_candidates_heading_v1.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["heading_agreement"] for row in rows], ["", "1.0", "-1.0"])
            self.assertEqual([row["distance_m"] for row in rows], ["0.000", "0.0", "0"])
            self.assertEqual([row["label"] for row in rows], ["1", "1", "0"])
