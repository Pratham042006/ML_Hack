"""Synthetic interface/validation tests; these are not evaluation results."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import copy
import math

import numpy as np

from src.data.pipeline import load_trip, generate_candidates, _point_segment_distances


class CandidateTests(unittest.TestCase):
    @staticmethod
    def trip():
        return {
            "trip_id": "synthetic",
            "observations": [
                {"observation_index": 7, "longitude": 0.002, "latitude": 0.0001, "timestamp": 1.0},
                {"observation_index": 9, "longitude": 0.5, "latitude": 0.5, "timestamp": 2.0},
            ],
            "network": {"nodes": [[0.0, 0.0], [0.01, 0.0]], "arcs": [[0, 1], [1, 0]]},
        }

    def test_segment_interior_endpoint_and_degenerate_distance(self):
        distances = _point_segment_distances(
            np.array([3., 4.]),
            np.array([[0., 0.], [0., 0.], [0., 0.]]),
            np.array([[10., 0.], [1., 0.], [0., 0.]]),
        )
        np.testing.assert_allclose(distances, [4, math.sqrt(20), 5])

    def test_long_segments_missing_observations_and_reference_independence(self):
        trip = self.trip()
        original = copy.deepcopy(trip)
        candidates = generate_candidates(trip)
        self.assertEqual(trip, original)
        self.assertEqual([(c["observation_index"], c["arc_id"]) for c in candidates], [(7, 0), (7, 1)])
        # The observation is far from both endpoints and the segment midpoint.
        # Its perpendicular distance to the interior is about 11.06 metres.
        for candidate in candidates:
            self.assertAlmostEqual(candidate["features"]["distance_m"], 11.057, delta=0.02)
            self.assertIsInstance(candidate["arc_id"], int)
        self.assertEqual(candidates[1]["geometry"], [[0.01, 0.0], [0.0, 0.0]])
        trip["reference_route"] = object()  # Must never be inspected.
        self.assertEqual(generate_candidates(trip), candidates)

    def test_candidate_cap_ties_and_unique_pairs(self):
        trip = self.trip()
        trip["network"]["arcs"] = [[0, 1] for _ in range(12)]
        candidates = generate_candidates(trip)
        self.assertEqual([c["arc_id"] for c in candidates], list(range(8)))
        self.assertEqual(len({(c["observation_index"], c["arc_id"]) for c in candidates}), 8)

    def test_no_fabricated_match_and_empty_inputs(self):
        trip = self.trip()
        trip["observations"] = trip["observations"][1:]
        self.assertEqual(generate_candidates(trip), [])
        trip["network"]["arcs"] = []
        self.assertEqual(generate_candidates(trip), [])

    def test_invalid_candidate_inputs(self):
        for field, value in [("observation_index", True), ("longitude", float("nan")), ("latitude", 85)]:
            with self.subTest(field=field):
                trip = self.trip()
                trip["observations"][0][field] = value
                with self.assertRaises(ValueError):
                    generate_candidates(trip)
        trip = self.trip()
        trip["network"]["arcs"] = [[0, 2]]
        with self.assertRaises(ValueError):
            generate_candidates(trip)

    def test_zero_length_arc_is_finite(self):
        trip = self.trip()
        trip["network"] = {"nodes": [[0.002, 0.0001]], "arcs": [[0, 0]]}
        candidates = generate_candidates(trip)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["features"]["distance_m"], 0.0)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.trip_dir = Path(self.temp.name) / "synthetic"
        self.trip_dir.mkdir()
        self.write("track", "8 49 1\n8.001 49 3\n")
        self.write("nodes", "8 49\n8.001 49\n")
        self.write("arcs", "0 1\n")

    def write(self, extension, text):
        (self.trip_dir / f"synthetic.{extension}").write_text(text, encoding="utf-8")

    def test_inference_never_opens_reference_and_preserves_order(self):
        self.write("route", "not a valid reference\n")
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            if path.suffix == ".route":
                raise AssertionError("Inference attempted to read reference")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            trip = load_trip(self.trip_dir)
        self.assertNotIn("reference_route", trip)
        self.assertEqual([o["observation_index"] for o in trip["observations"]], [0, 1])
        self.assertEqual([o["timestamp"] for o in trip["observations"]], [1.0, 3.0])
        self.assertEqual(trip["network"]["arcs"], [[0, 1]])

    def test_reference_is_separate_sequence(self):
        self.write("route", "0\n")
        trip = load_trip(self.trip_dir, include_reference=True)
        self.assertEqual(trip["reference_route"], [0])
        self.assertEqual(len(trip["observations"]), 2)

    def test_invalid_values_fail_without_cleaning(self):
        for extension, contents in [
            ("track", "8 49 nan\n"),
            ("track", "8 49 2\n8 49 1\n"),
            ("track", "8 49 1\n\n8 49 2\n"),
            ("track", "8 91 1\n"),
            ("arcs", "0 2\n"),
            ("arcs", "-1 0\n"),
        ]:
            with self.subTest(extension=extension, contents=contents):
                self.write("track", "8 49 1\n8.001 49 3\n")
                self.write("arcs", "0 1\n")
                self.write(extension, contents)
                with self.assertRaises(ValueError):
                    load_trip(self.trip_dir)

    def test_invalid_reference_id(self):
        self.write("route", "1\n")
        with self.assertRaises(ValueError):
            load_trip(self.trip_dir, include_reference=True)

    def test_missing_required_file(self):
        (self.trip_dir / "synthetic.nodes").unlink()
        with self.assertRaises(FileNotFoundError):
            load_trip(self.trip_dir)


if __name__ == "__main__":
    unittest.main()
