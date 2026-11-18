"""Synthetic alignment correctness tests, never real evaluation evidence."""

import copy
import itertools
import unittest

import numpy as np

from src.data.alignment import AlignmentConfig, _decode_costs, align_reference, assess_candidate_coverage


def synthetic_trip():
    return {
        "trip_id": "synthetic",
        "observations": [
            {"observation_index": 0, "longitude": 0.001, "latitude": 0.00001, "timestamp": 1.0},
            {"observation_index": 1, "longitude": 0.002, "latitude": 0.00001, "timestamp": 2.0},
            {"observation_index": 2, "longitude": 0.029, "latitude": 0.00001, "timestamp": 3.0},
        ],
        "network": {"nodes": [[0., 0.], [0.01, 0.], [0.02, 0.], [0.03, 0.]],
                    "arcs": [[0, 1], [1, 2], [2, 3]]},
        "reference_route": [0, 1, 2],
    }


class AlignmentTests(unittest.TestCase):
    def test_dynamic_program_and_marginals_against_exhaustive_paths(self):
        rng = np.random.default_rng(21)
        for width in [1, 2, 4]:
            emissions = rng.uniform(0, 4, size=(4, width))
            penalty = 0.2
            paths = list(itertools.combinations_with_replacement(range(width), 4))
            costs = [sum(emissions[t, p[t]] for t in range(4)) +
                     penalty * sum(max(b - a - 1, 0) for a, b in zip(p, p[1:])) for p in paths]
            path, cost, marginals = _decode_costs(emissions.copy(), penalty)
            self.assertAlmostEqual(cost, min(costs))
            self.assertEqual(tuple(path), paths[int(np.argmin(costs))])
            for t in range(4):
                for j in range(width):
                    expected = min(c for p, c in zip(paths, costs) if p[t] == j)
                    self.assertAlmostEqual(marginals[t, j], expected)

    def test_repeated_observations_and_skipped_arc(self):
        result = align_reference(synthetic_trip())
        self.assertEqual([r["route_position"] for r in result["labels"]], [0, 0, 2])
        self.assertEqual([r["status"] for r in result["labels"]], ["confident"] * 3)

    def test_repeated_arc_occurrences_are_distinct_ambiguous_states(self):
        trip = synthetic_trip()
        trip["network"]["arcs"] = [[0, 1], [1, 0]]
        trip["reference_route"] = [0, 1, 0]
        trip["observations"] = trip["observations"][:1]
        record = align_reference(trip)["labels"][0]
        self.assertEqual(record["status"], "ambiguous")
        self.assertIn(0, record["plausible_route_positions"])
        self.assertIn(2, record["plausible_route_positions"])

    def test_distant_and_empty_reference_unavailable(self):
        trip = synthetic_trip()
        trip["observations"][0]["latitude"] = 0.1
        self.assertEqual(align_reference(trip)["labels"][0]["status"], "unavailable")
        trip["reference_route"] = []
        result = align_reference(trip)
        self.assertEqual(len(result["labels"]), 3)
        self.assertTrue(all(record["arc_id"] is None for record in result["labels"]))

    def test_disconnected_reference_conservatively_unavailable(self):
        trip = synthetic_trip()
        trip["reference_route"] = [0, 2]
        self.assertTrue(all(record["status"] == "unavailable" for record in align_reference(trip)["labels"]))

    def test_candidate_misses_cannot_change_labels(self):
        trip = synthetic_trip()
        original = copy.deepcopy(trip)
        alignment = align_reference(trip)
        missing = assess_candidate_coverage(alignment, [])
        covered = assess_candidate_coverage(alignment, [
            {"observation_index": record["observation_index"], "arc_id": record["arc_id"]}
            for record in alignment["labels"]])
        self.assertEqual(trip, original)
        self.assertEqual(missing["counts"]["candidate_misses"], 3)
        self.assertEqual(covered["counts"]["candidate_misses"], 0)
        for result in [missing, covered]:
            for before, after in zip(alignment["labels"], result["labels"]):
                self.assertTrue(all(after[key] == value for key, value in before.items()))

    def test_gaps_and_resource_limit_qualify_labels(self):
        trip = synthetic_trip()
        trip["observations"][1]["timestamp"] = 20
        trip["observations"][2]["timestamp"] = 21
        records = align_reference(trip)["labels"]
        self.assertEqual([r["status"] for r in records[:2]], ["ambiguous", "ambiguous"])
        limited = align_reference(trip, AlignmentConfig(max_cells=1))
        self.assertTrue(all(r["status"] == "unavailable" for r in limited["labels"]))

    def test_ambiguous_and_unavailable_never_count_as_candidate_misses(self):
        alignment = {"labels": [
            {"observation_index": 0, "status": "ambiguous", "arc_id": 0},
            {"observation_index": 1, "status": "unavailable", "arc_id": None},
        ]}
        result = assess_candidate_coverage(alignment, [])
        self.assertEqual(result["counts"]["candidate_misses"], 0)
        self.assertIsNone(result["counts"]["candidate_miss_rate"])
        self.assertTrue(all(record["candidate_miss"] is None for record in result["labels"]))


if __name__ == "__main__":
    unittest.main()
