from functools import partial
import math
import unittest

from src.routing.decoder import DirectedRoadGraph, decode_with_diagnostics
from src.routing.map_diagnostics import matched_blocks
from src.routing.projected import ProjectedDistance, candidate_fractions, decode_projected


class ProjectedDistanceTests(unittest.TestCase):
    def setUp(self):
        self.network = {
            "nodes": [[0, 0], [0.001, 0], [0.002, 0], [0.003, 0], [0.004, 0]],
            "arcs": [[0, 1], [1, 2], [2, 3], [1, 0], [4, 4]],
        }
        self.graph = DirectedRoadGraph(self.network)

    def test_projection_clamps_and_respects_direction_and_degenerate_arcs(self):
        trip = {"network": self.network, "observations": [
            {"observation_index": 7, "longitude": 0.00025, "latitude": 0.00005},
        ]}
        candidates = [{"observation_index": 7, "arc_id": a} for a in (0, 1, 3, 4)]
        fractions = candidate_fractions(trip, candidates)
        self.assertAlmostEqual(fractions[7, 0], 0.25)
        self.assertAlmostEqual(fractions[7, 3], 0.75)
        self.assertEqual(fractions[7, 1], 0.0)
        self.assertIsNone(fractions[7, 4])
        self.assertNotIn("projection_fraction", candidates[0])

    def test_forward_jitter_and_larger_backward_shift(self):
        distance = partial(ProjectedDistance({
            (0, 0): 0.8, (1, 0): 0.9, (2, 0): 0.77, (3, 0): 0.3,
            (0, 1): 0.8, (3, 1): 0.3,
        }), delta_seconds=1)
        length = self.graph.arcs[0].length_m
        self.assertAlmostEqual(distance(self.graph, 0, 0, 1, 0), 0.1 * length)
        self.assertEqual(distance(self.graph, 0, 0, 2, 0), 0.0)
        self.assertAlmostEqual(distance(self.graph, 0, 0, 2, 0, delta_seconds=11), 1.97 * length)
        self.assertAlmostEqual(distance(self.graph, 0, 0, 3, 0), 1.5 * length)
        self.assertEqual(distance(self.graph, 0, 1, 3, 1), math.inf)

    def test_distinct_arcs_preserve_direction_skips_and_cutoff(self):
        d = partial(ProjectedDistance({(0, 0): 0.9, (1, 2): 0.2, (2, 3): 0.1}), delta_seconds=1)
        length = self.graph.arcs[0].length_m
        self.assertAlmostEqual(d(self.graph, 0, 0, 1, 2), 1.3 * length)
        self.assertEqual(d(self.graph, 0, 0, 1, 2, limit_m=100), math.inf)
        self.assertEqual(d(self.graph, 1, 2, 0, 0), math.inf)
        self.assertAlmostEqual(d(self.graph, 0, 0, 2, 3), 0.2 * length)

    def test_missing_or_null_falls_back_and_invalid_fraction_fails(self):
        d = partial(ProjectedDistance({(0, 0): 0.9, (1, 1): None}), delta_seconds=1)
        self.assertEqual(d(self.graph, 0, 0, 1, 1), self.graph.candidate_travel_distance_m(0, 1))
        self.assertEqual(d(self.graph, 0, 0, 2, 0), 0)
        for invalid in (True, 1.01, float("nan"), "0.5"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ProjectedDistance({(0, 0): invalid})

    def test_near_junction_transition_removes_midpoint_reset_only(self):
        observations = [
            {"observation_index": i, "timestamp": t, "longitude": lon, "latitude": 0.0}
            for i, t, lon in [(0, 0, 0.0009), (1, 1, 0.0011), (2, 40, 0.0012), (3, 41, 0.004)]
        ]
        scored = [
            {"observation_index": i, "arc_id": arc, "score": 1.0}
            for i, arc in [(0, 0), (1, 1), (2, 1), (3, 4)]
        ]
        midpoint, midpoint_events = decode_with_diagnostics(observations, scored, self.network)
        d = ProjectedDistance({(0, 0): 0.9, (1, 1): 0.1, (2, 1): 0.2, (3, 4): None})
        projected, events = decode_projected(observations, scored, self.network, distance=d)
        self.assertEqual(midpoint, projected)
        self.assertEqual([event["observation_index"] for event in midpoint_events], [1, 2, 3])
        self.assertEqual([event["observation_index"] for event in events], [2, 3])
        self.assertEqual([block["matched_count"] for block in matched_blocks(projected, events)], [2, 1, 1])
