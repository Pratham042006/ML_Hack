import unittest

from src.routing.continuity import audit_continuity
from src.routing.decoder import DirectedRoadGraph


class ContinuityAuditTests(unittest.TestCase):
    def test_short_limit_does_not_poison_wider_or_unbounded_search(self):
        graph = DirectedRoadGraph({
            "nodes": [[0, 0], [0.001, 0], [0.002, 0], [0.003, 0]],
            "arcs": [[0, 1], [1, 2], [2, 3]],
        })
        self.assertEqual(graph.candidate_travel_distance_m(0, 2, limit_m=150), float("inf"))
        distance = graph.candidate_travel_distance_m(0, 2, limit_m=250)
        self.assertAlmostEqual(distance, 222.39, places=2)
        self.assertEqual(graph.candidate_travel_distance_m(0, 2), distance)
        self.assertEqual(graph.candidate_travel_distance_m(0, 2, limit_m=150), float("inf"))
        self.assertEqual(graph.candidate_travel_distance_m(2, 0), float("inf"))

    def test_adjacent_pairs_have_distinct_failure_reasons(self):
        graph = DirectedRoadGraph({
            "nodes": [[0, 0], [0.001, 0], [0.002, 0], [0.003, 0]],
            "arcs": [[0, 1], [1, 2], [3, 2]],
        })
        times = [0, 1, 2, 3, 3, 40, 41, 42]
        arcs = [0, 0, 1, 2, 2, 2, None, 0]
        observations = [
            {"observation_index": 10 + 2 * i, "timestamp": timestamp}
            for i, timestamp in enumerate(times)
        ]
        counts, transitions = audit_continuity(
            observations, [{"arc_id": arc} for arc in arcs], graph
        )
        self.assertEqual(counts, {
            "plausible": 1, "over_speed_cap": 1, "disconnected": 1,
            "nonpositive_timestamp": 1, "large_gap": 1, "unmatched_endpoint": 2,
        })
        self.assertEqual(transitions[12]["travel_distance_m"], 0.0)
        self.assertGreater(transitions[14]["travel_distance_m"], 60.0)
        self.assertIsNone(transitions[16]["travel_distance_m"])
