"""Geometry cases that matter to the proposed directed routing interface."""

import unittest

from src.data.candidate_projection import project_candidate_positions


class ProjectionTests(unittest.TestCase):
    def test_direction_clamping_zero_length_and_missing_candidate(self):
        trip = {
            "observations": [
                {"observation_index": i, "longitude": longitude, "latitude": 0.0001}
                for i, longitude in [(7, 0.0025), (11, -0.001), (13, 0.011), (17, 0.006)]
            ],
            "network": {"nodes": [[0., 0.], [0.01, 0.]], "arcs": [[0, 1], [1, 0], [0, 0]]},
        }
        candidates = [{"observation_index": i, "arc_id": arc}
                      for i in (7, 11, 13) for arc in range(3)]
        positions = project_candidate_positions(trip, candidates)
        self.assertAlmostEqual(positions[7, 0], 0.25)
        self.assertAlmostEqual(positions[7, 1], 0.75)
        self.assertEqual((positions[11, 0], positions[11, 1]), (0.0, 1.0))
        self.assertEqual((positions[13, 0], positions[13, 1]), (1.0, 0.0))
        self.assertIsNone(positions[7, 2])
        self.assertNotIn((17, 0), positions)
        self.assertIsNone(positions.get((17, 0)))
        self.assertEqual(project_candidate_positions(trip, []), {})
        self.assertTrue(all("projection_fraction" not in row for row in candidates))
