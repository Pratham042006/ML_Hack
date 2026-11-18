from __future__ import annotations

import unittest

from src.routing.decoder import (
    decode_classifier_only,
    decode_with_diagnostics,
)


def observation(index: int, longitude: float, latitude: float, timestamp: float):
    return {
        "observation_index": index,
        "longitude": longitude,
        "latitude": latitude,
        "timestamp": timestamp,
    }


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.network = {
            "nodes": [
                [0.0000, 0.0000],
                [0.0010, 0.0000],
                [0.0020, 0.0000],
                [0.0000, 0.0010],
                [0.0010, 0.0010],
                [0.0020, 0.0010],
            ],
            "arcs": [[0, 1], [1, 2], [3, 4], [4, 5]],
        }

    def test_classifier_only_returns_one_result_per_observation(self):
        observations = [
            observation(0, 0.0, 0.0, 1.0),
            observation(1, 0.001, 0.0, 2.0),
        ]
        candidates = [
            {"observation_index": 0, "arc_id": 0, "score": 0.25},
            {"observation_index": 0, "arc_id": 2, "score": 0.75},
        ]
        result = decode_classifier_only(observations, candidates, self.network)
        self.assertEqual([prediction["arc_id"] for prediction in result], [2, None])

    def test_viterbi_prefers_a_connected_sequence(self):
        observations = [
            observation(0, 0.0005, 0.0005, 1.0),
            observation(1, 0.0015, 0.0005, 2.0),
        ]
        candidates = [
            {"observation_index": 0, "arc_id": 0, "score": 0.60},
            {"observation_index": 0, "arc_id": 2, "score": 0.40},
            {"observation_index": 1, "arc_id": 1, "score": 0.30},
            {"observation_index": 1, "arc_id": 3, "score": 0.70},
        ]
        classifier = decode_classifier_only(observations, candidates, self.network)
        viterbi, diagnostics = decode_with_diagnostics(
            observations, candidates, self.network, max_speed_mps=200.0
        )
        self.assertEqual([item["arc_id"] for item in classifier], [0, 3])
        self.assertEqual([item["arc_id"] for item in viterbi], [2, 3])
        self.assertEqual(diagnostics, [])

    def test_unreachable_transition_is_an_explicit_reset(self):
        observations = [
            observation(0, 0.0005, 0.0, 1.0),
            observation(1, 0.0015, 0.001, 2.0),
        ]
        candidates = [
            {"observation_index": 0, "arc_id": 0, "score": 1.0},
            {"observation_index": 1, "arc_id": 3, "score": 1.0},
        ]
        result, diagnostics = decode_with_diagnostics(
            observations, candidates, self.network, max_speed_mps=200.0
        )
        self.assertEqual([item["arc_id"] for item in result], [0, 3])
        self.assertIn("unreachable_transition_reset", [d["event"] for d in diagnostics])

    def test_missing_candidates_and_large_gap_are_reported(self):
        observations = [
            observation(0, 0.0, 0.0, 1.0),
            observation(1, 0.001, 0.0, 2.0),
            observation(2, 0.0015, 0.0, 100.0),
        ]
        candidates = [
            {"observation_index": 0, "arc_id": 0, "score": 1.0},
            {"observation_index": 2, "arc_id": 1, "score": 1.0},
        ]
        result, diagnostics = decode_with_diagnostics(
            observations, candidates, self.network
        )
        self.assertEqual([item["arc_id"] for item in result], [0, None, 1])
        events = [diagnostic["event"] for diagnostic in diagnostics]
        self.assertIn("missing_candidates", events)

    def test_large_gap_between_candidates_resets_sequence(self):
        observations = [
            observation(0, 0.0, 0.0, 1.0),
            observation(1, 0.001, 0.0, 100.0),
        ]
        candidates = [
            {"observation_index": 0, "arc_id": 0, "score": 1.0},
            {"observation_index": 1, "arc_id": 1, "score": 1.0},
        ]
        _, diagnostics = decode_with_diagnostics(
            observations, candidates, self.network
        )
        self.assertIn("large_gap_reset", [d["event"] for d in diagnostics])

    def test_all_zero_scores_use_uniform_emissions(self):
        observations = [observation(0, 0.0, 0.0, 1.0)]
        candidates = [
            {"observation_index": 0, "arc_id": 0, "score": 0.0},
            {"observation_index": 0, "arc_id": 2, "score": 0.0},
        ]
        result, diagnostics = decode_with_diagnostics(
            observations, candidates, self.network
        )
        self.assertEqual(result[0]["arc_id"], 0)
        self.assertIn("all_zero_scores_uniform", [d["event"] for d in diagnostics])

    def test_invalid_score_raises_clear_error(self):
        observations = [observation(0, 0.0, 0.0, 1.0)]
        candidates = [
            {"observation_index": 0, "arc_id": 0, "score": float("nan")}
        ]
        with self.assertRaisesRegex(ValueError, "finite number"):
            decode_with_diagnostics(observations, candidates, self.network)


if __name__ == "__main__":
    unittest.main()
