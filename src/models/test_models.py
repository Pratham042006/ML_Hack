"""Meaningful contract checks; all samples below are synthetic."""
import math
import unittest

from src.models.inference import score_candidates


def candidate(obs, arc, distance):
    return {"observation_index": obs, "arc_id": arc,
            "features": {"distance_m": distance}, "geometry": [[0.0, 0.0], [1.0, 1.0]]}


class BaselineTests(unittest.TestCase):
    def test_group_normalization_order_and_input_unchanged(self):
        rows = [candidate(1, 2, 20.0), candidate(0, 3, 0.0), candidate(1, 4, 2.0)]
        result = score_candidates(rows)
        self.assertEqual([r["arc_id"] for r in result], [2, 3, 4])
        self.assertAlmostEqual(result[0]["score"] + result[2]["score"], 1.0)
        self.assertEqual(result[1]["score"], 1.0)
        self.assertGreater(result[2]["score"], result[0]["score"])
        self.assertNotIn("score", rows[0])

    def test_far_candidates_and_empty_input(self):
        scores = score_candidates([candidate(0, 0, 1e300), candidate(0, 1, 1e300)])
        self.assertEqual([r["score"] for r in scores], [0.5, 0.5])
        self.assertEqual(score_candidates([]), [])

    def test_reject_invalid_and_duplicate_candidates(self):
        for distance in (-1, math.inf, math.nan, "2"):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                score_candidates([candidate(0, 0, distance)])
        with self.assertRaises(ValueError):
            score_candidates([candidate(0, 0, 1), candidate(0, 0, 2)])

    def test_artifact_interface_and_required_features(self):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression().fit([[0], [1], [10], [11]], [1, 1, 0, 0])
        artifact = {"format_version": 1, "feature_names": ["distance_m"], "estimator": model}
        result = score_candidates([candidate(0, 0, 1), candidate(0, 1, 10)], artifact)
        self.assertAlmostEqual(sum(r["score"] for r in result), 1)
        self.assertGreater(result[0]["score"], result[1]["score"])
        artifact["feature_names"] = ["not_available"]
        with self.assertRaisesRegex(ValueError, "not_available"):
            score_candidates([candidate(0, 0, 1)], artifact)


if __name__ == "__main__":
    unittest.main()
