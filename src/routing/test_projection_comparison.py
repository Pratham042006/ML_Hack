import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import joblib
from sklearn.linear_model import LogisticRegression

from src.data.pipeline import generate_candidates
from src.routing.projection_comparison import compare_trip


class ScorerComparisonTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.trip = Path(directory.name) / "train" / "00000042"
        self.trip.mkdir(parents=True)
        for suffix, text in {
            "track": "0.0002 0 0\n0.0002 0 1\n",
            "nodes": "0 0\n0.001 0\n",
            "arcs": "0 1\n1 0\n",
            "route": "1\n",
        }.items():
            (self.trip / f"00000042.{suffix}").write_text(text, encoding="utf-8")
        self.artifact = Path(directory.name) / "fixture.joblib"
        model = LogisticRegression().fit([[0, 0], [0, 0.1], [0, 0.9], [0, 1]], [0, 0, 1, 1])
        joblib.dump({
            "format_version": 1,
            "feature_names": ["distance_m", "fixture_feature"],
            "estimator": model,
        }, self.artifact)

    def test_paired_scores_reuse_candidates_and_proxy_denominator(self):
        calls = []

        def with_fixture_feature(trip):
            self.assertNotIn("reference_route", trip)
            calls.append(trip["trip_id"])
            candidates = generate_candidates(trip)
            for candidate in candidates:
                candidate["features"]["fixture_feature"] = candidate["arc_id"]
            return candidates

        report = compare_trip(
            self.trip, with_proxy=True, model_artifact=self.artifact,
            candidate_generator=with_fixture_feature,
        )
        self.assertEqual(calls, ["00000042"])
        self.assertEqual(report["candidates"], 4)
        model = report["model_comparison"]
        self.assertEqual(model["artifact_sha256"], hashlib.sha256(self.artifact.read_bytes()).hexdigest())
        self.assertEqual(model["changed_matches_vs_distance_baseline"], {
            "classifier_only": 2, "midpoint": 2, "projected": 2,
        })
        for results in (report["runs"], model["runs"]):
            for run in results.values():
                self.assertTrue(run["validation"]["all_checks_passed"])
                self.assertEqual((run["matched"], run["unmatched"]), (2, 0))
        self.assertEqual(report["offline_proxy"]["agreement"]["projected"]["all_confident"],
                         {"agreements": 0, "denominator": 2, "rate": 0.0})
        self.assertEqual(report["offline_proxy"]["model_agreement"]["projected"]["all_confident"],
                         {"agreements": 2, "denominator": 2, "rate": 1.0})

    def test_unavailable_artifact_feature_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Missing or invalid required model feature: fixture_feature"):
            compare_trip(self.trip, model_artifact=self.artifact)

    def test_baseline_only_requires_no_reference(self):
        (self.trip / "00000042.route").unlink()
        report = compare_trip(self.trip)
        self.assertNotIn("model_comparison", report)
        self.assertNotIn("offline_proxy", report)
        self.assertEqual(report["runs"]["projected"]["matched"], 2)
