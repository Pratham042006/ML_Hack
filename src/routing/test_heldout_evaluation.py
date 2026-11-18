from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.routing.heldout_evaluation import DECODERS, STUDENT_TEST_TRIPS, evaluate_student_test, pool_results
from src.routing.projection_comparison import load_evaluation_artifact


class HeldOutTests(unittest.TestCase):
    def test_paired_batch_with_nullable_heading_fixture(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "student_test"
            for trip_id in STUDENT_TEST_TRIPS:
                folder = root / trip_id
                folder.mkdir(parents=True)
                for suffix, contents in {
                    "track": "0.0002 0 0\n0.0003 0 1\n",
                    "nodes": "0 0\n0.001 0\n", "arcs": "0 1\n1 0\n", "route": "0\n",
                }.items():
                    (folder / f"{trip_id}.{suffix}").write_text(contents, encoding="utf-8")
            model_path = Path(directory) / "heading-fixture.joblib"
            training_sha = "a" * 64
            estimator = make_pipeline(SimpleImputer(), LogisticRegression()).fit(
                [[0, -1], [1, -1], [0, 1], [1, 1]], [0, 0, 1, 1],
            )
            joblib.dump({
                "format_version": 1, "feature_names": ["distance_m", "heading_agreement"],
                "nullable_features": ["heading_agreement"], "estimator": estimator,
                "training_csv_sha256": training_sha,
            }, model_path)
            artifact_sha = sha256(model_path.read_bytes()).hexdigest()
            report = evaluate_student_test(
                root, artifact=model_path, artifact_sha=artifact_sha,
                training_sha=training_sha, with_proxy=True,
            )
            self.assertEqual(report["mode"], "paired_heading")
            self.assertEqual(report["artifact"]["sha256"], artifact_sha)
            self.assertEqual(report["pooled"]["proxy_label_counts"]["confident"], 20)
            for decoder in DECODERS:
                self.assertEqual(report["pooled"]["heading_artifact"][decoder]["matched"], 20)

    def test_proxy_pool_sums_counts_instead_of_averaging_percentages(self):
        reports = []
        for hits, confident, misses in ((1, 1, 0), (3, 9, 2)):
            agreement = {
                "all_confident": {"agreements": hits, "denominator": confident},
                "confident_with_label_in_candidates": {
                    "agreements": hits, "denominator": confident - misses,
                },
            }
            run = {
                "matched": 10, "unmatched": 1, "decode_seconds": 0.5,
                "midpoint_audit": {"disconnected": 2}, "projected_audit": {"disconnected": 1},
                "diagnostics": {"event_counts": {"large_gap_reset": 1}, "matched_blocks": [{}]},
            }
            reports.append({
                "observations": 11, "candidates": 12, "observations_without_candidates": 1,
                "runs": dict.fromkeys(DECODERS, run),
                "model_comparison": {"runs": dict.fromkeys(DECODERS, run)},
                "offline_proxy": {
                    "label_counts": {
                        "confident": confident, "candidate_misses": misses,
                        "candidate_miss_denominator": confident, "candidate_miss_rate": 0,
                    },
                    "agreement": dict.fromkeys(DECODERS, agreement),
                    "model_agreement": dict.fromkeys(DECODERS, agreement),
                },
            })
        pooled = pool_results(reports, paired=True, with_proxy=True)
        result = pooled["heading_artifact"]["projected"]
        self.assertEqual(result["proxy_agreement"]["all_confident"],
                         {"agreements": 4, "denominator": 10, "rate": 0.4})
        self.assertEqual(result["proxy_agreement"]["confident_with_label_in_candidates"]["rate"], 0.5)
        self.assertEqual(result["reset_events"], {"large_gap_reset": 2})
        self.assertEqual(result["matched_blocks"], 2)
        self.assertEqual(pooled["proxy_label_counts"]["candidate_miss_rate"], 0.2)

    def test_split_and_hash_guards(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "expected the supplied student_test"):
                evaluate_student_test(root / "evaluator")
            with self.assertRaisesRegex(ValueError, "require trusted artifact"):
                evaluate_student_test(root / "student_test", artifact=root / "model.joblib")
            file = root / "model.joblib"
            file.write_bytes(b"untrusted fixture, must not be loaded")
            with self.assertRaisesRegex(ValueError, "artifact does not match"):
                load_evaluation_artifact(file, "0" * 64)
