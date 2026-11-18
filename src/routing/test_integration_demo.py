from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from src.data.pipeline import generate_candidates, load_trip
from src.models.inference import score_candidates
from src.routing.integration_demo import (
    build_integration_report,
    run_integration_demo,
    validate_pipeline_outputs,
)


def make_observation(index: int, lon: float, lat: float, ts: float) -> dict[str, Any]:
    return {
        "observation_index": index,
        "longitude": lon,
        "latitude": lat,
        "timestamp": ts,
    }


class IntegrationValidationTests(unittest.TestCase):
    def setUp(self):
        self.observations = [
            make_observation(0, 0.0, 0.0, 1.0),
            make_observation(1, 0.001, 0.0, 2.0),
            make_observation(2, 0.002, 0.0, 3.0),
        ]
        self.candidates = [
            {"observation_index": 0, "arc_id": 10, "features": {"distance_m": 5.0}, "geometry": [[0.0, 0.0], [0.001, 0.0]]},
            {"observation_index": 0, "arc_id": 11, "features": {"distance_m": 12.0}, "geometry": [[0.0, 0.0], [0.0, 0.001]]},
            {"observation_index": 1, "arc_id": 12, "features": {"distance_m": 3.0}, "geometry": [[0.001, 0.0], [0.002, 0.0]]},
            # Observation 2 has no candidate
        ]
        self.valid_predictions = [
            {"observation_index": 0, "arc_id": 10, "geometry": [[0.0, 0.0], [0.001, 0.0]]},
            {"observation_index": 1, "arc_id": 12, "geometry": [[0.001, 0.0], [0.002, 0.0]]},
            {"observation_index": 2, "arc_id": None, "geometry": []},
        ]

    def test_validation_passes_on_valid_output(self):
        res = validate_pipeline_outputs(self.observations, self.candidates, self.valid_predictions)
        self.assertTrue(res["all_checks_passed"])
        self.assertEqual(res["error_count"], 0)
        self.assertEqual(res["errors"], [])
        self.assertTrue(res["checks"]["exactly_one_prediction_per_observation"])
        self.assertTrue(res["checks"]["input_order_preserved"])
        self.assertTrue(res["checks"]["selected_arc_belongs_to_candidate_set"])
        self.assertTrue(res["checks"]["unmatched_outputs_have_null_arc_and_empty_geometry"])
        self.assertTrue(res["checks"]["matched_outputs_have_nonempty_geometry"])

    def test_validation_fails_on_count_mismatch(self):
        truncated = self.valid_predictions[:2]
        res = validate_pipeline_outputs(self.observations, self.candidates, truncated)
        self.assertFalse(res["all_checks_passed"])
        self.assertFalse(res["checks"]["exactly_one_prediction_per_observation"])
        self.assertIn("Prediction count (2) does not match observation count (3)", res["errors"][0])

    def test_validation_fails_on_order_mismatch(self):
        swapped = [self.valid_predictions[1], self.valid_predictions[0], self.valid_predictions[2]]
        res = validate_pipeline_outputs(self.observations, self.candidates, swapped)
        self.assertFalse(res["all_checks_passed"])
        self.assertFalse(res["checks"]["input_order_preserved"])
        self.assertTrue(any("index mismatch" in e for e in res["errors"]))

    def test_validation_fails_when_selected_arc_not_in_candidate_set(self):
        invalid_arc = [
            {"observation_index": 0, "arc_id": 999, "geometry": [[0.0, 0.0], [0.001, 0.0]]},
            self.valid_predictions[1],
            self.valid_predictions[2],
        ]
        res = validate_pipeline_outputs(self.observations, self.candidates, invalid_arc)
        self.assertFalse(res["all_checks_passed"])
        self.assertFalse(res["checks"]["selected_arc_belongs_to_candidate_set"])
        self.assertTrue(any("selected arc_id 999 not in candidate set" in e for e in res["errors"]))

    def test_validation_fails_when_unmatched_has_nonempty_geometry(self):
        bad_unmatched = [
            self.valid_predictions[0],
            self.valid_predictions[1],
            {"observation_index": 2, "arc_id": None, "geometry": [[0.0, 0.0], [0.001, 0.0]]},
        ]
        res = validate_pipeline_outputs(self.observations, self.candidates, bad_unmatched)
        self.assertFalse(res["all_checks_passed"])
        self.assertFalse(res["checks"]["unmatched_outputs_have_null_arc_and_empty_geometry"])
        self.assertTrue(any("unmatched prediction must have geometry=[]" in e for e in res["errors"]))

    def test_validation_fails_when_matched_has_empty_geometry(self):
        bad_matched = [
            {"observation_index": 0, "arc_id": 10, "geometry": []},
            self.valid_predictions[1],
            self.valid_predictions[2],
        ]
        res = validate_pipeline_outputs(self.observations, self.candidates, bad_matched)
        self.assertFalse(res["all_checks_passed"])
        self.assertFalse(res["checks"]["matched_outputs_have_nonempty_geometry"])


class IntegrationReportingTests(unittest.TestCase):
    def setUp(self):
        self.trip = {
            "trip_id": "test_trip_01",
            "observations": [
                make_observation(0, 0.0, 0.0, 1.0),
                make_observation(1, 0.001, 0.0, 2.0),
                make_observation(2, 0.002, 0.0, 100.0),
                make_observation(3, 0.003, 0.0, 101.0),
            ],
            "network": {
                "nodes": [[0.0, 0.0], [0.001, 0.0], [0.002, 0.0], [0.003, 0.0]],
                "arcs": [[0, 1], [1, 2], [2, 3]],
            },
        }
        self.candidates = [
            {"observation_index": 0, "arc_id": 0, "features": {"distance_m": 5.0}, "geometry": [[0.0, 0.0], [0.001, 0.0]]},
            {"observation_index": 1, "arc_id": 1, "features": {"distance_m": 5.0}, "geometry": [[0.001, 0.0], [0.002, 0.0]]},
            {"observation_index": 2, "arc_id": 1, "features": {"distance_m": 8.0}, "geometry": [[0.001, 0.0], [0.002, 0.0]]},
            {"observation_index": 2, "arc_id": 2, "features": {"distance_m": 2.0}, "geometry": [[0.002, 0.0], [0.003, 0.0]]},
        ]
        # Exercise real Pratham candidate scoring interface
        self.scored = score_candidates(self.candidates)
        self.predictions = [
            {"observation_index": 0, "arc_id": 0, "geometry": [[0.0, 0.0], [0.001, 0.0]]},
            {"observation_index": 1, "arc_id": 1, "geometry": [[0.001, 0.0], [0.002, 0.0]]},
            {"observation_index": 2, "arc_id": 2, "geometry": [[0.002, 0.0], [0.003, 0.0]]},
            {"observation_index": 3, "arc_id": None, "geometry": []},
        ]
        self.classifier_predictions = [
            {"observation_index": 0, "arc_id": 0, "geometry": [[0.0, 0.0], [0.001, 0.0]]},
            {"observation_index": 1, "arc_id": 1, "geometry": [[0.001, 0.0], [0.002, 0.0]]},
            {"observation_index": 2, "arc_id": 1, "geometry": [[0.001, 0.0], [0.002, 0.0]]},
            {"observation_index": 3, "arc_id": None, "geometry": []},
        ]
        self.diagnostics = [
            {"observation_index": 2, "event": "large_gap_reset", "delta_seconds": 98.0},
            {"observation_index": 3, "event": "missing_candidates"},
        ]
        self.timings = {
            "load_trip_s": 0.001,
            "generate_candidates_s": 0.005,
            "score_candidates_s": 0.002,
            "decode_s": 0.003,
            "total_pipeline_s": 0.011,
        }

    def test_report_structure_and_disclaimers(self):
        report = build_integration_report(
            trip=self.trip,
            candidates=self.candidates,
            scored=self.scored,
            predictions=self.predictions,
            classifier_predictions=self.classifier_predictions,
            diagnostics=self.diagnostics,
            timings=self.timings,
            trip_path="dummy/path",
        )
        self.assertIn("scorer", report["disclaimers"])
        self.assertIn("distance baseline", report["disclaimers"]["scorer"])
        self.assertIn("not a trained model", report["disclaimers"]["scorer"])
        self.assertIn("reference_labels", report["disclaimers"])
        self.assertIn("not loaded", report["disclaimers"]["reference_labels"])
        self.assertIn("coverage only", report["disclaimers"]["metrics_interpretation"])

        encoded = json.dumps(report)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["trip_id"], "test_trip_01")

    def test_coverage_reporting_not_accuracy(self):
        report = build_integration_report(
            trip=self.trip,
            candidates=self.candidates,
            scored=self.scored,
            predictions=self.predictions,
            classifier_predictions=self.classifier_predictions,
            diagnostics=self.diagnostics,
            timings=self.timings,
            trip_path="dummy/path",
        )
        cov = report["coverage"]
        self.assertEqual(cov["total_observations"], 4)
        self.assertEqual(cov["matched_observations"], 3)
        self.assertEqual(cov["unmatched_observations"], 1)
        self.assertEqual(cov["coverage_rate"], 0.75)
        self.assertIn("not", cov["note"].lower())
        self.assertIn("accuracy", cov["note"].lower())

    def test_candidate_counts_reporting(self):
        report = build_integration_report(
            trip=self.trip,
            candidates=self.candidates,
            scored=self.scored,
            predictions=self.predictions,
            classifier_predictions=self.classifier_predictions,
            diagnostics=self.diagnostics,
            timings=self.timings,
            trip_path="dummy/path",
        )
        counts = report["candidate_counts"]
        self.assertEqual(counts["total_candidates"], 4)
        self.assertEqual(counts["total_scored_candidates"], 4)
        self.assertEqual(counts["observations_with_candidates"], 3)
        self.assertEqual(counts["observations_without_candidates"], 1)

    def test_classifier_vs_viterbi_reporting(self):
        report = build_integration_report(
            trip=self.trip,
            candidates=self.candidates,
            scored=self.scored,
            predictions=self.predictions,
            classifier_predictions=self.classifier_predictions,
            diagnostics=self.diagnostics,
            timings=self.timings,
            trip_path="dummy/path",
        )
        diff = report["classifier_vs_viterbi"]
        self.assertEqual(diff["total_observations"], 4)
        self.assertEqual(diff["agreement_count"], 3)
        self.assertEqual(diff["difference_count"], 1)
        self.assertEqual(diff["agreement_rate"], 0.75)
        self.assertEqual(len(diff["sample_differences"]), 1)
        self.assertEqual(diff["sample_differences"][0]["observation_index"], 2)

    def test_diagnostic_counts_reporting(self):
        report = build_integration_report(
            trip=self.trip,
            candidates=self.candidates,
            scored=self.scored,
            predictions=self.predictions,
            classifier_predictions=self.classifier_predictions,
            diagnostics=self.diagnostics,
            timings=self.timings,
            trip_path="dummy/path",
        )
        diag = report["diagnostics"]
        self.assertEqual(diag["total_diagnostic_events"], 2)
        self.assertEqual(diag["event_breakdown"]["large_gap_reset"], 1)
        self.assertEqual(diag["event_breakdown"]["missing_candidates"], 1)


class RealTeammateInterfaceEndToEndTests(unittest.TestCase):
    def test_end_to_end_real_interfaces_with_synthetic_trip_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trip_id = "00000099"
            trip_dir = Path(tmp_dir) / trip_id
            trip_dir.mkdir()

            # Create input files (.track, .nodes, .arcs) - NO .route file
            track_content = (
                "-1.0000 48.0000 1.0\n"
                "-0.9995 48.0000 2.0\n"
                "-0.9990 48.0000 3.0\n"
                "-0.9500 48.0500 4.0\n"  # isolated -> unmatched
            )
            nodes_content = (
                "-1.0000 48.0000\n"
                "-0.9990 48.0000\n"
                "-0.9980 48.0000\n"
            )
            arcs_content = (
                "0 1\n"
                "1 2\n"
            )

            (trip_dir / f"{trip_id}.track").write_text(track_content, encoding="utf-8")
            (trip_dir / f"{trip_id}.nodes").write_text(nodes_content, encoding="utf-8")
            (trip_dir / f"{trip_id}.arcs").write_text(arcs_content, encoding="utf-8")

            # Run integration demo exercising real load_trip, generate_candidates, score_candidates, decode
            with patch("src.routing.integration_demo.load_trip", wraps=load_trip) as mock_load:
                report = run_integration_demo(trip_dir)
                mock_load.assert_called_once_with(Path(trip_dir), include_reference=False)

            self.assertEqual(report["trip_id"], trip_id)
            self.assertTrue(report["validation"]["all_checks_passed"])
            self.assertEqual(report["coverage"]["total_observations"], 4)
            self.assertEqual(report["coverage"]["matched_observations"], 3)
            self.assertEqual(report["coverage"]["unmatched_observations"], 1)
            self.assertGreater(report["candidate_counts"]["total_candidates"], 0)
            self.assertFalse((trip_dir / f"{trip_id}.route").exists())

    def test_cli_execution_from_module(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trip_id = "00000088"
            trip_dir = Path(tmp_dir) / trip_id
            trip_dir.mkdir()

            track_content = "-1.000 48.000 1.0\n-0.999 48.000 2.0\n"
            nodes_content = "-1.000 48.000\n-0.999 48.000\n"
            arcs_content = "0 1\n"

            (trip_dir / f"{trip_id}.track").write_text(track_content, encoding="utf-8")
            (trip_dir / f"{trip_id}.nodes").write_text(nodes_content, encoding="utf-8")
            (trip_dir / f"{trip_id}.arcs").write_text(arcs_content, encoding="utf-8")

            cmd = [sys.executable, "-m", "src.routing.integration_demo", str(trip_dir)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"CLI stderr: {proc.stderr}")

            parsed = json.loads(proc.stdout)
            self.assertEqual(parsed["trip_id"], trip_id)
            self.assertTrue(parsed["validation"]["all_checks_passed"])
            self.assertEqual(parsed["coverage"]["matched_observations"], 2)


if __name__ == "__main__":
    unittest.main()
