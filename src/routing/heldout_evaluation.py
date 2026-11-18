"""Evaluate a frozen, permitted student-test partition and pool count metrics."""

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re

from src.data.candidate_projection import project_candidate_positions
from src.data.heading import HeadingConfig
from src.data.pipeline import CANDIDATE_RADIUS_M, MAX_CANDIDATES_PER_OBSERVATION
from src.models.inference import BASELINE_DISTANCE_SCALE_M
from src.routing.decoder import DEFAULT_MAX_GAP_SECONDS, DEFAULT_MAX_SPEED_MPS
from src.routing.projected import BACKWARD_JITTER_M, BACKWARD_JITTER_MAX_GAP_SECONDS
from src.routing.projection_comparison import compare_trip


STUDENT_TEST_TRIPS = (
    "00000000", "00000007", "00000009", "00000040", "00000052",
    "00000056", "00000065", "00000076", "00000077", "00000091",
)
DECODERS = ("classifier_only", "midpoint", "projected")


def _pooled_proxy(rows):
    if not rows:
        return None
    pooled = {}
    for metric in ("all_confident", "confident_with_label_in_candidates"):
        hits = sum(row[metric]["agreements"] for row in rows)
        denominator = sum(row[metric]["denominator"] for row in rows)
        pooled[metric] = {"agreements": hits, "denominator": denominator,
                          "rate": hits / denominator if denominator else None}
    return pooled


def pool_results(reports, *, paired, with_proxy):
    proxy_reports = [report["offline_proxy"] for report in reports] if with_proxy else []
    pooled = {
        field: sum(report[field] for report in reports)
        for field in ("observations", "candidates", "observations_without_candidates")
    }
    pooled["load_generate_baseline_score_seconds_sum"] = round(
        sum(report.get("load_generate_score_seconds", 0) for report in reports), 4,
    )
    pooled["projection_preparation_seconds_sum"] = round(
        sum(report.get("projection_preparation_seconds", 0) for report in reports), 4,
    )
    if with_proxy:
        label_counts = Counter()
        for report in proxy_reports:
            label_counts.update({key: value for key, value in report["label_counts"].items()
                                 if key != "candidate_miss_rate"})
        denominator = label_counts["candidate_miss_denominator"]
        pooled["proxy_label_counts"] = {
            **label_counts,
            "candidate_miss_rate": label_counts["candidate_misses"] / denominator if denominator else None,
        }
    scorers = {"distance_baseline": ("agreement", [report["runs"] for report in reports])}
    if paired:
        scorers["heading_artifact"] = (
            "model_agreement", [report["model_comparison"]["runs"] for report in reports],
        )
        pooled["artifact_score_seconds_sum"] = round(
            sum(report["model_comparison"].get("score_seconds", 0) for report in reports), 4,
        )
    for scorer, (proxy_key, runs) in scorers.items():
        pooled[scorer] = {}
        for decoder in DECODERS:
            selected = [trip_runs[decoder] for trip_runs in runs]
            result = {
                "matched": sum(run["matched"] for run in selected),
                "unmatched": sum(run["unmatched"] for run in selected),
                "decode_seconds_sum": round(sum(run["decode_seconds"] for run in selected), 4),
            }
            for policy in ("midpoint_audit", "projected_audit"):
                counts = Counter()
                for run in selected:
                    counts.update(run[policy])
                result[policy] = dict(counts)
            if decoder != "classifier_only":
                events = Counter()
                for run in selected:
                    events.update(run["diagnostics"]["event_counts"])
                result["reset_events"] = dict(events)
                result["matched_blocks"] = sum(len(run["diagnostics"]["matched_blocks"]) for run in selected)
            result["proxy_agreement"] = _pooled_proxy([
                report[proxy_key][decoder] for report in proxy_reports
            ])
            pooled[scorer][decoder] = result
    return pooled


def _file_sha256(path):
    with path.open("rb") as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_student_test(root, *, artifact=None, artifact_sha=None, training_sha=None, with_proxy=False):
    root = Path(root).resolve()
    if root.name != "student_test":
        raise ValueError("expected the supplied student_test split directory")
    if artifact is not None and not all(
        value and re.fullmatch("[0-9a-fA-F]{64}", value) for value in (artifact_sha, training_sha)
    ):
        raise ValueError("paired runs require trusted artifact and training CSV SHA-256 values")

    folders = [root / trip_id for trip_id in STUDENT_TEST_TRIPS]
    suffixes = ("track", "nodes", "arcs")
    required = suffixes + (("route",) if with_proxy else ())
    for folder in folders:
        if folder.resolve().parent != root or any(
            not (folder / f"{folder.name}.{suffix}").is_file() for suffix in required
        ):
            raise ValueError(f"missing or invalid student-test inputs for {folder.name}")

    reports = []
    for folder in folders:
        report = compare_trip(
            folder, with_proxy=with_proxy, model_artifact=artifact,
            expected_artifact_sha256=artifact_sha, expected_training_sha256=training_sha,
            position_provider=project_candidate_positions,
        )
        report["input_sha256"] = {
            suffix: _file_sha256(folder / f"{folder.name}.{suffix}") for suffix in required
        }
        reports.append(report)

    return {
        "partition": "student_test",
        "trip_ids": STUDENT_TEST_TRIPS,
        "mode": "paired_heading" if artifact is not None else "distance_baseline_only",
        "proxy_evaluation": with_proxy,
        "settings": {
            "distance_scale_m": BASELINE_DISTANCE_SCALE_M,
            "candidate_radius_m": CANDIDATE_RADIUS_M,
            "max_candidates": MAX_CANDIDATES_PER_OBSERVATION,
            "heading": asdict(HeadingConfig()),
            "max_gap_seconds": DEFAULT_MAX_GAP_SECONDS,
            "max_speed_mps": DEFAULT_MAX_SPEED_MPS,
            "backward_jitter_m": BACKWARD_JITTER_M,
            "backward_jitter_max_gap_seconds": BACKWARD_JITTER_MAX_GAP_SECONDS,
            "missing_projection": "use midpoint for the whole transition",
        },
        "artifact": {"sha256": artifact_sha.lower(), "training_csv_sha256": training_sha.lower()}
        if artifact is not None else None,
        "pooled": pool_results(reports, paired=artifact is not None, with_proxy=with_proxy),
        "trips": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student_test", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline-only", action="store_true")
    mode.add_argument("--model-artifact", type=Path, help="Trusted team-supplied heading model")
    parser.add_argument("--artifact-sha256")
    parser.add_argument("--training-sha256")
    parser.add_argument("--with-proxy", action="store_true")
    args = parser.parse_args()
    try:
        report = evaluate_student_test(
            args.student_test, artifact=args.model_artifact, artifact_sha=args.artifact_sha256,
            training_sha=args.training_sha256, with_proxy=args.with_proxy,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
