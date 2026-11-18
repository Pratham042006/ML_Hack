"""Verify data against the merged scorer or a read-only scorer from a Git commit."""

import argparse
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import tempfile

from src.data.pipeline import load_trip, generate_candidates


def _score_from_ref(candidates, ref):
    # Read a trusted, reviewed teammate commit without checking out/editing it.
    result = subprocess.run(
        ["git", "show", f"{ref}:src/models/inference.py"],
        check=True, capture_output=True, encoding="utf-8",
        cwd=Path(__file__).resolve().parents[2],
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "inference.py"
        path.write_text(result.stdout, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("pathsnap_reviewed_scorer", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.score_candidates(candidates)


def check(trip_dir, model_ref=None):
    trip = load_trip(trip_dir)
    candidates = generate_candidates(trip)
    if model_ref:
        scored = _score_from_ref(candidates, model_ref)
    else:
        from src.models.inference import score_candidates
        scored = score_candidates(candidates)
    if len(scored) != len(candidates):
        raise AssertionError("Scorer changed candidate count")
    groups = defaultdict(list)
    seen = set()
    for original, output in zip(candidates, scored):
        key = (original["observation_index"], original["arc_id"])
        if key in seen:
            raise AssertionError("Duplicate candidate IDs")
        seen.add(key)
        if any(output.get(name) != value for name, value in original.items()):
            raise AssertionError("Scorer changed input candidate fields/order")
        score = output["score"]
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise AssertionError("Invalid candidate score")
        groups[key[0]].append(score)
    sums = [math.fsum(scores) for scores in groups.values()]
    if any(not math.isclose(total, 1, abs_tol=1e-9) for total in sums):
        raise AssertionError("Candidate scores are not normalized per observation")
    return {
        "trip_id": trip["trip_id"], "mode": "distance_baseline",
        "model_ref": model_ref or "working_tree",
        "observations": len(trip["observations"]), "candidates": len(candidates),
        "observations_without_candidates": [
            observation["observation_index"] for observation in trip["observations"]
            if observation["observation_index"] not in groups
        ],
        "max_score_sum_error": max((abs(total - 1) for total in sums), default=None),
        "candidate_fields_and_order_preserved": True,
        "first_scored_candidate": scored[0] if scored else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip_dir")
    parser.add_argument("--model-ref", help="Trusted reviewed Git ref, e.g. 8f0d557")
    args = parser.parse_args()
    print(json.dumps(check(args.trip_dir, args.model_ref), indent=2))
