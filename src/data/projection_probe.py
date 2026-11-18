"""Measure projected positions and apparent backwards motion using only inference data."""

import argparse
from collections import Counter
import json

import numpy as np

from src.data.candidate_projection import project_candidate_positions
from src.data.pipeline import _project_local, generate_candidates, load_trip


def probe(trip_dir, *, jitter_m=5.0, max_gap_seconds=10.0):
    if not np.isfinite(jitter_m) or jitter_m < 0:
        raise ValueError("Jitter tolerance must be finite and nonnegative")
    if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("Maximum time gap must be finite and positive")

    trip = load_trip(trip_dir)
    candidates = generate_candidates(trip)
    fractions = project_candidate_positions(trip, candidates)
    nodes, arcs = trip["network"]["nodes"], trip["network"]["arcs"]
    observations = trip["observations"]
    projected = _project_local(nodes + [[o["longitude"], o["latitude"]] for o in observations])
    endpoints = np.asarray(arcs)
    lengths = np.linalg.norm(projected[endpoints[:, 1]] - projected[endpoints[:, 0]], axis=1)
    nearest = {}
    for candidate in candidates:
        # generate_candidates orders each observation's candidates by distance.
        nearest.setdefault(candidate["observation_index"], candidate["arc_id"])

    transitions = Counter({key: 0 for key in ("eligible", "forward_or_stationary",
                                              "backward_within_tolerance", "backward_beyond_tolerance",
                                              "large_or_nonpositive_time_gap", "missing_position")})
    backwards_examples = []
    for previous, current in zip(observations, observations[1:]):
        left, right = previous["observation_index"], current["observation_index"]
        arc_id = nearest.get(left)
        if arc_id is None or nearest.get(right) != arc_id:
            continue
        if not 0 < current["timestamp"] - previous["timestamp"] <= max_gap_seconds:
            transitions["large_or_nonpositive_time_gap"] += 1
            continue
        start, end = fractions.get((left, arc_id)), fractions.get((right, arc_id))
        if start is None or end is None:
            transitions["missing_position"] += 1
            continue
        transitions["eligible"] += 1
        progress = float((end - start) * lengths[arc_id])
        if progress >= 0:
            transitions["forward_or_stationary"] += 1
        elif progress >= -jitter_m:
            transitions["backward_within_tolerance"] += 1
        else:
            transitions["backward_beyond_tolerance"] += 1
            if len(backwards_examples) < 5:
                backwards_examples.append({"arc_id": arc_id, "from_observation": left,
                                            "to_observation": right, "progress_m": progress})

    return {"trip_id": trip["trip_id"], "observations": len(observations),
            "candidates": len(candidates), "zero_length_positions": sum(f is None for f in fractions.values()),
            "observations_without_candidates": len(observations) - len(nearest),
            "jitter_m": jitter_m, "max_gap_seconds": max_gap_seconds,
            "same_nearest_arc_transitions": dict(transitions), "backwards_examples": backwards_examples,
            "sample_positions": [
                {"observation_index": i, "arc_id": arc, "projection_fraction": f}
                for (i, arc), f in list(fractions.items())[:5]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip_dirs", nargs="+")
    parser.add_argument("--jitter-m", type=float, default=5.0)
    parser.add_argument("--max-gap-seconds", type=float, default=10.0)
    args = parser.parse_args()
    print(json.dumps([probe(trip_dir, jitter_m=args.jitter_m, max_gap_seconds=args.max_gap_seconds)
                      for trip_dir in args.trip_dirs], indent=2, allow_nan=False))
