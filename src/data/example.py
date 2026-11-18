"""Inspect one real development trip without accessing reference data."""

import argparse
import json
from collections import Counter
from time import perf_counter

from src.data.pipeline import load_trip, generate_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip_dir", help="Directory containing <trip_id>.track/.nodes/.arcs")
    args = parser.parse_args()
    started = perf_counter()
    trip = load_trip(args.trip_dir)
    loaded = perf_counter()
    candidates = generate_candidates(trip)
    finished = perf_counter()
    counts = Counter(candidate["observation_index"] for candidate in candidates)
    observations = trip["observations"]
    intervals = [
        current["timestamp"] - previous["timestamp"]
        for previous, current in zip(observations, observations[1:])
    ]
    print(json.dumps({
        "trip_id": trip["trip_id"],
        "observation_count": len(observations),
        "node_count": len(trip["network"]["nodes"]),
        "arc_count": len(trip["network"]["arcs"]),
        "first_observation": observations[0],
        "first_node": trip["network"]["nodes"][0],
        "first_arc": trip["network"]["arcs"][0],
        "duplicate_timestamp_steps": sum(interval == 0 for interval in intervals),
        "gaps_over_one_second": sum(interval > 1 for interval in intervals),
        "max_gap_seconds": max(intervals, default=None),
        "reference_loaded": "reference_route" in trip,
        "candidate_count": len(candidates),
        "observations_without_candidates": [
            observation["observation_index"] for observation in observations
            if not counts[observation["observation_index"]]
        ],
        "first_candidate": candidates[0] if candidates else None,
        "load_seconds": round(loaded - started, 3),
        "candidate_seconds": round(finished - loaded, 3),
    }, indent=2))


if __name__ == "__main__":
    main()
