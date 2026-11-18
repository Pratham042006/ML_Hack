"""Diagnostic for permitted training/development references; NOT label creation."""

import argparse
from collections import Counter
import json

from src.data.pipeline import load_trip, generate_candidates


def investigate(trip_dir):
    trip = load_trip(trip_dir, include_reference=True)
    route = trip["reference_route"]
    arcs = trip["network"]["arcs"]
    frequencies = Counter(route)
    directed_breaks, shared_node_breaks = [], []
    for position, (previous, current) in enumerate(zip(route, route[1:]), 1):
        if arcs[previous][1] != arcs[current][0]:
            directed_breaks.append(position)
        if not set(arcs[previous]).intersection(arcs[current]):
            shared_node_breaks.append(position)
    # Deliberately ignore the reference when generating candidates. This checks
    # only set overlap with a route; it does NOT determine pointwise truth.
    candidates = generate_candidates(trip)
    route_set = set(route)
    overlaps = {c["observation_index"] for c in candidates if c["arc_id"] in route_set}
    return {
        "trip_id": trip["trip_id"], "partition": "caller must verify training/student-test",
        "observations": len(trip["observations"]), "reference_entries": len(route),
        "unique_reference_arcs": len(frequencies),
        "arc_ids_repeated_in_reference": sum(count > 1 for count in frequencies.values()),
        "directed_continuity_break_count": len(directed_breaks),
        "first_directed_break_positions": directed_breaks[:10],
        "no_shared_endpoint_break_count": len(shared_node_breaks),
        "observations_with_any_candidate_in_reference_arc_set": len(overlaps),
        "warning": "Set overlap is NOT candidate recall, pointwise accuracy or verified alignment. No labels produced.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip_dir", help="Permitted training/student-test trip ONLY")
    args = parser.parse_args()
    print(json.dumps(investigate(args.trip_dir), indent=2))
