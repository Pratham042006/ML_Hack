"""Compare adjacent matches under a shared directed distance policy."""

import math

from src.routing.decoder import DEFAULT_MAX_GAP_SECONDS, DEFAULT_MAX_SPEED_MPS


def audit_continuity(observations, predictions, graph, *, travel_distance=None):
    """Return mutually exclusive pair counts and pair details keyed by destination."""
    counts = dict.fromkeys(
        ("unmatched_endpoint", "nonpositive_timestamp", "large_gap",
         "disconnected", "over_speed_cap", "plausible"),
        0,
    )
    transitions = {}
    for i in range(1, len(observations)):
        previous, current = observations[i - 1 : i + 1]
        first_arc, second_arc = (p["arc_id"] for p in predictions[i - 1 : i + 1])
        delta = current["timestamp"] - previous["timestamp"]
        distance = None
        if first_arc is None or second_arc is None:
            status = "unmatched_endpoint"
        elif delta <= 0:
            status = "nonpositive_timestamp"
        elif delta > DEFAULT_MAX_GAP_SECONDS:
            status = "large_gap"
        else:
            if travel_distance is None:
                travel = graph.candidate_travel_distance_m(first_arc, second_arc)
            else:
                travel = travel_distance(
                    graph, previous["observation_index"], first_arc,
                    current["observation_index"], second_arc,
                    delta_seconds=delta,
                )
            if not math.isfinite(travel):
                status = "disconnected"
            else:
                distance = round(travel, 2)
                status = "over_speed_cap" if travel / delta > DEFAULT_MAX_SPEED_MPS else "plausible"
        counts[status] += 1
        transitions[current["observation_index"]] = {
            "status": status,
            "delta_seconds": delta,
            "travel_distance_m": distance,
        }
    return counts, transitions
