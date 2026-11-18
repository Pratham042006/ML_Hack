"""Experimental position-aware transition cost. The shared decoder uses midpoints."""

import math

import numpy as np

from src.data.pipeline import _project_local
from src.routing.decoder import _decode_with_diagnostics


# Proposed by the data team, fixed independently of proxy results.
BACKWARD_JITTER_M = 5.0
BACKWARD_JITTER_MAX_GAP_SECONDS = 10.0


def candidate_fractions(trip, candidates):
    """Build a routing-local sidecar using the candidate generator's metric plane."""
    if not candidates:
        return {}
    nodes = trip["network"]["nodes"]
    arcs = trip["network"]["arcs"]
    observations = trip["observations"]
    projected = _project_local(list(nodes) + [
        [o["longitude"], o["latitude"]] for o in observations
    ])
    points = {o["observation_index"]: point
              for o, point in zip(observations, projected[len(nodes):])}
    fractions = {}
    for candidate in candidates:
        index, arc = candidate["observation_index"], candidate["arc_id"]
        start, end = arcs[arc]
        vector = projected[end] - projected[start]
        squared_length = vector @ vector
        fraction = None
        if squared_length > 0:
            along = ((points[index] - projected[start]) @ vector) / squared_length
            fraction = float(np.clip(along, 0, 1))
        fractions[index, arc] = fraction
    return fractions


class ProjectedDistance:
    """Directed travel between two projected positions on candidate arcs."""

    def __init__(self, fractions):
        self.fractions = dict(fractions)
        for key, value in self.fractions.items():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not 0 <= value <= 1
            ):
                raise ValueError(f"projection fraction for {key} must be null or in [0, 1]")

    def __call__(
        self, graph, first_index, first_arc, second_index, second_arc, *,
        delta_seconds, limit_m=math.inf,
    ):
        first = self.fractions.get((first_index, first_arc))
        second = self.fractions.get((second_index, second_arc))
        previous, current = graph.arcs[first_arc], graph.arcs[second_arc]
        if first is None or second is None or previous.length_m == 0 or current.length_m == 0:
            return graph.candidate_travel_distance_m(first_arc, second_arc, limit_m=limit_m)

        if first_arc == second_arc:
            progress = (second - first) * previous.length_m
            jitter = -BACKWARD_JITTER_M <= progress < 0 and (
                0 < delta_seconds <= BACKWARD_JITTER_MAX_GAP_SECONDS
            )
            if progress >= 0 or jitter:
                distance = max(0.0, progress)
                return distance if distance <= limit_m else math.inf
            # A larger backwards shift can only be explained by a directed loop.

        tail = (1 - first) * previous.length_m
        head = second * current.length_m
        budget = limit_m - tail - head
        if budget < 0:
            return math.inf
        between = graph.shortest_node_distance_m(previous.to_node, current.from_node, limit_m=budget)
        return tail + between + head


def decode_projected(observations, scored_candidates, network, *, distance):
    return _decode_with_diagnostics(
        observations, scored_candidates, network, travel_distance=distance,
    )
