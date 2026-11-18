"""Opt-in directed candidate positions for routing experiments."""

from __future__ import annotations

import numpy as np

from src.data.pipeline import _project_local


def project_candidate_positions(trip, candidates) -> dict[tuple[int, int], float | None]:
    """Map (observation_index, arc_id) to the nearest position on the directed arc.

    Pass one loaded trip and its candidates. Fractions are clamped to [0, 1],
    measured from the from-node. Zero-length arcs return None. The result is
    separate from candidate records so callers can evaluate it independently.
    """
    if not candidates:
        return {}

    observations = trip["observations"]
    nodes = trip["network"]["nodes"]
    arcs = trip["network"]["arcs"]
    coordinates = [[o["longitude"], o["latitude"]] for o in observations]
    projected = _project_local(list(nodes) + coordinates)
    points = {
        observation["observation_index"]: point
        for observation, point in zip(observations, projected[len(nodes):])
    }

    fractions = {}
    for candidate in candidates:
        observation_index, arc_id = candidate["observation_index"], candidate["arc_id"]
        if arc_id < 0 or arc_id >= len(arcs):
            raise ValueError(f"Candidate arc {arc_id} is outside this trip's network")
        from_node, to_node = arcs[arc_id]
        start, end = projected[from_node], projected[to_node]
        direction = end - start
        length_squared = float(np.dot(direction, direction))
        fraction = None
        if length_squared:
            along = float(np.dot(points[observation_index] - start, direction)) / length_squared
            fraction = min(1.0, max(0.0, along))
        fractions[(observation_index, arc_id)] = fraction
    return fractions
