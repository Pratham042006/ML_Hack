"""Opt-in heading features from GPS and directed road geometry."""

from dataclasses import dataclass
import math

import numpy as np

from src.data.pipeline import _project_local


@dataclass(frozen=True)
class HeadingConfig:
    min_displacement_m: float = 5.0
    max_gap_seconds: float = 10.0

    def __post_init__(self):
        for value in (self.min_displacement_m, self.max_gap_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Heading thresholds must be finite and positive")


class TripHeading:
    """Prepare directions once per trip for an experimental feature sidecar."""

    def __init__(self, trip, config=HeadingConfig()):
        observations = trip["observations"]
        nodes = trip["network"]["nodes"]
        self.arcs = trip["network"]["arcs"]
        coordinates = [[o["longitude"], o["latitude"]] for o in observations]
        metric = _project_local(list(nodes) + coordinates)
        self.nodes = metric[:len(nodes)]
        self.observations = {}
        self.steps = {}
        previous = None
        for observation, point in zip(observations, metric[len(nodes):]):
            index = observation["observation_index"]
            direction, distance, seconds = None, None, None
            reason = "first_observation"
            if previous is not None:
                previous_timestamp, previous_point = previous
                displacement = point - previous_point
                seconds = observation["timestamp"] - previous_timestamp
                distance = float(np.linalg.norm(displacement))
                if not math.isfinite(seconds) or seconds <= 0:
                    reason = "nonpositive_or_invalid_time_gap"
                elif seconds > config.max_gap_seconds:
                    reason = "large_time_gap"
                elif distance < config.min_displacement_m:
                    reason = "insufficient_displacement"
                else:
                    reason = None
                    direction = displacement / distance
            self.observations[index] = (direction, reason)
            self.steps[index] = (distance, seconds)
            previous = (observation["timestamp"], point)

    def agreement(self, observation_index, arc_id):
        """Return (cosine, missing reason). Missing values are Python None."""
        if not 0 <= arc_id < len(self.arcs):
            raise ValueError(f"Arc {arc_id} is outside this trip")
        gps, reason = self.observations[observation_index]
        if gps is None:
            return None, reason
        a, b = self.arcs[arc_id]
        road = self.nodes[b] - self.nodes[a]
        length = float(np.linalg.norm(road))
        if length == 0:
            return None, "zero_length_arc"
        # The dot product of unit directions is cos(GPS heading - road heading).
        # It works across north without subtracting wrapped angles.
        return min(1.0, max(-1.0, float(np.dot(gps, road / length)))), None
