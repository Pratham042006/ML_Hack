"""Directed-road classifier baseline and Viterbi route decoder.

The public ``decode`` entry point follows ``docs/INTEGRATION_CONTRACT.md``.
It returns matched segments per observation; it does not expand the result into
every intermediate road segment on a shortest path.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Iterable


EARTH_RADIUS_M = 6_371_008.8
DEFAULT_MAX_GAP_SECONDS = 30.0
DEFAULT_MAX_SPEED_MPS = 60.0
SCORE_EPSILON = 1e-12


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _haversine_m(first: Iterable[float], second: Iterable[float]) -> float:
    lon1, lat1 = (_finite_number(v, "coordinate") for v in first)
    lon2, lat2 = (_finite_number(v, "coordinate") for v in second)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


@dataclass(frozen=True)
class Arc:
    arc_id: int
    from_node: int
    to_node: int
    length_m: float


class DirectedRoadGraph:
    """Validated adapter for the contract's trip-local directed network."""

    def __init__(self, network: dict[str, Any]):
        if not isinstance(network, dict):
            raise ValueError("network must be a dictionary")
        raw_nodes = network.get("nodes")
        raw_arcs = network.get("arcs")
        if not isinstance(raw_nodes, list) or not isinstance(raw_arcs, list):
            raise ValueError("network must contain list-valued nodes and arcs")

        self.nodes: list[tuple[float, float]] = []
        for node_id, raw_node in enumerate(raw_nodes):
            if not isinstance(raw_node, (list, tuple)) or len(raw_node) != 2:
                raise ValueError(f"node {node_id} must be [longitude, latitude]")
            lon = _finite_number(raw_node[0], f"node {node_id} longitude")
            lat = _finite_number(raw_node[1], f"node {node_id} latitude")
            self.nodes.append((lon, lat))

        self.arcs: list[Arc] = []
        self._adjacency: list[list[tuple[int, float]]] = [
            [] for _ in self.nodes
        ]
        for arc_id, raw_arc in enumerate(raw_arcs):
            if not isinstance(raw_arc, (list, tuple)) or len(raw_arc) != 2:
                raise ValueError(f"arc {arc_id} must be [from_node_id, to_node_id]")
            from_node, to_node = raw_arc
            if (
                isinstance(from_node, bool)
                or isinstance(to_node, bool)
                or not isinstance(from_node, int)
                or not isinstance(to_node, int)
            ):
                raise ValueError(f"arc {arc_id} endpoints must be integer node IDs")
            if not (0 <= from_node < len(self.nodes)) or not (
                0 <= to_node < len(self.nodes)
            ):
                raise ValueError(f"arc {arc_id} references a missing node")
            length_m = _haversine_m(self.nodes[from_node], self.nodes[to_node])
            arc = Arc(arc_id, from_node, to_node, length_m)
            self.arcs.append(arc)
            self._adjacency[from_node].append((to_node, length_m))

        self._distance_cache: dict[tuple[int, int], float] = {}
        self._search_cutoffs: dict[tuple[int, int], float] = {}

    def validate_arc_id(self, arc_id: Any) -> int:
        if isinstance(arc_id, bool) or not isinstance(arc_id, int):
            raise ValueError("candidate arc_id must be an integer")
        if not 0 <= arc_id < len(self.arcs):
            raise ValueError(f"candidate arc_id {arc_id} is outside the trip network")
        return arc_id

    def arc_geometry(self, arc_id: int) -> list[list[float]]:
        arc = self.arcs[arc_id]
        return [list(self.nodes[arc.from_node]), list(self.nodes[arc.to_node])]

    def shortest_node_distance_m(
        self, source: int, target: int, *, limit_m: float = math.inf
    ) -> float:
        """Return infinity when no directed path fits within the search limit."""
        if source == target:
            return 0.0 if limit_m >= 0.0 else math.inf
        key = (source, target)
        if key in self._distance_cache:
            distance = self._distance_cache[key]
            return distance if distance <= limit_m else math.inf
        if key in self._search_cutoffs and limit_m <= self._search_cutoffs[key]:
            return math.inf

        distances = {source: 0.0}
        queue = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances.get(node):
                continue
            if distance > limit_m:
                self._search_cutoffs[key] = limit_m
                return math.inf
            if node == target:
                self._distance_cache[key] = distance
                return distance
            for next_node, edge_length in self._adjacency[node]:
                candidate = distance + edge_length
                if candidate < distances.get(next_node, math.inf):
                    distances[next_node] = candidate
                    heapq.heappush(queue, (candidate, next_node))

        self._distance_cache[key] = math.inf
        return math.inf

    def candidate_travel_distance_m(
        self, previous_arc_id: int, arc_id: int, *, limit_m: float = math.inf
    ) -> float:
        """Approximate directed travel between candidate-segment midpoints.

        Staying on the same directed arc costs zero because the candidate
        contract has no projected position or fraction along the segment.
        Otherwise the cost is half the previous arc, a directed shortest path
        from its end node to the next arc's start node, then half the next arc.
        A reverse-direction arc is therefore reachable only if that directed
        reverse arc exists, and its turn can be reached in the directed graph.
        """

        if previous_arc_id == arc_id:
            return 0.0
        previous = self.arcs[previous_arc_id]
        current = self.arcs[arc_id]
        previous_half = previous.length_m / 2.0
        current_half = current.length_m / 2.0
        if previous_half + current_half > limit_m:
            return math.inf
        between = self.shortest_node_distance_m(
            previous.to_node, current.from_node, limit_m=limit_m - previous_half - current_half
        )
        if not math.isfinite(between):
            return math.inf
        return previous_half + between + current_half


def _validate_observations(
    observations: list[dict[str, Any]],
) -> tuple[list[int], list[float]]:
    if not isinstance(observations, list):
        raise ValueError("observations must be a list")
    indices: list[int] = []
    timestamps: list[float] = []
    seen: set[int] = set()
    for position, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise ValueError(f"observation {position} must be a dictionary")
        index = observation.get("observation_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"observation {position} has an invalid observation_index")
        if index in seen:
            raise ValueError(f"duplicate observation_index {index}")
        seen.add(index)
        indices.append(index)
        timestamps.append(
            _finite_number(observation.get("timestamp"), f"observation {index} timestamp")
        )
        _finite_number(observation.get("longitude"), f"observation {index} longitude")
        _finite_number(observation.get("latitude"), f"observation {index} latitude")
    return indices, timestamps


def _candidate_groups(
    scored_candidates: list[dict[str, Any]],
    valid_observation_indices: set[int],
    graph: DirectedRoadGraph,
) -> dict[int, list[dict[str, Any]]]:
    if not isinstance(scored_candidates, list):
        raise ValueError("scored_candidates must be a list")
    groups: dict[int, list[dict[str, Any]]] = {
        index: [] for index in valid_observation_indices
    }
    seen_pairs: set[tuple[int, int]] = set()
    for position, candidate in enumerate(scored_candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate {position} must be a dictionary")
        observation_index = candidate.get("observation_index")
        if observation_index not in valid_observation_indices:
            raise ValueError(
                f"candidate {position} refers to unknown observation_index "
                f"{observation_index!r}"
            )
        arc_id = graph.validate_arc_id(candidate.get("arc_id"))
        pair = (observation_index, arc_id)
        if pair in seen_pairs:
            raise ValueError(
                f"duplicate candidate arc_id {arc_id} for observation {observation_index}"
            )
        seen_pairs.add(pair)
        score = _finite_number(candidate.get("score"), f"candidate {position} score")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"candidate {position} score must be in [0, 1]")
        copied = dict(candidate)
        copied["arc_id"] = arc_id
        copied["score"] = score
        groups[observation_index].append(copied)
    return groups


def _geometry(candidate: dict[str, Any], graph: DirectedRoadGraph) -> list[list[float]]:
    raw = candidate.get("geometry")
    if isinstance(raw, list) and len(raw) >= 2:
        try:
            return [
                [
                    _finite_number(point[0], "candidate geometry longitude"),
                    _finite_number(point[1], "candidate geometry latitude"),
                ]
                for point in raw
                if isinstance(point, (list, tuple)) and len(point) == 2
            ]
        except ValueError:
            pass
    return graph.arc_geometry(candidate["arc_id"])


def _unmatched(observation_index: int) -> dict[str, Any]:
    return {"observation_index": observation_index, "arc_id": None, "geometry": []}


def _prediction(
    observation_index: int,
    candidate: dict[str, Any],
    graph: DirectedRoadGraph,
) -> dict[str, Any]:
    return {
        "observation_index": observation_index,
        "arc_id": candidate["arc_id"],
        "geometry": _geometry(candidate, graph),
    }


def decode_classifier_only(
    observations: list[dict[str, Any]],
    scored_candidates: list[dict[str, Any]],
    network: dict[str, Any],
) -> list[dict[str, Any]]:
    """Choose each observation's highest-scoring candidate independently."""

    graph = DirectedRoadGraph(network)
    observation_indices, _ = _validate_observations(observations)
    groups = _candidate_groups(scored_candidates, set(observation_indices), graph)
    predictions: list[dict[str, Any]] = []
    for observation_index in observation_indices:
        candidates = groups[observation_index]
        if not candidates:
            predictions.append(_unmatched(observation_index))
            continue
        best = max(enumerate(candidates), key=lambda item: (item[1]["score"], -item[0]))[1]
        predictions.append(_prediction(observation_index, best, graph))
    return predictions


def _emission_logs(
    candidates: list[dict[str, Any]],
) -> tuple[list[float], bool]:
    total = sum(candidate["score"] for candidate in candidates)
    all_zero = total == 0.0
    if all_zero:
        probability = 1.0 / len(candidates)
        return [math.log(probability)] * len(candidates), True
    return [
        math.log(max(candidate["score"] / total, SCORE_EPSILON))
        for candidate in candidates
    ], False


def _transition_log_probability(
    graph: DirectedRoadGraph,
    previous_candidate: dict[str, Any],
    candidate: dict[str, Any],
    previous_observation: dict[str, Any],
    observation: dict[str, Any],
    delta_seconds: float,
    max_speed_mps: float,
    travel_distance=None,
) -> float:
    max_travel_m = max_speed_mps * delta_seconds
    # Allow for rounding in the bound; the original speed check is authoritative.
    search_limit_m = max_travel_m + max(1.0, max_travel_m) * 1e-12
    if travel_distance is None:
        network_distance = graph.candidate_travel_distance_m(
            previous_candidate["arc_id"], candidate["arc_id"], limit_m=search_limit_m
        )
    else:
        network_distance = travel_distance(
            graph, previous_observation["observation_index"], previous_candidate["arc_id"],
            observation["observation_index"], candidate["arc_id"],
            delta_seconds=delta_seconds, limit_m=search_limit_m,
        )
    if not math.isfinite(network_distance):
        return -math.inf
    if delta_seconds > 0.0 and network_distance / delta_seconds > max_speed_mps:
        return -math.inf

    observed_distance = _haversine_m(
        (previous_observation["longitude"], previous_observation["latitude"]),
        (observation["longitude"], observation["latitude"]),
    )
    scale_m = max(25.0, 0.5 * observed_distance + 5.0 * max(delta_seconds, 1.0))
    return -abs(network_distance - observed_distance) / scale_m


def decode_with_diagnostics(
    observations: list[dict[str, Any]],
    scored_candidates: list[dict[str, Any]],
    network: dict[str, Any],
    *,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run Viterbi decoding and return explicit route-break diagnostics."""
    return _decode_with_diagnostics(
        observations, scored_candidates, network,
        max_gap_seconds=max_gap_seconds, max_speed_mps=max_speed_mps,
    )


def _decode_with_diagnostics(
    observations, scored_candidates, network, *,
    max_gap_seconds=DEFAULT_MAX_GAP_SECONDS, max_speed_mps=DEFAULT_MAX_SPEED_MPS,
    travel_distance=None,
):
    max_gap_seconds = _finite_number(max_gap_seconds, "max_gap_seconds")
    max_speed_mps = _finite_number(max_speed_mps, "max_speed_mps")
    if max_gap_seconds <= 0.0 or max_speed_mps <= 0.0:
        raise ValueError("gap and speed thresholds must be positive")

    graph = DirectedRoadGraph(network)
    observation_indices, timestamps = _validate_observations(observations)
    groups = _candidate_groups(scored_candidates, set(observation_indices), graph)
    predictions = [_unmatched(index) for index in observation_indices]
    diagnostics: list[dict[str, Any]] = []

    block_positions: list[int] = []
    block_candidates: list[list[dict[str, Any]]] = []
    block_scores: list[list[float]] = []
    block_backpointers: list[list[int | None]] = []

    def finalize_block() -> None:
        nonlocal block_positions, block_candidates, block_scores, block_backpointers
        if not block_positions:
            return
        state = max(
            range(len(block_scores[-1])), key=lambda index: block_scores[-1][index]
        )
        states = [state]
        for step in range(len(block_positions) - 1, 0, -1):
            previous_state = block_backpointers[step][states[-1]]
            if previous_state is None:
                raise RuntimeError("internal Viterbi backpointer is missing")
            states.append(previous_state)
        states.reverse()
        for position, candidates, state_index in zip(
            block_positions, block_candidates, states
        ):
            predictions[position] = _prediction(
                observation_indices[position], candidates[state_index], graph
            )
        block_positions = []
        block_candidates = []
        block_scores = []
        block_backpointers = []

    def start_block(position: int, candidates: list[dict[str, Any]]) -> None:
        emissions, all_zero = _emission_logs(candidates)
        block_positions.append(position)
        block_candidates.append(candidates)
        block_scores.append(emissions)
        block_backpointers.append([None] * len(candidates))
        if all_zero:
            diagnostics.append(
                {
                    "observation_index": observation_indices[position],
                    "event": "all_zero_scores_uniform",
                }
            )

    for position, observation_index in enumerate(observation_indices):
        candidates = groups[observation_index]
        if not candidates:
            finalize_block()
            diagnostics.append(
                {"observation_index": observation_index, "event": "missing_candidates"}
            )
            continue

        if not block_positions:
            start_block(position, candidates)
            continue

        previous_position = block_positions[-1]
        delta_seconds = timestamps[position] - timestamps[previous_position]
        if delta_seconds <= 0.0 or delta_seconds > max_gap_seconds:
            finalize_block()
            diagnostics.append(
                {
                    "observation_index": observation_index,
                    "event": (
                        "nonpositive_timestamp_reset"
                        if delta_seconds <= 0.0
                        else "large_gap_reset"
                    ),
                    "delta_seconds": delta_seconds,
                }
            )
            start_block(position, candidates)
            continue

        emissions, all_zero = _emission_logs(candidates)
        previous_candidates = block_candidates[-1]
        previous_scores = block_scores[-1]
        current_scores = [-math.inf] * len(candidates)
        backpointers: list[int | None] = [None] * len(candidates)

        for current_index, candidate in enumerate(candidates):
            best_score = -math.inf
            best_previous: int | None = None
            for previous_index, previous_candidate in enumerate(previous_candidates):
                transition = _transition_log_probability(
                    graph,
                    previous_candidate,
                    candidate,
                    observations[previous_position],
                    observations[position],
                    delta_seconds,
                    max_speed_mps,
                    travel_distance,
                )
                score = previous_scores[previous_index] + transition
                if score > best_score:
                    best_score = score
                    best_previous = previous_index
            if best_previous is not None and math.isfinite(best_score):
                current_scores[current_index] = best_score + emissions[current_index]
                backpointers[current_index] = best_previous

        if not any(math.isfinite(score) for score in current_scores):
            finalize_block()
            diagnostics.append(
                {
                    "observation_index": observation_index,
                    "event": "unreachable_transition_reset",
                }
            )
            start_block(position, candidates)
            continue

        block_positions.append(position)
        block_candidates.append(candidates)
        block_scores.append(current_scores)
        block_backpointers.append(backpointers)
        if all_zero:
            diagnostics.append(
                {
                    "observation_index": observation_index,
                    "event": "all_zero_scores_uniform",
                }
            )

    finalize_block()
    return predictions, diagnostics


def decode(
    observations: list[dict[str, Any]],
    scored_candidates: list[dict[str, Any]],
    network: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return one Viterbi-selected road segment for every input observation."""

    predictions, _ = decode_with_diagnostics(
        observations, scored_candidates, network
    )
    return predictions
