"""Approximate reference labels, kept completely separate from inference.

States are positions in the full ordered reference route, not unique arc IDs.
No inference candidates or model scores are accepted by the aligner.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from src.data.pipeline import _project_local, _point_segment_distances


METHOD_VERSION = "monotone-reference-v1"


@dataclass(frozen=True)
class AlignmentConfig:
    distance_scale_m: float = 20.0
    emission_cap_m: float = 100.0
    max_label_distance_m: float = 50.0
    skipped_position_cost: float = 0.05
    ambiguity_cost_delta: float = 2.0
    gps_gap_seconds: float = 10.0
    max_cells: int = 10_000_000

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(self.max_cells, int):
            raise ValueError("max_cells must be an integer")
        if self.emission_cap_m < self.max_label_distance_m:
            raise ValueError("emission_cap_m must cover max_label_distance_m")


def _decode_costs(emissions, skip_cost):
    """Exact O(T*R) monotone DP and per-state full-path min-marginal costs.

    Stay or advance to any later route position. Advancing from k to j costs
    skip_cost * max(j-k-1, 0); no start/end position is forced.
    """
    count, width = emissions.shape
    positions = np.arange(width)
    forward = np.empty_like(emissions)
    parents = np.empty((count, width), dtype=np.int32)
    forward[0] = emissions[0]
    parents[0] = -1
    for t in range(1, count):
        previous = forward[t - 1]
        adjusted = previous - skip_cost * positions
        prefix = np.minimum.accumulate(adjusted)
        # First attaining index makes tie-breaking reproducible.
        new_minimum = np.r_[True, adjusted[1:] < prefix[:-1]]
        argmin = np.maximum.accumulate(np.where(new_minimum, positions, 0))
        advance = np.r_[np.inf, prefix[:-1] + skip_cost * (positions[1:] - 1)]
        stay = previous <= advance
        parents[t] = np.where(stay, positions, np.r_[0, argmin[:-1]])
        forward[t] = emissions[t] + np.minimum(previous, advance)
    path = np.empty(count, dtype=np.int32)
    path[-1] = int(np.argmin(forward[-1]))
    for t in range(count - 1, 0, -1):
        path[t - 1] = parents[t, path[t]]
    best_cost = float(forward[-1, path[-1]])
    # Backward recurrence; reuse emissions storage for min-marginals after its
    # current row is consumed. Caller supplies a disposable emission matrix.
    backward = emissions[-1].copy()
    for t in range(count - 1, -1, -1):
        if t < count - 1:
            suffix = np.minimum.accumulate((backward + skip_cost * positions)[::-1])[::-1]
            advance = np.r_[suffix[1:] - skip_cost * (positions[:-1] + 1), np.inf]
            backward = emissions[t] + np.minimum(backward, advance)
        emissions[t] = forward[t] + backward - emissions[t]
    return path, best_cost, emissions


def _empty_record(observation, reason):
    return {
        "observation_index": observation["observation_index"],
        "status": "unavailable", "reasons": [reason],
        "arc_id": None, "route_position": None, "distance_m": None,
        "alternative_cost_gap": None, "plausible_position_count": 0,
        "plausible_route_positions": [], "label_source": "approximate_reference_alignment",
        "method_version": METHOD_VERSION,
    }


def align_reference(trip, config=None):
    """Infer one diagnostic record per observation without reading candidates.

    'confident' is a heuristic status, never a calibrated probability or verified
    ground truth. Ambiguous/unavailable records are not suitable for training.
    Disconnected reference sequences are conservatively unavailable as a whole.
    """
    config = config or AlignmentConfig()
    observations = trip["observations"]
    route = trip.get("reference_route", [])
    network = trip["network"]
    metadata = {
        "method_version": METHOD_VERSION,
        "label_source": "approximate_reference_alignment",
        "verified_ground_truth": False, "config": asdict(config),
        "state_definition": "zero-based occurrence position in full ordered reference_route",
    }
    if not observations:
        return {"provenance": metadata, "labels": [], "best_path_cost": None}
    reason = None
    if not route:
        reason = "empty_or_missing_reference"
    elif len(route) * len(observations) > config.max_cells:
        reason = "alignment_resource_limit"
    elif any(network["arcs"][a][1] != network["arcs"][b][0] for a, b in zip(route, route[1:])):
        reason = "disconnected_or_direction_inconsistent_reference"
    if reason:
        return {"provenance": metadata,
                "labels": [_empty_record(o, reason) for o in observations],
                "best_path_cost": None}
    timestamps = np.array([o["timestamp"] for o in observations], dtype=float)
    if not np.isfinite(timestamps).all() or (np.diff(timestamps) < 0).any():
        raise ValueError("Alignment requires finite, nondecreasing timestamps")
    nodes = network["nodes"]
    xy = _project_local(list(nodes) + [[o["longitude"], o["latitude"]] for o in observations])
    route_arcs = np.array([network["arcs"][arc_id] for arc_id in route], dtype=int)
    starts, ends = xy[route_arcs[:, 0]], xy[route_arcs[:, 1]]
    distances = np.array([_point_segment_distances(p, starts, ends) for p in xy[len(nodes):]])
    emissions = (np.minimum(distances, config.emission_cap_m) / config.distance_scale_m) ** 2
    path, best_cost, marginals = _decode_costs(emissions, config.skipped_position_cost)
    # Intervals lacking temporal evidence qualify both neighboring labels.
    gap_neighbors = set()
    duplicate_neighbors = set()
    for i, delta in enumerate(np.diff(timestamps)):
        if delta > config.gps_gap_seconds:
            gap_neighbors.update((i, i + 1))
        if delta == 0:
            duplicate_neighbors.update((i, i + 1))
    labels = []
    for i, (observation, position) in enumerate(zip(observations, path)):
        position = int(position)
        alternative = marginals[i].copy()
        alternative[position] = np.inf
        alternative_cost = float(alternative.min())
        margin = max(0.0, alternative_cost - best_cost) if math.isfinite(alternative_cost) else None
        plausible = np.flatnonzero(marginals[i] <= best_cost + config.ambiguity_cost_delta + 1e-8)
        residual = float(distances[i, position])
        reasons = []
        if residual > config.max_label_distance_m:
            reasons.append("distant_from_aligned_route_segment")
        if len(plausible) > 1:
            reasons.append("multiple_plausible_route_positions")
        if i in gap_neighbors:
            reasons.append("adjacent_large_gps_gap")
        if i in duplicate_neighbors:
            reasons.append("adjacent_duplicate_timestamp")
        status = "unavailable" if residual > config.max_label_distance_m else (
            "ambiguous" if reasons else "confident")
        labels.append({
            "observation_index": observation["observation_index"],
            "status": status, "reasons": reasons,
            "arc_id": int(route[position]), "route_position": position,
            "distance_m": residual, "alternative_cost_gap": margin,
            "plausible_position_count": int(len(plausible)),
            "plausible_route_positions": [int(j) for j in plausible[:10]],
            "label_source": "approximate_reference_alignment", "method_version": METHOD_VERSION,
        })
    return {"provenance": metadata, "labels": labels, "best_path_cost": best_cost}


def assess_candidate_coverage(alignment, candidates):
    """Add coverage AFTER alignment; never change route positions or statuses.

    Miss denominator is confident inferred labels only, not all observations.
    Ambiguous/unavailable observations remain in the ledger with null miss flags.
    """
    candidate_ids = {}
    for candidate in candidates:
        candidate_ids.setdefault(candidate["observation_index"], set()).add(candidate["arc_id"])
    counts = {"total_observations": len(alignment["labels"]), "confident": 0,
              "ambiguous": 0, "unavailable": 0, "candidate_misses": 0,
              "no_candidates_all_observations": 0, "training_eligible": 0}
    labels = []
    for label in alignment["labels"]:
        ids = candidate_ids.get(label["observation_index"], set())
        usable = label["status"] == "confident"
        miss = label["arc_id"] not in ids if usable else None
        counts[label["status"]] += 1
        counts["candidate_misses"] += int(miss is True)
        counts["no_candidates_all_observations"] += int(not ids)
        eligible = usable and not miss
        counts["training_eligible"] += int(eligible)
        labels.append({**label, "candidate_count": len(ids),
                       "candidate_miss": miss, "training_eligible": eligible})
    counts["candidate_miss_denominator"] = counts["confident"]
    counts["candidate_miss_rate"] = (
        counts["candidate_misses"] / counts["confident"] if counts["confident"] else None)
    return {**alignment, "labels": labels, "counts": counts}
