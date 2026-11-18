"""Trip loading and validation. Raw inputs and row-based IDs stay unchanged."""

from pathlib import Path
import math

import numpy as np


# Initial integration settings, not selected using evaluation labels.
CANDIDATE_RADIUS_M = 50.0
MAX_CANDIDATES_PER_OBSERVATION = 8


def _project_local(coordinates):
    """Return approximate local WGS84 metric coordinates for one small area.

    A local equirectangular projection uses WGS84 meridional/prime-vertical
    radii of curvature at the area's central latitude. It is not a geodesic
    distance calculation. Reject broad/polar areas rather than silently
    applying the approximation outside its intended range.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Coordinates must have shape (n, 2): longitude, latitude")
    if not np.isfinite(coordinates).all():
        raise ValueError("Coordinates must be finite")
    if (np.abs(coordinates[:, 0]) > 180).any() or (np.abs(coordinates[:, 1]) > 90).any():
        raise ValueError("Coordinates outside longitude/latitude bounds")
    lower, upper = coordinates.min(axis=0), coordinates.max(axis=0)
    if ((upper - lower) > 2.0).any() or (np.abs(coordinates[:, 1]) > 75).any():
        raise ValueError("Local projection requires span <= 2 degrees and latitude within +/-75")
    longitude0, latitude0 = (lower + upper) / 2
    phi = math.radians(latitude0)
    eccentricity_squared = 6.6943799901413165e-3
    denominator = 1 - eccentricity_squared * math.sin(phi) ** 2
    prime_vertical_radius = 6378137.0 / math.sqrt(denominator)
    meridional_radius = 6378137.0 * (1 - eccentricity_squared) / denominator ** 1.5
    scale = np.array([prime_vertical_radius * math.cos(phi), meridional_radius])
    return np.radians(coordinates - [longitude0, latitude0]) * scale


def _point_segment_distances(point, starts, ends):
    """Shortest distances to CLOSED segments, including zero-length segments."""
    vectors = ends - starts
    lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
    fractions = np.divide(
        np.einsum("ij,ij->i", point - starts, vectors), lengths_squared,
        out=np.zeros(len(starts)), where=lengths_squared > 0,
    )
    closest = starts + np.clip(fractions, 0, 1)[:, None] * vectors
    return np.linalg.norm(point - closest, axis=1)


def generate_candidates(trip) -> list[dict]:
    """Return up to eight nearest arcs within 50 approximate metric metres.

    One trip per call. Order is observation order, then distance and arc ID.
    Reference fields are never consulted. Empty candidate sets stay empty.
    Directed/reverse arcs remain separate IDs; geometry follows input endpoints.
    """
    observations = trip["observations"]
    nodes = trip["network"]["nodes"]
    arcs = trip["network"]["arcs"]
    seen = set()
    for observation in observations:
        index = observation["observation_index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in seen:
            raise ValueError("Observation indices must be unique nonnegative integers")
        seen.add(index)
    if not observations or not arcs:
        return []
    if not nodes:
        raise ValueError("Network arcs require nodes")
    for endpoints in arcs:
        if len(endpoints) != 2 or any(
            isinstance(index, bool) or not isinstance(index, int)
            or not 0 <= index < len(nodes) for index in endpoints
        ):
            raise ValueError("Arc endpoints must be valid integer node IDs")

    observation_coordinates = [
        [observation["longitude"], observation["latitude"]] for observation in observations
    ]
    projected = _project_local(list(nodes) + observation_coordinates)
    projected_nodes, points = projected[:len(nodes)], projected[len(nodes):]
    endpoints = np.asarray(arcs, dtype=np.int64)
    starts, ends = projected_nodes[endpoints[:, 0]], projected_nodes[endpoints[:, 1]]
    minimum, maximum = np.minimum(starts, ends), np.maximum(starts, ends)
    candidates = []
    for observation, point in zip(observations, points):
        # Bounding-box filtering cannot miss a long segment just because its
        # endpoint or midpoint is far away. Exact segment distance follows.
        nearby = np.flatnonzero(
            (minimum[:, 0] <= point[0] + CANDIDATE_RADIUS_M)
            & (maximum[:, 0] >= point[0] - CANDIDATE_RADIUS_M)
            & (minimum[:, 1] <= point[1] + CANDIDATE_RADIUS_M)
            & (maximum[:, 1] >= point[1] - CANDIDATE_RADIUS_M)
        )
        distances = _point_segment_distances(point, starts[nearby], ends[nearby])
        valid = distances <= CANDIDATE_RADIUS_M
        nearby, distances = nearby[valid], distances[valid]
        order = np.lexsort((nearby, distances))[:MAX_CANDIDATES_PER_OBSERVATION]
        for position in order:
            arc_id = int(nearby[position])
            from_node, to_node = arcs[arc_id]
            candidates.append({
                "observation_index": observation["observation_index"],
                "arc_id": arc_id,
                "features": {"distance_m": float(distances[position])},
                "geometry": [list(nodes[from_node]), list(nodes[to_node])],
            })
    return candidates


def _rows(path, width, converter):
    """Reject malformed rows instead of silently shifting row-based IDs."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) != width:
                raise ValueError(
                    f"{path}:{line_number}: expected {width} columns, got {len(fields)}"
                )
            try:
                values = [converter(field) for field in fields]
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid {converter.__name__} value"
                ) from exc
            yield line_number, values


def _check_coordinates(longitude, latitude, path, line_number):
    if not (
        math.isfinite(longitude)
        and math.isfinite(latitude)
        and -180 <= longitude <= 180
        and -90 <= latitude <= 90
    ):
        raise ValueError(f"{path}:{line_number}: invalid longitude/latitude")


def load_trip(trip_dir, include_reference=False) -> dict:
    """Load <trip_id>.track/.nodes/.arcs from one trip directory.

    Coordinates are longitude/latitude; timestamps are seconds. No sorting,
    deletion, imputation or ID renumbering is performed. Decreasing timestamps
    are rejected; equal timestamps and positive gaps are retained. The .route
    file is accessed only when include_reference=True.
    """
    directory = Path(trip_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Trip directory does not exist: {directory}")
    trip_id = directory.name

    def trip_file(extension):
        path = directory / f"{trip_id}.{extension}"
        if not path.is_file():
            raise FileNotFoundError(f"Required trip file does not exist: {path}")
        return path

    track_path = trip_file("track")
    observations = []
    previous_timestamp = None
    for line_number, (longitude, latitude, timestamp) in _rows(track_path, 3, float):
        _check_coordinates(longitude, latitude, track_path, line_number)
        if not math.isfinite(timestamp):
            raise ValueError(f"{track_path}:{line_number}: timestamp must be finite")
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise ValueError(
                f"{track_path}:{line_number}: decreasing timestamp; input was not reordered"
            )
        observations.append({
            "observation_index": line_number - 1,
            "longitude": longitude,
            "latitude": latitude,
            "timestamp": timestamp,
        })
        previous_timestamp = timestamp
    if not observations:
        raise ValueError(f"{track_path}: no GPS observations")

    nodes_path = trip_file("nodes")
    nodes = []
    for line_number, coordinates in _rows(nodes_path, 2, float):
        _check_coordinates(*coordinates, nodes_path, line_number)
        nodes.append(coordinates)
    if not nodes:
        raise ValueError(f"{nodes_path}: no road nodes")

    arcs_path = trip_file("arcs")
    arcs = []
    for line_number, endpoints in _rows(arcs_path, 2, int):
        if any(node_id < 0 or node_id >= len(nodes) for node_id in endpoints):
            raise ValueError(
                f"{arcs_path}:{line_number}: node IDs must be in [0, {len(nodes) - 1}]"
            )
        arcs.append(endpoints)
    if not arcs:
        raise ValueError(f"{arcs_path}: no road arcs")

    trip = {
        "trip_id": trip_id,
        "observations": observations,
        "network": {"nodes": nodes, "arcs": arcs},
    }
    if include_reference:
        route_path = trip_file("route")
        reference_route = []
        for line_number, (arc_id,) in _rows(route_path, 1, int):
            if not 0 <= arc_id < len(arcs):
                raise ValueError(
                    f"{route_path}:{line_number}: arc ID must be in [0, {len(arcs) - 1}]"
                )
            reference_route.append(arc_id)
        trip["reference_route"] = reference_route
    return trip
