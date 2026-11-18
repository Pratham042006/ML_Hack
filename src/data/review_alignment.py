"""Review inferred labels locally with threshold comparisons and close-up maps."""

import argparse
from collections import Counter
from dataclasses import replace
from html import escape
import json
from pathlib import Path

import numpy as np

from src.data.alignment import AlignmentConfig, align_reference, assess_candidate_coverage
from src.data.pipeline import generate_candidates, load_trip, _point_segment_distances, _project_local


def _reference_breaks(trip):
    arcs, route = trip["network"]["arcs"], trip["reference_route"]
    return [position for position in range(1, len(route))
            if arcs[route[position - 1]][1] != arcs[route[position]][0]]


def section_evidence(trip):
    """Check whether continuous reference sections have distinct ordered GPS anchors."""
    route = trip["reference_route"]
    breaks = _reference_breaks(trip)
    if not breaks:
        return {"break_positions": [], "sections": [], "split_windows": []}
    nodes, arcs = trip["network"]["nodes"], trip["network"]["arcs"]
    observations = trip["observations"]
    coordinates = [[o["longitude"], o["latitude"]] for o in observations]
    projected = _project_local(nodes + coordinates)
    positions = [0, *breaks, len(route)]
    sections = list(zip(positions, positions[1:]))
    starts, ends = np.asarray([arcs[arc_id] for arc_id in route]).T
    anchors = [[] for _ in sections]
    for i, point in enumerate(projected[len(nodes):]):
        distances = _point_segment_distances(point, projected[starts], projected[ends])
        nearest = [float(distances[a:b].min()) for a, b in sections]
        best, second = sorted(range(len(nearest)), key=nearest.__getitem__)[:2]
        if nearest[best] <= 50 and nearest[second] - nearest[best] >= 20:
            anchors[best].append(observations[i]["observation_index"])
    splits = [
        {"break_position": position, "after_observation": before[-1] if before else None,
         "before_observation": after[0] if after else None,
         "ordered_nonoverlapping_anchors": bool(before and after and before[-1] < after[0])}
        for position, before, after in zip(breaks, anchors, anchors[1:])
    ]
    experiments = []
    if all(window["ordered_nonoverlapping_anchors"] for window in splits):
        candidates = generate_candidates(trip)
        for (a, b), indices in zip(sections, anchors):
            # Restrict experiments to uniquely anchored observation bounds. The
            # interval between sections is deliberately left outside all fits.
            subset = [o for o in observations if indices[0] <= o["observation_index"] <= indices[-1]]
            trial = {**trip, "reference_route": route[a:b], "observations": subset}
            coverage = assess_candidate_coverage(align_reference(trial), candidates)
            experiments.append({"position_start": a, "position_end_exclusive": b,
                                "first_observation": indices[0], "last_observation": indices[-1],
                                "counts": coverage["counts"]})
    return {
        "break_positions": breaks,
        "sections": [
            {"position_start": a, "position_end_exclusive": b, "anchors": len(indices),
             "first_anchor": indices[0] if indices else None,
             "last_anchor": indices[-1] if indices else None}
            for (a, b), indices in zip(sections, anchors)
        ],
        "split_windows": splits,
        "anchor_rule": "Nearest section within 50m and at least 20m nearer than the runner-up",
        "experimental_section_counts": experiments,
        "experimental_observations_outside_section_windows": len(observations) - sum(
            trial["counts"]["total_observations"] for trial in experiments),
    }


def _sensitivity(trip, candidates):
    defaults = AlignmentConfig()
    baseline = assess_candidate_coverage(align_reference(trip, defaults), candidates)
    results = [{"setting": "default", **baseline["counts"]}]
    for field, value in [("max_label_distance_m", 35.), ("max_label_distance_m", 65.),
                         ("ambiguity_cost_delta", 1.), ("ambiguity_cost_delta", 3.)]:
        changed = assess_candidate_coverage(align_reference(trip, replace(defaults, **{field: value})),
                                            candidates)
        results.append({"setting": f"{field}={value}", **changed["counts"],
                        "status_changes": sum(a["status"] != b["status"] for a, b in
                                              zip(baseline["labels"], changed["labels"])),
                        "selected_arc_changes": sum(a["arc_id"] != b["arc_id"] for a, b in
                                                    zip(baseline["labels"], changed["labels"]))})
    return results, baseline["labels"]


def _parallel_candidate(trip, label, candidates):
    if label["arc_id"] is None:
        return None
    nodes, arcs = trip["network"]["nodes"], trip["network"]["arcs"]
    base = [nodes[i] for i in arcs[label["arc_id"]]]
    for candidate in candidates:
        if candidate["arc_id"] == label["arc_id"]:
            continue
        a, b, c, d = _project_local(base + candidate["geometry"])
        u, v = b - a, d - c
        length, other_length = np.linalg.norm(u), np.linalg.norm(v)
        if min(length, other_length) == 0:
            continue
        direction = u / length
        normal = np.array([-direction[1], direction[0]])
        offsets = (np.array([c, d]) - a) @ normal
        projections = (np.array([c, d]) - a) @ direction
        if (abs(direction @ (v / other_length)) >= 0.966
                and offsets[0] * offsets[1] > 0
                and 3 <= min(abs(offsets)) <= max(abs(offsets)) <= 40
                and min(projections) < length and max(projections) > 0):
            return candidate["arc_id"]
    return None


def _write_map(path, trip, label, nearby):
    nodes, arcs, route = trip["network"]["nodes"], trip["network"]["arcs"], trip["reference_route"]
    index = label["observation_index"]
    observations = [o for o in trip["observations"] if abs(o["observation_index"] - index) <= 8]
    center = next(o for o in observations if o["observation_index"] == index)
    origin = [center["longitude"], center["latitude"]]
    geometry = [[nodes[i] for i in arcs[arc_id]] for arc_id in
                dict.fromkeys(c["arc_id"] for c in nearby)]
    positions = list(dict.fromkeys(label["plausible_route_positions"] + (
        [label["route_position"]] if label["route_position"] is not None else [])))
    references = [[nodes[i] for i in arcs[route[p]]] for p in positions]
    gps = [[o["longitude"], o["latitude"]] for o in observations]
    # Center a local metric map on the inspected observation.
    coords = [origin] + [p for segment in geometry + references for p in segment] + gps
    projected = _project_local(coords)
    metric = projected - projected[0]
    span = max(60., float(np.max(np.abs(metric)))) * 1.1
    pixels = metric * [315 / span, -315 / span] + [340, 340]
    parts = ['<rect width="680" height="680" fill="#f5f5f4"/>']
    for i in range(len(geometry) + len(references)):
        (x1, y1), (x2, y2) = pixels[1 + 2 * i:3 + 2 * i]
        color = "#6b7280"
        if i >= len(geometry):
            position = positions[i - len(geometry)]
            color = "#b91c1c" if position == label["route_position"] else "#d97706"
        parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                     f'stroke="{color}" stroke-width="4"/>')
    for observation, (x, y) in zip(observations, pixels[-len(gps):]):
        radius = 7 if observation["observation_index"] == index else 3
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="#1d4ed8"/>')
    title = escape(f"Trip {trip['trip_id']} · Observation {index}")
    path.write_text(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{title}</title>'
        f'<h1>{title}</h1><p>Blue GPS observations · grey candidate roads · red selected reference '
        f'· orange plausible alternatives</p><p>Map is {2 * span:.0f} m across approximately. '
        f'One local metric projection is shared across the plotted geometry.</p>'
        f'<svg viewBox="0 0 680 680" width="680" role="img" aria-label="{title}">'
        + "".join(parts) + "</svg></html>", encoding="utf-8")


def review(training_root, prepared_dir, output_dir):
    root, prepared, output = map(Path, (training_root, prepared_dir, output_dir))
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    if manifest["partition"] != "training":
        raise ValueError("Use only the prepared training partition")
    report = {"dataset_sha256": manifest["dataset_sha256"], "sensitivity": {}, "cases": [],
              "disconnected_references": {}}
    by_reason = Counter()
    for summary in manifest["trips"]:
        trip_id = summary["trip_id"]
        if summary["status"] == "failed":
            continue
        # Resource-limit failures can mask a reference break in the aligner.
        if not {"disconnected_or_direction_inconsistent_reference", "alignment_resource_limit"
                }.intersection(summary["reason_counts"]):
            continue
        trip = load_trip(root / trip_id, include_reference=True)
        breaks = _reference_breaks(trip)
        if breaks:
            by_reason["disconnected_trips"] += 1
            report["disconnected_references"][trip_id] = {"break_positions": breaks}
            if trip_id == "00000002":
                report["disconnected_references"][trip_id] = section_evidence(trip)
    report["disconnected_trip_count"] = by_reason["disconnected_trips"]

    found = set()
    for trip_id in ("00000001", "00000003", "00000018", "00000008", "00000090", "00000095"):
        trip = load_trip(root / trip_id, include_reference=True)
        candidates = generate_candidates(trip)
        changes, labels = _sensitivity(trip, candidates)
        report["sensitivity"][trip_id] = changes
        by_index = {}
        for candidate in candidates:
            by_index.setdefault(candidate["observation_index"], []).append(candidate)
        route = trip["reference_route"]
        frequencies = Counter(route)
        for label in labels:
            index = label["observation_index"]
            nearby = by_index.get(index, [])
            cases = {}
            if "gps_gap" not in found and "adjacent_large_gps_gap" in label["reasons"]:
                cases["gps_gap"] = None
            if "ambiguous" not in found and label["status"] == "ambiguous":
                cases["ambiguous"] = None
            plausible = [route[i] for i in label["plausible_route_positions"]]
            if "repeated_occurrence" not in found and label["arc_id"] is not None and (
                len(set(plausible)) < len(plausible) or frequencies[label["arc_id"]] > 1
            ):
                cases["repeated_occurrence"] = None
            if "parallel_road" not in found and label["status"] == "ambiguous":
                parallel = _parallel_candidate(trip, label, nearby)
                if parallel is not None:
                    cases["parallel_road"] = parallel
            for case, other_arc in cases.items():
                found.add(case)
                filename = f"{case}.html"
                _write_map(output / filename, trip, label, nearby)
                report["cases"].append({"case": case, "trip_id": trip_id, "map": filename,
                                        "parallel_candidate_arc_id": other_arc,
                                        "selected_arc_occurrences": [i for i, arc in enumerate(route)
                                                                     if arc == label["arc_id"]],
                                        **label})
    (output / "diagnostics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_root")
    parser.add_argument("prepared_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    result = review(args.training_root, args.prepared_dir, args.output_dir)
    print(json.dumps({"disconnected_trip_count": result["disconnected_trip_count"],
                      "cases": [case["case"] for case in result["cases"]]}, indent=2))
