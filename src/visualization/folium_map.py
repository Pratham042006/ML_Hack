"""Folium map for the technical report and for development checks.

This is NOT the evaluator UI. It is allowed to draw the reference route (for development/student
trips only, and only with --show-reference), always on its own clearly labelled layer.

Examples (run from the repository root):

    python -m src.visualization.folium_map --trip-dir data/train/00000001 --out outputs/map_train1.html
    python -m src.visualization.folium_map --trip-dir data/student_test/00000052 \
        --predictions outputs/response_00000052.json --show-reference --out outputs/map_52.html

--predictions takes the JSON returned by POST /predict, so the map shows exactly what the API produced.
"""
import argparse
import json
from pathlib import Path

import folium


def read_table(path: Path, columns: int, cast) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()  # tabs/spaces/CRLF
            if len(parts) >= columns:
                rows.append([cast(p) for p in parts[:columns]])
    return rows


def load_trip_files(trip_dir: Path, with_reference: bool) -> dict:
    tid = trip_dir.name
    data = {
        "trip_id": tid,
        "track": read_table(trip_dir / f"{tid}.track", 3, float),   # lon, lat, t
        "nodes": read_table(trip_dir / f"{tid}.nodes", 2, float),   # lon, lat
        "arcs": read_table(trip_dir / f"{tid}.arcs", 2, int),       # from, to
        "route": [],
    }
    if with_reference:
        data["route"] = [r[0] for r in read_table(trip_dir / f"{tid}.route", 1, int)]
    return data


def arc_line(data: dict, arc_id: int):
    a, b = data["arcs"][arc_id]
    (lon1, lat1), (lon2, lat2) = data["nodes"][a], data["nodes"][b]
    return [[lat1, lon1], [lat2, lon2]]  # folium wants [lat, lon]


def build_map(data: dict, response: dict = None, margin: float = 0.01) -> folium.Map:
    track = data["track"]
    lats = [p[1] for p in track]
    lons = [p[0] for p in track]
    bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
    fmap = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], zoom_start=14, tiles="OpenStreetMap")

    # Road network near the trip only (the full network can be huge).
    lat_lo, lat_hi = bounds[0][0] - margin, bounds[1][0] + margin
    lon_lo, lon_hi = bounds[0][1] - margin, bounds[1][1] + margin
    inside = lambda n: lat_lo <= data["nodes"][n][1] <= lat_hi and lon_lo <= data["nodes"][n][0] <= lon_hi
    network = [arc_line(data, i) for i, (a, b) in enumerate(data["arcs"]) if inside(a) or inside(b)]
    net_layer = folium.FeatureGroup(name=f"Road network near trip ({len(network)} arcs)", show=False)
    if network:
        folium.PolyLine(network, color="#6b7280", weight=1, opacity=0.6).add_to(net_layer)
    net_layer.add_to(fmap)

    # Raw GPS.
    gps_layer = folium.FeatureGroup(name="GPS observations")
    folium.PolyLine([[p[1], p[0]] for p in track], color="#1d4ed8", weight=1, opacity=0.4, dash_array="4 4").add_to(gps_layer)
    for i, (lon, lat, t) in enumerate(track):
        folium.CircleMarker([lat, lon], radius=2, color="#1d4ed8", fill=True, fill_opacity=0.7,
                            tooltip=f"GPS #{i}, t={t}s").add_to(gps_layer)
    gps_layer.add_to(fmap)
    folium.Marker([track[0][1], track[0][0]], tooltip="Start", icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker([track[-1][1], track[-1][0]], tooltip="End", icon=folium.Icon(color="black")).add_to(fmap)

    legend = [("#1d4ed8", "GPS observations"), ("#6b7280", "Road network (hidden by default)")]

    # Reference route (development only).
    if data["route"]:
        ref_layer = folium.FeatureGroup(name="Reference route (development only, not model output)")
        for arc_id in data["route"]:
            folium.PolyLine(arc_line(data, arc_id), color="#16a34a", weight=5, opacity=0.8,
                            tooltip=f"reference arc {arc_id}").add_to(ref_layer)
        ref_layer.add_to(fmap)
        legend.append(("#16a34a", "Reference route (development only)"))

    # Predictions straight from the API response.
    if response:
        mode = response.get("mode", "unknown")
        label = f"Predicted route ({mode.upper()}" + (", synthetic, NOT a result)" if mode == "mock" else ")")
        pred_layer = folium.FeatureGroup(name=label)
        un_layer = folium.FeatureGroup(name="Unmatched observations")
        seen, obs = set(), response["observations"]
        for p in response["predictions"]:
            if p["arc_id"] is None:
                o = obs[p["observation_index"]]
                folium.CircleMarker([o["latitude"], o["longitude"]], radius=5, color="#ea580c", fill=True,
                                    tooltip=f"Unmatched GPS #{o['observation_index']}").add_to(un_layer)
            elif p["arc_id"] not in seen and len(p["geometry"]) >= 2:
                seen.add(p["arc_id"])
                folium.PolyLine([[lat, lon] for lon, lat in p["geometry"]], color="#dc2626", weight=4,
                                opacity=0.85, tooltip=f"arc {p['arc_id']}").add_to(pred_layer)
        pred_layer.add_to(fmap)
        un_layer.add_to(fmap)
        legend += [("#dc2626", label), ("#ea580c", "Unmatched observations")]

    rows = "".join(f'<div><span style="background:{c};width:14px;height:4px;display:inline-block;'
                   f'margin-right:6px"></span>{t}</div>' for c, t in legend)
    fmap.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:#fff;padding:8px 10px;'
        f'border:1px solid #999;border-radius:6px;font:12px sans-serif"><b>Trip {data["trip_id"]}</b>{rows}</div>'))
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds(bounds, padding=(20, 20))
    return fmap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trip-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--predictions", type=Path, help="JSON saved from POST /predict")
    ap.add_argument("--show-reference", action="store_true", help="draw .route (development trips only)")
    ap.add_argument("--margin", type=float, default=0.01, help="degrees of padding for the road network")
    args = ap.parse_args()

    response = json.loads(args.predictions.read_text()) if args.predictions else None
    if response and response.get("trip_id") != args.trip_dir.name:
        raise SystemExit(f"Predictions are for trip {response.get('trip_id')}, not {args.trip_dir.name}.")
    data = load_trip_files(args.trip_dir, with_reference=args.show_reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build_map(data, response, args.margin).save(str(args.out))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
