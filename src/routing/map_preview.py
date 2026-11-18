"""Generate a reference-free Folium preview for the routing map handoff."""

import argparse
from pathlib import Path

import folium

from src.data.pipeline import generate_candidates, load_trip
from src.models.inference import score_candidates
from src.routing.decoder import decode_with_diagnostics
from src.routing.map_diagnostics import RESET_EVENTS, matched_blocks


def build_preview(observations, predictions, events):
    """Draw one line per road segment, grouped by block, with explicit breaks."""
    points = {o["observation_index"]: (o["latitude"], o["longitude"]) for o in observations}
    positions = {p["observation_index"]: pos for pos, p in enumerate(predictions)}
    preview = folium.Map()
    if points:
        preview.fit_bounds(list(points.values()))

    gps = folium.FeatureGroup(name="GPS observations")
    matches = folium.FeatureGroup(name="Matched segments per observation")
    breaks = folium.FeatureGroup(name="Unmatched observations and resets")
    colors = ("#1d4ed8", "#0f766e", "#7e22ce")
    for index, point in points.items():
        folium.CircleMarker(point, radius=2, tooltip=f"GPS #{index}").add_to(gps)
    for block_number, block in enumerate(matched_blocks(predictions, events), 1):
        start = positions[block["first_observation_index"]]
        stop = positions[block["last_observation_index"]] + 1
        color = colors[(block_number - 1) % len(colors)]
        for prediction in predictions[start:stop]:
            folium.PolyLine(
                [(lat, lon) for lon, lat in prediction["geometry"]],
                color=color, weight=4,
                tooltip=f"Block {block_number}, arc {prediction['arc_id']}, GPS #{prediction['observation_index']}",
            ).add_to(matches)
    for prediction in predictions:
        index = prediction["observation_index"]
        if prediction["arc_id"] is None:
            folium.CircleMarker(
                points[index], radius=5, color="#ea580c", tooltip=f"Unmatched GPS #{index}",
            ).add_to(breaks)
    for event in events:
        reason = event["event"]
        if reason in RESET_EVENTS:
            index = event["observation_index"]
            folium.CircleMarker(
                points[index], radius=6, color="#b91c1c", tooltip=f"{reason} at GPS #{index}",
            ).add_to(breaks)
    for layer in (gps, matches, breaks):
        layer.add_to(preview)
    folium.LayerControl().add_to(preview)
    return preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.trip.resolve().parent.name not in {"train", "student_test"}:
        parser.error("expected a permitted development trip under train or student_test")
    trip = load_trip(args.trip, include_reference=False)
    observations = trip["observations"]
    scored = score_candidates(generate_candidates(trip))
    predictions, events = decode_with_diagnostics(observations, scored, trip["network"])
    with args.out.open("x", encoding="utf-8") as output:
        output.write(build_preview(observations, predictions, events).get_root().render())
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
