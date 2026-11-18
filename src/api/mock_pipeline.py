"""MOCK mode: used only when the real teammate modules are not importable yet.

It reads the .track file (GPS only, never .route) so the UI has real GPS points to draw, and it
produces SYNTHETIC predictions. Those predictions are NOT road matches and must never be
reported as results. The API always labels this mode "mock" and adds a warning.
"""
from pathlib import Path


def load_observations(trip_dir: Path) -> list:
    trip_id = Path(trip_dir).name
    observations = []
    with open(Path(trip_dir) / f"{trip_id}.track", "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()  # handles tabs, spaces and CRLF
            if len(parts) < 3:
                continue
            observations.append(
                {
                    "observation_index": len(observations),
                    "longitude": float(parts[0]),
                    "latitude": float(parts[1]),
                    "timestamp": float(parts[2]),
                }
            )
    return observations


def synthetic_predictions(observations: list) -> list:
    """Fake 'segments' that simply join consecutive GPS points. Last point stays unmatched
    so the UI's unmatched-observation handling is exercised too."""
    predictions = []
    for i, obs in enumerate(observations):
        if i + 1 < len(observations):
            nxt = observations[i + 1]
            geometry = [[obs["longitude"], obs["latitude"]], [nxt["longitude"], nxt["latitude"]]]
            predictions.append({"observation_index": i, "arc_id": i, "geometry": geometry})
        else:
            predictions.append({"observation_index": i, "arc_id": None, "geometry": []})
    return predictions
