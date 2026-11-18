import unittest

import folium

from src.routing.map_preview import build_preview


class MapPreviewTests(unittest.TestCase):
    def test_renderer_adds_segments_and_markers_without_joining_blocks(self):
        observations = [
            {"observation_index": i, "longitude": i * 0.001, "latitude": 0}
            for i in (0, 2, 4, 6, 8)
        ]
        predictions = [
            {"observation_index": i, "arc_id": arc,
             "geometry": [[i * 0.001, 0], [i * 0.001 + 0.0005, 0]] if arc is not None else []}
            for i, arc in [(0, 1), (2, 2), (4, 1), (6, None), (8, 1)]
        ]
        events = [{"observation_index": 4, "event": "large_gap_reset"}]
        preview = build_preview(observations, predictions, events)
        layers = {layer.layer_name: layer for layer in preview._children.values()
                  if isinstance(layer, folium.FeatureGroup)}
        lines = list(layers["Matched segments per observation"]._children.values())
        self.assertEqual(len(lines), 4)
        for line, prediction in zip(lines, (p for p in predictions if p["arc_id"] is not None)):
            self.assertIsInstance(line, folium.PolyLine)
            self.assertEqual(line.locations, [[lat, lon] for lon, lat in prediction["geometry"]])
        self.assertEqual([line.options["color"] for line in lines],
                         ["#1d4ed8", "#1d4ed8", "#0f766e", "#7e22ce"])
        html = preview.get_root().render()
        self.assertIn("large_gap_reset at GPS #4", html)
        self.assertIn("Unmatched GPS #6", html)
        self.assertEqual(html.count("L.polyline("), 4)
