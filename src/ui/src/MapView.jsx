import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// The API sends [longitude, latitude]; Leaflet wants [latitude, longitude].
const toLatLng = ([lon, lat]) => [lat, lon];

export default function MapView({ result }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const layersRef = useRef({ group: null, control: null });

  // Create the map once.
  useEffect(() => {
    const map = L.map(containerRef.current, { preferCanvas: true }).setView([56.8, 35.9], 12);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);
    mapRef.current = map;
    return () => map.remove();
  }, []);

  // Redraw whenever a new result arrives.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const state = layersRef.current;
    if (state.control) state.control.remove();
    if (state.group) state.group.forEach((layer) => layer.remove());
    state.group = [];
    state.control = null;
    if (!result) return;

    const { observations, predictions, mode } = result;
    const isMock = mode === "mock";

    // Layer 1: raw GPS points (blue) + a thin trace line.
    const gpsPoints = L.layerGroup();
    observations.forEach((o) => {
      L.circleMarker([o.latitude, o.longitude], {
        radius: 3, color: "#1d4ed8", weight: 1, fillOpacity: 0.7,
      })
        .bindPopup(`GPS #${o.observation_index}<br/>t = ${o.timestamp}s`)
        .addTo(gpsPoints);
    });
    const trace = L.polyline(observations.map((o) => [o.latitude, o.longitude]), {
      color: "#1d4ed8", weight: 1, opacity: 0.4, dashArray: "4 4",
    });

    // Layer 2: predicted road segments (red), each arc drawn once.
    const predicted = L.layerGroup();
    const seen = new Set();
    predictions.forEach((p) => {
      if (p.arc_id === null || p.arc_id === undefined || !p.geometry || p.geometry.length < 2) return;
      if (seen.has(p.arc_id)) return;
      seen.add(p.arc_id);
      L.polyline(p.geometry.map(toLatLng), { color: "#dc2626", weight: 4, opacity: 0.85 })
        .bindPopup(`Segment (arc ${p.arc_id})`)
        .addTo(predicted);
    });

    // Layer 3: unmatched observations (orange), so gaps are visible and not hidden.
    const unmatched = L.layerGroup();
    predictions.forEach((p) => {
      if (p.arc_id !== null && p.arc_id !== undefined) return;
      const o = observations[p.observation_index];
      if (!o) return;
      L.circleMarker([o.latitude, o.longitude], {
        radius: 6, color: "#ea580c", weight: 2, fillColor: "#fdba74", fillOpacity: 0.9,
      })
        .bindPopup(`Unmatched GPS #${o.observation_index}`)
        .addTo(unmatched);
    });

    // Draw reset markers; matched road geometries remain separate segments.
    const resets = L.layerGroup();
    const byIndex = new Map(observations.map((o) => [o.observation_index, o]));
    (result.diagnostics?.events || []).filter((e) => e.event.endsWith("_reset")).forEach((event) => {
      const o = byIndex.get(event.observation_index);
      if (!o) return;
      L.circleMarker([o.latitude, o.longitude], { radius: 7, color: "#7c3aed", weight: 3, fillOpacity: 0.5 })
        .bindPopup(`Route block reset at GPS #${event.observation_index}: ${event.event}`)
        .addTo(resets);
    });

    const predictedName = isMock ? "Predicted segments (MOCK, synthetic)" : "Predicted road segments";
    [trace, gpsPoints, predicted, unmatched, resets].forEach((l) => l.addTo(map));
    state.group = [trace, gpsPoints, predicted, unmatched, resets];
    state.control = L.control
      .layers(null, {
        "GPS trace": trace,
        "GPS observations": gpsPoints,
        [predictedName]: predicted,
        "Unmatched observations": unmatched,
        "Decoder resets": resets,
      })
      .addTo(map);

    if (observations.length) {
      map.fitBounds(L.latLngBounds(observations.map((o) => [o.latitude, o.longitude])), { padding: [30, 30] });
    }
  }, [result]);

  return <div ref={containerRef} className="map" />;
}
