import { useEffect, useState } from "react";
import MapView from "./MapView.jsx";
import { defaultApiUrl, fetchTrips, runPrediction, saveApiUrl } from "./api.js";

const MODE_INFO = {
  mock: { label: "MOCK", cls: "mode-mock", text: "Synthetic output for interface testing. Not a real result." },
  baseline: { label: "BASELINE", cls: "mode-baseline", text: "Distance-based baseline scoring. Not a trained model." },
  trained: { label: "TRAINED", cls: "mode-trained", text: "Scored with a saved trained model artifact." },
};

export default function App() {
  const [apiUrl, setApiUrl] = useState(defaultApiUrl());
  const [trips, setTrips] = useState([]);
  const [tripId, setTripId] = useState("");
  const [loadingTrips, setLoadingTrips] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  async function loadTrips() {
    setLoadingTrips(true);
    setError("");
    try {
      saveApiUrl(apiUrl);
      const ids = await fetchTrips(apiUrl);
      setTrips(ids);
      setTripId(ids[0] || "");
      if (!ids.length) setError("The API is reachable but reports no trips. Check PATHSNAP_DATA_DIR on the backend.");
    } catch (e) {
      setTrips([]);
      setTripId("");
      setError(e.message);
    } finally {
      setLoadingTrips(false);
    }
  }

  useEffect(() => {
    loadTrips();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function run() {
    setRunning(true);
    setError("");
    setResult(null);
    try {
      setResult(await runPrediction(apiUrl, tripId));
    } catch (e) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  }

  const mode = result ? MODE_INFO[result.mode] || { label: result.mode.toUpperCase(), cls: "mode-trained", text: "" } : null;
  const matched = result ? result.predictions.filter((p) => p.arc_id !== null && p.arc_id !== undefined).length : 0;
  const unmatched = result ? result.predictions.length - matched : 0;
  const segments = result ? new Set(result.predictions.filter((p) => p.arc_id !== null).map((p) => p.arc_id)).size : 0;
  const hasMetrics = result && Object.keys(result.metrics || {}).length > 0;

  return (
    <div className="page">
      <header>
        <h1>PathSnap Evaluator</h1>
        {mode && <span className={`badge ${mode.cls}`}>{mode.label}</span>}
      </header>

      <section className="panel controls">
        <label>
          Backend URL
          <input value={apiUrl} onChange={(e) => setApiUrl(e.target.value)} spellCheck={false} />
        </label>
        <button onClick={loadTrips} disabled={loadingTrips}>
          {loadingTrips ? "Loading..." : "Load trips"}
        </button>
        <label>
          Trip
          <select value={tripId} onChange={(e) => setTripId(e.target.value)} disabled={!trips.length}>
            {trips.map((id) => (
              <option key={id} value={id}>{id}</option>
            ))}
          </select>
        </label>
        <button className="primary" onClick={run} disabled={!tripId || running}>
          {running ? "Running..." : "Run prediction"}
        </button>
      </section>

      {error && <div className="banner error">{error}</div>}
      {running && <div className="banner info">Running the pipeline for trip {tripId}...</div>}
      {mode && <div className={`banner ${mode.cls}-banner`}><b>{mode.label} mode.</b> {mode.text}</div>}

      {result && (
        <section className="panel stats">
          <div><span>Trip</span><b>{result.trip_id}</b></div>
          <div><span>GPS observations</span><b>{result.observations.length}</b></div>
          <div><span>Matched</span><b>{matched}</b></div>
          <div><span>Unmatched</span><b>{unmatched}</b></div>
          <div><span>Distinct segments</span><b>{segments}</b></div>
          <div><span>Decoder resets</span><b>{(result.diagnostics?.events || []).filter((e) => e.event.endsWith("_reset")).length}</b></div>
          <div><span>Decoder</span><b>{result.decoder || "not reported"}</b></div>
          <div>
            <span>Metrics</span>
            <b>{hasMetrics ? "" : "Not calculated"}</b>
            {hasMetrics && (
              <ul className="metrics">
                {Object.entries(result.metrics).map(([k, v]) => (
                  <li key={k}>{k}: {String(v)}</li>
                ))}
              </ul>
            )}
          </div>
        </section>
      )}

      {result && <p>Road segments are individual matches; intermediate roads are not displayed as a complete route.</p>}
      <MapView result={result} />

      {result && result.warnings.length > 0 && (
        <section className="panel">
          <h2>Warnings</h2>
          <ul>{result.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
        </section>
      )}
    </div>
  );
}
