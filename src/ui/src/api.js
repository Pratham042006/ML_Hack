// All HTTP calls to the PathSnap API live here. The UI only ever talks to the API,
// never to model files or reference labels.

const STORAGE_KEY = "pathsnap_api_url";
const FALLBACK_URL = "http://localhost:8000";

export function defaultApiUrl() {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved) return saved;
  } catch (e) {
    /* storage unavailable: ignore */
  }
  return import.meta.env.VITE_API_URL || FALLBACK_URL;
}

export function saveApiUrl(url) {
  try {
    window.localStorage.setItem(STORAGE_KEY, url);
  } catch (e) {
    /* ignore */
  }
}

function clean(url) {
  return url.trim().replace(/\/+$/, "");
}

async function request(baseUrl, path, options) {
  let response;
  try {
    response = await fetch(`${clean(baseUrl)}${path}`, options);
  } catch (e) {
    throw new Error(
      `Cannot reach the API at ${clean(baseUrl)}. Check that the backend is running, the IP and port ` +
        `are correct, both laptops are on the same network, and the firewall allows the port.`
    );
  }
  let body = null;
  try {
    body = await response.json();
  } catch (e) {
    /* non-JSON body */
  }
  if (!response.ok) {
    let detail = body && body.detail;
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
    throw new Error(detail || `API returned HTTP ${response.status}.`);
  }
  return body;
}

export async function fetchTrips(baseUrl) {
  const body = await request(baseUrl, "/trips");
  return body.trip_ids || [];
}

export async function runPrediction(baseUrl, tripId) {
  return request(baseUrl, "/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trip_id: tripId }),
  });
}
