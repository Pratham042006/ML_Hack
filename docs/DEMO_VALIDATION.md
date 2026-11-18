# Demo validation

On 18 November 2026, the user confirmed that a second laptop opened the UI, loaded the trip list from the first laptop's API and displayed a prediction. The API logs showed requests from a different LAN client, including a successful `GET /trips` and `POST /predict` for training trip `00000005`: trained scoring, projected Viterbi, 1,045 observations and 1,045 predictions. This was a development demonstration, not final evaluator testing.

Configuration: API bound to `0.0.0.0:8003`, `PATHSNAP_DATA_DIR` pointed to the local permitted training partition, `PATHSNAP_DECODER=projected`, and `PATHSNAP_MODEL_ARTIFACT` pointed to the local four-feature artifact described in `src/models/FEATURE_EXPANSION.md`. The Vite UI was served on `0.0.0.0:5174`; the second laptop used the backend laptop's LAN address and port 8003. IP addresses are network-dependent and are intentionally not hardcoded.

Separate local checks exercised baseline mode, trained mode, invalid-trip HTTP 404, missing model error, candidate/decoder interfaces, route reset events and UI production build. Reset markers represent block boundaries; road segments are not expanded paths. Metrics remain empty in prediction responses until a defensible server-side evaluation is supplied.

At the time of testing, the UI dependency audit still reported four high-severity entries through the `picomatch` development-tool chain. Vite was updated from 5 to 6.4.6 and rebuilt; the registry's latest `picomatch` was still in the advisory range. Keep the Vite development server on a trusted local network, do not expose it publicly, and update when a compatible fixed dependency is available. The official scoring rule remains to be confirmed with the organizers.
