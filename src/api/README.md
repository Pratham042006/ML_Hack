# src/api (owner: Prithika)

FastAPI backend. It is the assembly point: it calls the data loader, scorer and decoder in order.

## Run (from the repository root, with the venv active)

    pip install fastapi "uvicorn[standard]" pydantic python-dotenv
    python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Settings come from environment variables or a local `.env` (see the root `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `PATHSNAP_DATA_DIR` | `data/student_test` | folder with one sub-folder per trip |
| `PATHSNAP_MODEL_ARTIFACT` | unset | saved model; unset means baseline scoring |
| `PATHSNAP_CORS_ORIGINS` | `*` | comma-separated UI origins |

## Endpoints
- `GET /trips` -> `{"trip_ids": [...]}` (read from the data dir, never hardcoded)
- `POST /predict` `{"trip_id": "00000052"}` -> `{trip_id, observations, predictions, metrics, mode, warnings}`
- `GET /status` -> which teammate modules are importable right now (debug aid)

## Modes (honest labels)
- `mock`: any teammate module is missing. GPS points are real, predictions are SYNTHETIC. Not a result.
- `baseline`: all three modules integrated, no model artifact set.
- `trained`: all three modules integrated and `PATHSNAP_MODEL_ARTIFACT` points to a real file.
The API does not verify whether the decoder is real Viterbi; read the routing owner's notes.

## Safety rules enforced in code
- `.route` is never read here; the loader is always called with `include_reference=False`.
- Responses are rebuilt from a whitelist of fields (observation_index/longitude/latitude/timestamp
  and observation_index/arc_id/geometry), so reference labels cannot reach the UI.
- `metrics` is `{}` (not calculated) until a defensible evaluation exists (`evaluation.py`).
- Errors: 404 unknown trip, 503 bad data dir, 500 with a message naming the failing component.

## Integration check
    python -m src.api.integration_check
    python -m src.api.integration_check --api-url http://<laptop1-ip>:8000
