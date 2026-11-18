"""PathSnap FastAPI app. Run from the repository root:

    python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""
import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api import config
from src.api.runner import PipelineError, resolve_components, run_prediction
from src.api.trips import DataDirError, list_trip_ids, trip_path

log = logging.getLogger("uvicorn.error")  # shows up in the uvicorn console next to access logs

app = FastAPI(title="PathSnap API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    client = request.client.host if request.client else "?"
    log.info("%s %s %s -> %s (%.2fs)", client, request.method, request.url.path,
             response.status_code, time.perf_counter() - started)
    return response


def _available_trips() -> list:
    try:
        return list_trip_ids(config.DATA_DIR)
    except DataDirError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/trips")
def get_trips():
    return {"trip_ids": _available_trips()}


@app.get("/status")
def get_status():
    """Debug helper for integration: which real components are importable right now."""
    components, problems, sig_warnings = resolve_components()
    return {
        "data_dir": str(config.DATA_DIR),
        "model_artifact": config.MODEL_ARTIFACT,
        "components_found": sorted(components),
        "components_missing": problems,
        "signature_warnings": sig_warnings,
    }


@app.post("/predict")
def predict(request: PredictRequest):
    available = _available_trips()
    try:
        folder = trip_path(config.DATA_DIR, request.trip_id, available)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown trip_id '{request.trip_id}'. Call GET /trips for the available IDs.",
        )
    started = time.perf_counter()
    try:
        result = run_prediction(folder, config.MODEL_ARTIFACT)
    except PipelineError as exc:
        log.error("predict %s failed: %s", request.trip_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))
    log.info("predict %s: mode=%s observations=%d predictions=%d warnings=%d in %.2fs",
             request.trip_id, result["mode"], len(result["observations"]),
             len(result["predictions"]), len(result["warnings"]), time.perf_counter() - started)
    return {"trip_id": request.trip_id, **result}
