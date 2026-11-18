"""Runs the prediction pipeline for ONE trip.

Real mode (all three teammate modules importable):
    load_trip(include_reference=False) -> generate_candidates -> score_candidates -> decode
Mock mode (any module missing): GPS points from .track + SYNTHETIC predictions, clearly labelled.

Safety rules enforced here:
  * include_reference is always False. If a loader still returns a "reference_route", it is dropped.
  * The response is rebuilt from a whitelist of fields, so reference labels cannot reach the UI.
  * Nothing is invented: a failed component raises PipelineError, it is never papered over.
"""
import importlib
import inspect
import math
from pathlib import Path
from typing import Optional

from src.api import config, mock_pipeline
from src.api.evaluation import compute_metrics

# component -> (module, function, parameter names the contract promises, in order)
CONTRACT = {
    "loader": ("src.data.pipeline", "load_trip", ["trip_dir", "include_reference"]),
    "candidates": ("src.data.pipeline", "generate_candidates", ["trip"]),
    "scorer": ("src.models.inference", "score_candidates", ["candidates", "model_artifact"]),
    "decoder": ("src.routing.decoder", "decode", ["observations", "scored_candidates", "network"]),
}

OBSERVATION_FIELDS = ("observation_index", "longitude", "latitude", "timestamp")
PREDICTION_FIELDS = ("observation_index", "arc_id", "geometry")


class PipelineError(Exception):
    """A pipeline step failed. The message says which component and what to fix."""


def resolve_components() -> tuple:
    """Return ({name: callable}, [problems], [signature warnings]). Safe to call any time."""
    found, problems, sig_warnings = {}, [], []
    for name, (module_name, func_name, expected) in CONTRACT.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # ImportError, SyntaxError, missing dependency inside teammate code...
            problems.append(f"{module_name} could not be imported ({type(exc).__name__}: {exc})")
            continue
        func = getattr(module, func_name, None)
        if not callable(func):
            problems.append(f"{module_name}.{func_name} is missing")
            continue
        actual = list(inspect.signature(func).parameters)
        if actual[: len(expected)] != expected:
            sig_warnings.append(
                f"{module_name}.{func_name} has parameters {actual} but the contract says {expected}"
            )
        found[name] = func
    return found, list(dict.fromkeys(problems)), sig_warnings


def _clean(value):
    """Make a value JSON-safe: numpy -> python, NaN/inf -> None, tuples -> lists."""
    if hasattr(value, "item") and not isinstance(value, (list, tuple, dict)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "tolist"):
        return _clean(value.tolist())
    return value


def _whitelist(records: list, fields: tuple) -> list:
    return [{f: _clean(rec.get(f)) for f in fields} for rec in records]


def _validate(observations: list, predictions: list) -> None:
    if len(predictions) != len(observations):
        raise PipelineError(
            f"Decoder returned {len(predictions)} records for {len(observations)} observations. "
            "It must return exactly one record per observation (ask the routing owner)."
        )
    for i, pred in enumerate(predictions):
        if pred.get("observation_index") != i:
            raise PipelineError(
                f"Decoder record {i} has observation_index={pred.get('observation_index')!r}; "
                "records must be in the same order as the observations."
            )
        if pred.get("arc_id") is None and pred.get("geometry"):
            raise PipelineError(f"Decoder record {i} has no arc_id but has geometry.")


def run_prediction(trip_dir: Path, model_artifact: Optional[str]) -> dict:
    trip_dir = Path(trip_dir)
    warnings = []
    components, problems, sig_warnings = resolve_components()
    warnings.extend(sig_warnings)

    if problems:
        observations = mock_pipeline.load_observations(trip_dir)
        predictions = mock_pipeline.synthetic_predictions(observations)
        warnings.insert(0, "MOCK MODE: predictions are synthetic and are NOT road matches or results.")
        warnings.extend(f"Not integrated yet: {p}" for p in dict.fromkeys(problems))
        return {
            "observations": _whitelist(observations, OBSERVATION_FIELDS),
            "predictions": _whitelist(predictions, PREDICTION_FIELDS),
            "metrics": {},
            "diagnostics": {"events": [], "matched_blocks": []},
            "decoder": "mock",
            "mode": "mock",
            "warnings": warnings,
        }

    if model_artifact and not Path(model_artifact).is_file():
        raise PipelineError(
            f"Model artifact '{model_artifact}' was not found. Fix PATHSNAP_MODEL_ARTIFACT "
            "or unset it to use the baseline scorer."
        )

    try:
        trip = components["loader"](str(trip_dir), include_reference=False)
    except Exception as exc:
        raise PipelineError(f"Data loader (load_trip) failed: {type(exc).__name__}: {exc}") from exc
    if "reference_route" in trip:
        trip = {k: v for k, v in trip.items() if k != "reference_route"}
        warnings.append("Loader returned a reference_route with include_reference=False; it was dropped.")
    observations, network = trip["observations"], trip["network"]

    try:
        candidates = components["candidates"](trip)
        if model_artifact:
            from src.models.inference import _load_artifact
            artifact = _load_artifact(model_artifact)
            if "heading_agreement" in artifact["feature_names"]:
                from src.data.heading import HeadingConfig, add_heading_features
                settings = artifact.get("heading_config")
                if settings is None:
                    warnings.append("Legacy heading model: using heading-v1 defaults (5 m, 10 s).")
                    settings = {"min_displacement_m": 5.0, "max_gap_seconds": 10.0}
                if {'distance_margin_m', 'projection_fraction', 'candidate_rank', 'segment_length_m'} & set(artifact['feature_names']):
                    from src.data.candidate_features import add_candidate_features
                    candidates = add_candidate_features(trip, candidates, HeadingConfig(**settings))
                else:
                    candidates = add_heading_features(trip, candidates, HeadingConfig(**settings))
    except Exception as exc:
        raise PipelineError(f"Candidate generation failed: {type(exc).__name__}: {exc}") from exc

    covered = {c["observation_index"] for c in candidates}
    no_candidates = len(observations) - len(covered)
    if no_candidates:
        warnings.append(f"{no_candidates} of {len(observations)} observations had no road candidates.")

    try:
        scored = components["scorer"](candidates, model_artifact=model_artifact)
    except Exception as exc:
        raise PipelineError(f"Candidate scoring failed: {type(exc).__name__}: {exc}") from exc

    try:
        from src.routing.decoder import decode_with_diagnostics
        from src.routing.map_diagnostics import matched_blocks
        if config.DECODER == "projected":
            from src.data.candidate_projection import project_candidate_positions
            from src.routing.projected import ProjectedDistance, decode_projected
            positions = project_candidate_positions(trip, candidates)
            predictions, events = decode_projected(observations, scored, network, distance=ProjectedDistance(positions))
        elif config.DECODER == "midpoint":
            predictions, events = decode_with_diagnostics(observations, scored, network)
        else:
            raise ValueError("PATHSNAP_DECODER must be midpoint or projected")
        blocks = matched_blocks(predictions, events)
    except Exception as exc:
        raise PipelineError(f"Route decoding failed: {type(exc).__name__}: {exc}") from exc
    _validate(observations, predictions)

    mode = "trained" if model_artifact else "baseline"
    warnings.append("Predictions are matched segments; intermediate roads are not expanded.")
    return {
        "observations": _whitelist(observations, OBSERVATION_FIELDS),
        "predictions": _whitelist(predictions, PREDICTION_FIELDS),
        "metrics": compute_metrics(trip_dir.name, predictions),
        "diagnostics": {"events": _clean(events), "matched_blocks": _clean(blocks)},
        "decoder": config.DECODER,
        "mode": mode,
        "warnings": warnings,
    }
