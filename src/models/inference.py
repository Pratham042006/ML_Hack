"""Score one trip's candidates without reading reference routes.

An artifact is a trusted joblib file produced by the training script, or the
loaded artifact dictionary. Load serialized artifacts only from trusted sources.
"""
from collections import defaultdict
from collections.abc import Mapping
import math
from pathlib import Path
from numbers import Real


BASELINE_DISTANCE_SCALE_M = 20.0


def _validate_candidates(candidates):
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("Each candidate must be a mapping")
        try:
            key = (candidate["observation_index"], candidate["arc_id"])
            distance = candidate["features"]["distance_m"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Each candidate needs observation_index, arc_id and features.distance_m") from exc
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in key):
            raise ValueError("observation_index and arc_id must be nonnegative integers")
        if key in seen:
            raise ValueError(f"Duplicate candidate for observation and arc: {key}")
        seen.add(key)
        if isinstance(distance, bool) or not isinstance(distance, Real) or not math.isfinite(distance) or distance < 0:
            raise ValueError("features.distance_m must be finite and nonnegative")


def _load_artifact(artifact):
    if isinstance(artifact, (str, Path)):
        import joblib
        artifact = joblib.load(artifact)
    if not isinstance(artifact, dict) or artifact.get("format_version") != 1:
        raise ValueError("Expected a version 1 PathSnap model artifact")
    if not artifact.get("feature_names") or "estimator" not in artifact:
        raise ValueError("Model artifact needs feature_names and estimator")
    return artifact


def score_candidates(candidates, model_artifact=None):
    """Return copied candidates with per-observation normalized scores.

    Baseline uses exp(-distance_m / 20), normalized separately for each GPS
    observation using a numerically stable calculation. It is not a trained
    model. For a trained artifact, positive-class binary probabilities are
    normalized across that observation's candidates. Neither transformation
    guarantees calibrated probabilities over the candidate set.

    Input must contain candidates for one trip only. Empty observations have no
    candidate records and consequently no scores; the decoder handles them.
    """
    candidates = list(candidates)
    _validate_candidates(candidates)
    if not candidates:
        return []

    groups = defaultdict(list)
    for i, candidate in enumerate(candidates):
        groups[candidate["observation_index"]].append(i)

    if model_artifact is None:
        # Subtract each group's smallest distance before exponentiation to avoid
        # all weights underflowing when all roads happen to be far away.
        weights = [0.0] * len(candidates)
        for indices in groups.values():
            minimum = min(float(candidates[i]["features"]["distance_m"]) for i in indices)
            for i in indices:
                distance = float(candidates[i]["features"]["distance_m"])
                weights[i] = math.exp(-(distance - minimum) / BASELINE_DISTANCE_SCALE_M)
    else:
        artifact = _load_artifact(model_artifact)
        feature_names = artifact["feature_names"]
        rows = []
        for candidate in candidates:
            features = candidate["features"]
            row = []
            for name in feature_names:
                value = features.get(name)
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                    raise ValueError(f"Missing or invalid required model feature: {name}")
                row.append(float(value))
            rows.append(row)
        estimator = artifact["estimator"]
        classes = list(estimator.classes_)
        if 1 not in classes:
            raise ValueError("Model artifact has no positive class 1")
        weights = [float(row[classes.index(1)]) for row in estimator.predict_proba(rows)]
        if any(not math.isfinite(v) or v < 0 or v > 1 for v in weights):
            raise ValueError("Model returned invalid probabilities")

    result = [dict(candidate) for candidate in candidates]
    for indices in groups.values():
        total = math.fsum(weights[i] for i in indices)
        # Equal weights are an explicit fallback if a model assigns zero to all.
        for i in indices:
            result[i]["score"] = weights[i] / total if total else 1.0 / len(indices)
    return result
