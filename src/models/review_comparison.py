"""Review 1 comparison on approximate training-trip labels; run from repo root."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from src.models.experiments import make_models
from src.models.inference import BASELINE_DISTANCE_SCALE_M


def measure(model, x, starts, group_sizes):
    start = perf_counter()
    if model is None:
        nearest = np.repeat(np.minimum.reduceat(x[:, 0], starts), group_sizes)
        raw = np.exp(-(x[:, 0] - nearest) / BASELINE_DISTANCE_SCALE_M)
    else:
        raw = model.predict_proba(x)[:, list(model.classes_).index(1)]
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite model scores")
    sums = np.repeat(np.add.reduceat(raw, starts), group_sizes)
    scores = np.divide(raw, sums, out=1.0 / np.repeat(group_sizes, group_sizes), where=sums > 0)
    maximum = np.repeat(np.maximum.reduceat(scores, starts), group_sizes)
    indices = np.arange(len(scores))
    winners = np.minimum.reduceat(np.where(scores == maximum, indices, len(scores)), starts)
    return raw, scores, winners, perf_counter() - start


def compare(output):
    folders = sorted((output / "alignment").iterdir())
    if len(folders) != 80:
        raise ValueError(f"Expected 80 training exports, found {len(folders)}")
    summaries = [json.loads((p / "summary.json").read_text()) for p in folders]
    provenance = [json.loads((p / "provenance.json").read_text()) for p in folders]
    if any(p["partition"] != "training" or p["verified_ground_truth"] for p in provenance):
        raise ValueError("Expected only approximate training exports")
    counts = pd.DataFrame({s["trip_id"]: s["counts"] for s in summaries}).T
    frames = [pd.read_csv(p / "approximate_candidates.csv", dtype={"trip_id": str}) for p in folders]
    data = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    keys = ["trip_id", "observation_index"]
    data.sort_values(keys + ["distance_m", "arc_id"], inplace=True, ignore_index=True)
    if data.duplicated(keys + ["arc_id"]).any() or not data.groupby(keys).label.sum().eq(1).all():
        raise ValueError("Expected unique candidates and exactly one positive per included observation")
    if len(data.groupby(keys)) != counts.training_eligible.sum():
        raise ValueError("Candidate CSVs and observation ledger disagree")

    # Split the complete trip inventory so missed-only trips contribute to Top-1.
    # GroupKFold weights the fold allocation by the number of eligible observations.
    groups = data[keys].drop_duplicates().trip_id.tolist()
    groups.extend(counts.index[counts.training_eligible == 0].tolist())
    splits = GroupKFold(n_splits=3).split(groups, groups=groups)
    models = make_models(42)
    # Use liblinear for this small binary feature set after LBFGS emitted local
    # floating-point warnings in the earlier trial.
    models["logistic_regression"].set_params(logisticregression__solver="liblinear")
    results = {name: [] for name in ("distance_baseline", *models)}
    all_test_trips = []
    for train, test in splits:
        train_trips, test_trips = {groups[i] for i in train}, {groups[i] for i in test}
        assert train_trips.isdisjoint(test_trips)
        all_test_trips.extend(test_trips)
        training, validation = (data[data.trip_id.isin(ids)] for ids in (train_trips, test_trips))
        x_train, y_train = training[["distance_m"]].to_numpy(), training.label.to_numpy()
        x_test, y_test = validation[["distance_m"]].to_numpy(), validation.label.to_numpy()
        group_sizes = validation.groupby(keys, sort=False).size().to_numpy()
        starts = np.r_[0, np.cumsum(group_sizes)[:-1]]
        eligible = len(starts)
        confident = int(counts.loc[list(test_trips), "confident"].sum())
        misses = int(counts.loc[list(test_trips), "candidate_misses"].sum())
        assert eligible + misses == confident
        for name, template in {"distance_baseline": None, **models}.items():
            model = None
            start = perf_counter()
            if template is not None:
                model = clone(template).fit(x_train, y_train)
            fit_s = perf_counter() - start if model is not None else None
            raw, scores, winners, score_s = measure(model, x_test, starts, group_sizes)
            correct = int(y_test[winners].sum())
            correct_by_trip = validation.iloc[winners].groupby("trip_id").label.sum().to_dict()
            results[name].append({
                "validation_trips": sorted(test_trips), "confident_labels": confident,
                "eligible_observations": eligible, "candidate_misses": misses,
                "correct": correct, "correct_by_trip": correct_by_trip,
                "top1_including_misses": correct / confident,
                "conditional_log_loss": float(-np.log(scores[y_test == 1].clip(1e-15, 1)).mean()),
                "candidate_macro_f1": float(f1_score(y_test, raw >= 0.5, average="macro")) if model is not None else None,
                "candidate_rows": len(y_test), "training_seconds": fit_s, "scoring_seconds": score_s,
            })
        print(f"Finished fold {len(results['distance_baseline'])}: {len(test_trips)} held-out trips, {confident:,} usable proxy labels", flush=True)
    assert sorted(all_test_trips) == sorted(counts.index)
    return {
        "method_version": provenance[0]["method_version"], "verified_ground_truth": False,
        "feature_names": ["distance_m"], "total_training_trips": len(folders),
        "trips_with_candidate_present_labels": int(counts.training_eligible.gt(0).sum()),
        "counts": {name: int(counts[name].sum()) for name in (
            "total_observations", "confident", "ambiguous", "unavailable",
            "candidate_misses", "no_candidates_all_observations", "training_eligible")},
        "per_trip_counts": counts.drop(columns=["candidate_miss_rate"]).astype(int).to_dict(orient="index"),
        "configurations": {name: str(model) for name, model in models.items()},
        "fold_results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/review1"))
    args = parser.parse_args()
    result = compare(args.output)
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
