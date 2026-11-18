"""Trip-grouped experiments on explicitly labeled candidate rows.

Only candidate-present observations with exactly one positive label are accepted.
This is conditional evaluation; it cannot measure candidate misses. See README.
"""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def make_models(seed):
    return {
        "logistic_regression": make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=500, random_state=seed)),
        "gaussian_naive_bayes": make_pipeline(SimpleImputer(), GaussianNB()),
        "decision_tree": make_pipeline(SimpleImputer(), DecisionTreeClassifier(max_depth=8, random_state=seed)),
        "random_forest": make_pipeline(SimpleImputer(), RandomForestClassifier(n_estimators=80, max_depth=12, n_jobs=-1, random_state=seed)),
        "hist_gradient_boosting": make_pipeline(SimpleImputer(), HistGradientBoostingClassifier(max_iter=80, random_state=seed)),
    }


def load_candidate_data(path, feature_names):
    data = pd.read_csv(path, dtype={"trip_id": str})
    required = {"trip_id", "observation_index", "arc_id", "label", *feature_names}
    if missing := required - set(data.columns):
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    if data.empty or data[list(required)].isna().any().any():
        raise ValueError("CSV must be non-empty and required columns cannot contain missing values")
    if not data["label"].isin([0, 1]).all():
        raise ValueError("label must be 0 or 1")
    keys = ["trip_id", "observation_index", "arc_id"]
    if data.duplicated(keys).any():
        raise ValueError("Duplicate trip/observation/arc candidate")
    positives = data.groupby(["trip_id", "observation_index"])["label"].sum()
    if not positives.eq(1).all():
        raise ValueError("Each included observation must have exactly one positive candidate; handle misses separately")
    for name in feature_names:
        data[name] = pd.to_numeric(data[name], errors="raise")
        if not np.isfinite(data[name]).all():
            raise ValueError(f"Non-finite feature: {name}")
    return data


def evaluate_fold(model, train, validation, feature_names):
    start = time.perf_counter()
    model.fit(train[feature_names].to_numpy(), train["label"].to_numpy())
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    probability = model.predict_proba(validation[feature_names].to_numpy())[:, list(model.classes_).index(1)]
    inference_seconds = time.perf_counter() - start
    y = validation["label"].to_numpy()
    winner_data = validation[["trip_id", "observation_index", "label"]].copy()
    winner_data["score"] = probability
    winner_index = winner_data.groupby(["trip_id", "observation_index"], sort=False)["score"].idxmax()
    return {
        "candidate_macro_f1_threshold_0_5": float(f1_score(y, probability >= 0.5, average="macro", zero_division=0)),
        "candidate_positive_f1_threshold_0_5": float(f1_score(y, probability >= 0.5, pos_label=1, zero_division=0)),
        "candidate_log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "conditional_observation_top1": float(accuracy_score(np.ones(len(winner_index)), winner_data.loc[winner_index, "label"])),
        "candidate_count": int(len(validation)),
        "observation_count": int(len(winner_index)),
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
    }


def run_experiments(data, feature_names, folds=3, seed=42):
    if data["trip_id"].nunique() < folds:
        raise ValueError("Need at least as many distinct trips as folds")
    splits = list(GroupKFold(n_splits=folds).split(data, groups=data["trip_id"]))
    results = {"scope": "candidate-present observations only; not end-to-end accuracy", "features": feature_names, "seed": seed, "folds": folds, "models": {}}
    for name, template in make_models(seed).items():
        records = []
        for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
            train = data.iloc[train_idx]
            valid = data.iloc[valid_idx]
            model = make_models(seed)[name]
            metrics = evaluate_fold(model, train, valid, feature_names)
            records.append({"fold": fold, "training_trips": sorted(train.trip_id.unique().tolist()), "validation_trips": sorted(valid.trip_id.unique().tolist()), **metrics})
        results["models"][name] = {"configuration": repr(template), "fold_results": records}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV of candidate-present, labeled observation-candidate pairs")
    parser.add_argument("--features", default="distance_m", help="Comma-separated inference feature names")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, help="Result JSON path")
    parser.add_argument("--refit-model", choices=list(make_models(42)), help="Optional explicit model to refit on all supplied trips")
    parser.add_argument("--artifact", help="Required with --refit-model; trusted joblib output path")
    args = parser.parse_args()
    if bool(args.refit_model) != bool(args.artifact):
        parser.error("--refit-model and --artifact must be supplied together")
    features = [name.strip() for name in args.features.split(",") if name.strip()]
    if not features or len(set(features)) != len(features):
        parser.error("--features must name unique features")
    if {"label", "trip_id", "observation_index", "arc_id"} & set(features):
        parser.error("Labels and trip-specific identifiers must not be model features")
    if "distance_m" not in features:
        parser.error("distance_m is required by the shared inference contract")
    data = load_candidate_data(args.csv, features)
    results = run_experiments(data, features, args.folds, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    if args.refit_model:
        model = make_models(args.seed)[args.refit_model]
        model.fit(data[features].to_numpy(), data["label"].to_numpy())
        Path(args.artifact).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"format_version": 1, "feature_names": features, "estimator": model, "model_name": args.refit_model}, args.artifact)
    print(f"Saved conditional cross-validation results to {args.output}")


if __name__ == "__main__":
    main()
