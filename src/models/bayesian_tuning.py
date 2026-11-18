"""Nested, trip-grouped Bayesian search for GaussianNB variance smoothing.

Input must be a permitted candidate-present training export; labels are proxies.
Outer scores estimate the tuning procedure, not verified end-to-end accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import norm
from scipy.linalg import solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import GroupKFold

from src.models.experiments import evaluate_fold, load_candidate_data, make_models


def model_for(log_smoothing):
    return make_models(42)["gaussian_naive_bayes"].set_params(
        gaussiannb__var_smoothing=10.0 ** float(log_smoothing))


def search(data, trials, seed):
    if trials < 4:
        raise ValueError("At least four trials required")
    groups = data.trip_id.to_numpy()
    n_splits = min(3, data.trip_id.nunique())
    if n_splits < 2:
        raise ValueError("Bayesian search requires at least two training trips")
    splits = list(GroupKFold(n_splits=n_splits).split(data, groups=groups))
    rng = np.random.RandomState(seed)
    domain = np.linspace(-12, 0, 241)
    tried, losses, records = [], [], []
    initial = [-12.0, -9.0, -6.0, -3.0]
    for iteration in range(trials):
        if iteration < len(initial):
            x = initial[iteration]
        else:
            gp = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=2.0, length_scale_bounds="fixed", nu=2.5)
                + WhiteKernel(noise_level=1e-4, noise_level_bounds="fixed"),
                normalize_y=False, optimizer=None, random_state=seed)
            gp.fit(np.array(tried).reshape(-1, 1), np.array(losses))
            points = domain.reshape(-1, 1)
            covariance = gp.kernel_(points, gp.X_train_)
            mu = np.einsum("ij,j->i", covariance, gp.alpha_)
            projection = solve_triangular(gp.L_, covariance.T, lower=True)
            variance = np.maximum(gp.kernel_.diag(points) - np.einsum("ij,ij->j", projection, projection), 0.0)
            sigma = np.sqrt(variance)
            if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
                raise ValueError("Nonfinite Gaussian-process prediction")
            improvement = min(losses) - mu - 0.001
            z = np.divide(improvement, sigma, out=np.zeros_like(mu), where=sigma > 0)
            expected = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            for previous in tried:
                expected[np.isclose(domain, previous)] = -np.inf
            best = np.flatnonzero(expected == np.max(expected))
            x = float(domain[rng.choice(best)])
        fold_results = []
        for train_idx, valid_idx in splits:
            train, valid = data.iloc[train_idx], data.iloc[valid_idx]
            if train.label.nunique() != 2:
                raise ValueError("An inner training fold lacks both candidate classes")
            result = evaluate_fold(model_for(x), train, valid, ["distance_m"])
            fold_results.append({"validation_trips": sorted(valid.trip_id.unique().tolist()), **result})
        loss = float(np.mean([f["candidate_log_loss"] for f in fold_results]))
        tried.append(x)
        losses.append(loss)
        records.append({"trial": iteration + 1, "log10_var_smoothing": x,
                        "var_smoothing": 10 ** x, "mean_candidate_log_loss": loss,
                        "fold_results": fold_results})
        print(f"  trial {iteration + 1}/{trials}: log10 smoothing={x:.2f}, loss={loss:.5f}", flush=True)
    best = int(np.argmin(losses))
    return tried[best], records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = Path(args.csv)
    data = load_candidate_data(path, ["distance_m"])
    if data.trip_id.nunique() < 4:
        raise ValueError("Need at least four distinct trips for nested validation")
    outer = GroupKFold(n_splits=3)
    records = []
    for i, (train_idx, valid_idx) in enumerate(outer.split(data, groups=data.trip_id), 1):
        train, valid = data.iloc[train_idx], data.iloc[valid_idx]
        print(f"Outer fold {i}: {train.trip_id.nunique()} training trips, {valid.trip_id.nunique()} held out", flush=True)
        best, trials = search(train, args.trials, args.seed + i)
        untuned = evaluate_fold(model_for(-9), train, valid, ["distance_m"])
        tuned = evaluate_fold(model_for(best), train, valid, ["distance_m"])
        records.append({"fold": i, "training_trips": sorted(train.trip_id.unique().tolist()),
                        "validation_trips": sorted(valid.trip_id.unique().tolist()),
                        "best_log10_var_smoothing": best, "inner_trials": trials,
                        "untuned": untuned, "tuned": tuned})
    print("Final search on all supplied training trips", flush=True)
    best, trials = search(data, args.trials, args.seed)
    estimator = model_for(best).fit(data[["distance_m"]].to_numpy(), data.label.to_numpy())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"method": "Gaussian process Bayesian optimization, expected improvement; nested trip-grouped CV",
              "optimization_target": "mean inner-fold candidate-level binary log loss (approximate labels)",
              "scope": "candidate-present observations with heuristic confident proxy labels only; not end-to-end accuracy",
              "csv_sha256": sha, "n_trips": int(data.trip_id.nunique()), "n_rows": int(len(data)),
              "seed": args.seed, "search_log10_bounds": [-12, 0], "outer_folds": records,
              "final_log10_var_smoothing": best, "final_trials": trials}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    Path(args.artifact).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"format_version": 1, "feature_names": ["distance_m"], "estimator": estimator,
                 "model_name": "gaussian_naive_bayes_bayesian", "training_csv_sha256": sha,
                 "label_source": "approximate_reference_alignment", "verified_ground_truth": False}, args.artifact)
    print(f"Saved {args.output} and {args.artifact}", flush=True)


if __name__ == "__main__":
    main()
