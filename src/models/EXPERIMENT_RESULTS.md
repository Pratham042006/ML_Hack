# Distance-only experiments and Bayesian tuning

## Dataset and limitations

The first-review run used the 80 supplied training trips, with approximate labels produced by `monotone-reference-v1`. Of 208,417 observations, 69,372 had heuristic confident labels, 61,133 were ambiguous and 77,912 were unavailable. Among confident labels, 491 reference candidates were missing; 68,881 observations from 68 trips had a reference candidate present. Labels are inferred partly from proximity; agreement with them is not verified accuracy and can favor distance. Twelve trips contributed no candidate-present training observations. Neither student-test nor reserved evaluator trips were used.

## Five-model first-review comparison

Three-fold cross-validation kept trips together. All models used only `distance_m` and the same folds; logistic regression used `liblinear`. Macro F1 is candidate-level binary F1 averaged across folds. Top-1 includes the 491 candidate misses in its denominator; ambiguous and unavailable observations are excluded.

| Model | Top-1 vs. proxy incl. misses | Candidate macro F1 |
|---|---:|---:|
| Distance baseline (untrained) | 60.57% | — |
| Logistic regression | 60.57% | 0.698 |
| Gaussian naive Bayes | 60.52% | 0.745 |
| Decision tree | 60.39% | 0.730 |
| Random forest | 60.33% | 0.727 |
| Histogram gradient boosting | 60.50% | 0.727 |

These are existing actual first-review results, not new independent replication. Use the baseline for the demo unless a fair subsequent comparison supports a change. No learned model exceeded it on this proxy top-1 measure. Fold membership, counts, configurations and the frozen CSV hash are in `src/models/experiment_manifest.json`. Detailed fold metrics remain in local `outputs/review1/results.json`, which is ignored by Git. The reproducible comparator is `python -m src.models.review_comparison --output outputs/review1`; it reads `outputs/review1/alignment`, and overwrites that directory's `results.json`.

## Bayesian optimization (preliminary)

A new nested, trip-grouped search tunes GaussianNB's `var_smoothing`, selected here for its higher proxy candidate macro F1 in the first comparison. Eight trials in each search use a Gaussian-process surrogate and expected improvement over log10 smoothing in [-12, 0], minimizing mean inner-fold candidate-level binary log loss. Three outer folds compare tuned and default versions on identical held-out candidate-present observations. All preprocessing is fit inside each fold. These folds are based on candidate-row counts and differ from the first-review comparison folds; do not directly compare the two tables as matched-fold estimates.

| Mean across 3 outer folds | Untuned | Tuned |
|---|---:|---:|
| Candidate macro F1 (threshold 0.5) | 0.7464 | 0.7464 |
| Candidate binary log loss | 0.3939 | 0.3934 |
| Conditional observation top-1 | 61.79% | 61.76% |

These are unweighted fold means, not pooled or end-to-end accuracy. The inner search selected `var_smoothing=0.001` on all outer training folds and in the final training search. This small preliminary search improves binary log loss slightly but does not demonstrate a top-1 improvement; it does not prove GaussianNB is the final best model. Ambiguous labels, unavailable labels and candidate misses are excluded from this tuning table.

Run from repository root with training exports available:

```sh
python -m src.models.prepare_training outputs/review1/alignment --output outputs/model_work/training_checked.csv --manifest outputs/model_work/checked_manifest.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -W error::RuntimeWarning -m src.models.bayesian_tuning outputs/model_work/training_checked.csv --output outputs/model_work/bayesian_valid_results.json --artifact outputs/model_work/gaussian_nb_valid.joblib --trials 8
```

The output records trial values, inner and outer held-out trip IDs, metrics, data hash and the fitted artifact. The ignored local `outputs/model_work/checked_manifest.json` retains per-trip coverage and hashes. Share the actual exports and manifest securely with teammates if permitted; never commit the raw dataset or evaluator labels. Only load trusted joblib artifacts. The artifact is exploratory, not a validated production/final model.

## Remaining work

- Validate alignment quality and investigate unavailable or ambiguous samples with Suhaas; obtain official scoring guidance if available.
- Compare features beyond distance, such as heading agreement, using the same trip folds and data available at inference. The alignment itself uses distance, which risks favoring a distance-only scorer.
- Ask Prithika to independently verify the model plots and use the same data and folds for any further model comparisons.
- Evaluate a chosen model and routing pipeline on permitted held-out development data; report coverage and misses explicitly. Keep reserved evaluator trips unused until final evaluation.
