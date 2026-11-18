# Additional training and demonstration configuration

Five classifiers were refitted using the same distance-plus-heading training export; each artifact stores the heading thresholds and fitted preprocessing. They were then compared on the ten student-test trips against approximate labels. All candidate scores were fixed before reading offline labels. The student-test partition was previously inspected in routing work and is development data, not an untouched final test. No evaluator data was used.

| Model | Proxy top-1, misses included | Candidate macro F1 | Candidate binary log loss |
|---|---:|---:|---:|
| Logistic regression | 83.30% | 0.860 | 0.256 |
| Gaussian naive Bayes | 83.48% | 0.856 | 0.275 |
| Decision tree | 83.03% | 0.857 | 0.249 |
| Random forest | 83.13% | 0.857 | 0.247 |
| Histogram gradient boosting | 83.51% | 0.859 | 0.245 |

Top-1 denominator: 9,118 confident inferred labels, including 16 candidate misses as failures. Candidate F1 and log loss are conditional on the inferred candidate being present. A further 8,529 ambiguous and 1,513 unavailable observations are excluded. The local logistic-regression run emitted numerical-library warnings despite finite output; reproduce that model in a clean environment.

## Bayesian search of histogram gradient boosting

A six-trial Gaussian-process expected-improvement search over `max_leaf_nodes` 7–63 minimized mean candidate binary log loss over the three frozen training-trip folds. The default is 31; search selected 15. Mean training-fold candidate macro F1 was 0.8374 untuned and 0.8396 tuned; mean binary log loss was 0.25758 and 0.25732. These folds also selected the tuned value, so the tuned CV comparison is selection-biased.

| Student-test proxy metric | Untuned | Tuned |
|---|---:|---:|
| Classifier top-1 including misses | 83.51% | 83.42% |
| Candidate macro F1 | 0.8593 | 0.8587 |
| Candidate binary log loss | 0.24498 | 0.24479 |
| Total scoring time, 10 local trips | 0.179 s | 0.159 s |

The timing is a single local measurement and excludes candidate generation and routing. Tuning slightly improved conditional candidate log loss, but not top-1 or macro F1. The earlier **untuned two-feature heading histogram model with projected Viterbi** was a provisional configuration: it achieved 85.44% agreement with confident inferred labels in the paired routing comparison, compared with 85.51% for distance + projected. The seven-agreement difference does not establish superiority. Both had 126 reset events. The routing score used a separately refitted, sorted-data histogram artifact, not the row-order artifact used for the classifier-only table above; see `RECONCILIATION.md`. The later four-feature model is the current provisional demo choice; see `FEATURE_EXPANSION.md`. Official accuracy and final model selection require stronger reference evidence.

## Reproduction

- `python -m src.models.fit_and_evaluate ...` fits five artifacts and scores permitted student-test proxies.
- `python -m src.models.tune_histogram ...` records the training-only Bayesian search.
- `python -m src.models.tuned_cv_comparison ...` and `python -m src.models.evaluate_tuning ...` compare frozen settings.
- `python -m src.models.feature_ablation ...` evaluates a third feature on training folds only.

Each script accepts `--help`; data, artifacts, detailed results and manifest files remain local under `outputs/`. Student-test references never enter inference. No weights or raw records are committed.
