# Candidate-distance margin ablation

A training-fold-only exploratory test evaluated `distance_margin_m = distance_m - minimum_candidate_distance_m` alongside distance and heading. It used the frozen heading-v1 training rows, the same three trip folds, and a histogram-gradient-boosting classifier. No student-test or evaluator data was used to select this feature.

| Features | Correct among candidate-present inferred labels | Conditional agreement |
|---|---:|---:|
| Distance and heading | 55,483 / 68,881 | 80.55% |
| Distance, heading and margin | 55,734 / 68,881 | 80.91% |

This is a 251-observation (0.36 percentage point) increase on approximate labels. Candidate misses and unavailable labels are excluded; it is not end-to-end accuracy. The feature is not in the shared inference pipeline or chosen demo artifact. Validate it on appropriate training splits and review generalization before deployment. Reproduce with `src.models.feature_ablation` and the local frozen manifests; generated results remain under `outputs/`.
