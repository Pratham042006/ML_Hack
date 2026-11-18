# Heading feature comparison

Heading v1 compares the previous GPS movement with the candidate arc direction. It uses a 5 m minimum displacement and a positive gap of at most 10 seconds. Missing headings are `None` at inference and blank in the CSV, then imputed by the model pipeline fitted within each training fold. This feature is generated for a heading artifact in the API and paired-routing tool; baseline inference remains distance-only.

Three fixed trip-grouped folds, with identical labels, observations, candidates and ordering, compare distance only and distance plus heading. The source snapshot has 208,417 observations, 69,372 confident inferred labels and 491 candidate misses. The top-1 denominator includes those 491 misses; 61,133 ambiguous and 77,912 unavailable observations are excluded. Heading is missing for 9,100 of 68,881 candidate-present observations; all remain included. The GPS/network input hashes for all 80 trips match the original alignment provenance.

| Model | Distance only | Distance + heading |
|---|---:|---:|
| Distance baseline | 60.57% | — |
| Logistic regression | 60.57% | 79.84% |
| Gaussian naive Bayes | 60.52% | 79.86% |
| Decision tree | 60.39% | 79.66% |
| Random forest | 60.33% | 79.83% |
| Histogram gradient boosting | 60.50% | 79.98% |

These are agreement with approximate labels, not verified accuracy. The reference alignment uses proximity and has coverage/selection bias; improvements require independent validation on permitted held-out data. These scores include no decoder. No evaluator data was used. On this machine logistic regression produced floating-point runtime warnings in matrix multiplication although output probabilities were finite; independently reproduce the logistic result in a clean numerical environment. Other models completed without those warnings.

A provisional histogram-gradient-boosting model was refitted on the eligible training rows and saved locally as `outputs/model_work/heading_comparison_v1/provisional_heading_model.joblib`. It is not a final verified model. Local `results.json` contains per-fold counts and scores; `routing_check.json` contains a paired routing integration check on training trip 00000005. API integration with the artifact also passed. Only load trusted joblib files.

Run from the repository root with the heading export and frozen trip manifest available:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.models.heading_experiment outputs/heading-v1-frozen/training_candidates_heading_v1.csv --heading-manifest outputs/heading-v1-frozen/manifest.json --fold-manifest src/models/experiment_manifest.json --output-dir outputs/model_work/heading_comparison_v1_new
```

The heading model and generated datasets are local and ignored by Git. The previous distance-only snapshot has not been altered. Coordinate any new feature schema or threshold change with the team and use the same settings during training and inference.
