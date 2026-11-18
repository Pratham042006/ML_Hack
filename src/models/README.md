# Pratham's candidate-scoring component

Status: **integration baseline and conditional experiment framework**. No real-data model has been trained or evaluated here. The training label alignment to `.route` remains unresolved and must be agreed with the team/organizers. Synthetic runs verify code only and are not project results.

## Inference handoff

From the repository root:

```python
from src.models.inference import score_candidates
scored = score_candidates(candidates)  # explicit distance baseline
# scored = score_candidates(candidates, "path/to/trusted_model.joblib")
```

Input follows `docs/INTEGRATION_CONTRACT.md`: candidates from **one trip** with `observation_index`, trip-local `arc_id`, `features.distance_m` and geometry. Output retains input order/fields and adds `score`. Scores sum to one per observation; an empty input returns an empty list. No route/reference input is read. Missing candidates are represented by absence from the list and handled by the decoder.

Baseline weight is `exp(-distance_m / 20)` normalized per observation (numerically stabilized). Twenty metres is an initial integration setting, **not a tuned value**. For a trained model, positive-class binary scores are normalized across each observation's candidates; a group of all-zero model scores falls back to equal weights. These scores are not guaranteed calibrated candidate probabilities. Missing or invalid required features cause an error. Never load untrusted joblib artifacts.

## Conditional experiment framework

`experiments.py` compares five models with trip-grouped CV: logistic regression, Gaussian naive Bayes, decision tree, random forest and histogram gradient boosting. It logs per-fold trip membership, candidate-level binary macro/positive F1 at a 0.5 threshold, candidate-level log loss, conditional observation top-1 accuracy, counts and timings. It is an initial framework: it **does not** calculate candidate misses, end-to-end accuracy, route continuity, student-test comparisons or Bayesian optimization yet. It does not establish actual observation labels. Do not present its conditional accuracy as overall accuracy.

Input CSV: one row per candidate with columns `trip_id,observation_index,arc_id,distance_m,label` and optionally additional named numeric features. The `label` is 1 for the selected reference candidate and 0 for others. Each included observation must have exactly one positive candidate. Record label provenance separately. Observations with unavailable/ambiguous labels or absent reference candidates cannot be represented by this CSV; a future full evaluation must count and report them separately. The feature columns must be available at inference and must not encode route references, labels, trip IDs or arc IDs.

Example from repository root using **synthetic data only**:

```sh
python -m unittest src.models.test_models
python -m src.models.synthetic_fixture /tmp/pathsnap_synthetic_candidates.csv
python -m src.models.experiments /tmp/pathsnap_synthetic_candidates.csv --folds 3 --output /tmp/pathsnap_synthetic_results.json --refit-model logistic_regression --artifact /tmp/pathsnap_synthetic_model.joblib
```

Do not report the fixture's metrics as actual performance. With valid real training labels, replace the fixture path and record label provenance and partition separately. All learned preprocessing is fitted inside each CV training fold. Trip IDs are saved in the output for auditing. Refitting is explicit and must only use permitted training trips; the tool cannot infer partition authorization from a CSV. Do not select or tune models using reserved evaluator data.

## Outstanding integration and research

- Agree with Suhaas on feature names and an independently reviewed observation-to-reference alignment; obtain organizer's scoring guidance.
- Extend evaluation to count candidate misses and unlabeled observations, use the student-test partition appropriately and compute sequence metrics with Adithya.
- Implement a recorded Bayesian search after actual model comparison; save trials and compare tuned with untuned models.
- Coordinate dependencies and trained artifact location with Prithika. The starter root requirements already include numpy, pandas and scikit-learn (which depends on joblib).
