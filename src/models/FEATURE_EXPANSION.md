# Four- and six-feature experiments

The original model uses distance and heading agreement. Two further features are now implemented: `distance_margin_m`, the candidate's distance minus the nearest candidate's distance, and `projection_fraction`, the GPS projection along the directed segment. The six-feature experiment also adds `candidate_rank` (one-based distance/arc-ID order within the observation) and `segment_length_m` (the candidate arc's endpoint-chord length in the local metric plane). These use inference-available GPS and network data, not reference routes.

The new exports preserve the frozen rows, labels and trip folds. Heading and projection can be missing; the estimator's imputer is fitted inside each training fold. Input hashes were verified against the heading export; original snapshot remains unchanged. Logistic regression produced numerical-library warnings despite finite outputs and needs clean-environment reproduction.

| Histogram model | Training CV proxy top-1 incl. misses | Student-test classifier proxy top-1 | Student-test projected proxy top-1 |
|---|---:|---:|---:|
| Two features | 79.98% | 83.36% | 85.44% |
| Four features | 80.84% | 84.42% | 86.05% |
| Six features | 80.85% | 84.00% | 85.42% |

Training CV denominator is 69,372 confident inferred labels; student-test denominator is 9,118. Candidate misses count as incorrect. Ambiguous and unavailable observations are excluded. These are agreement with approximate labels, not verified accuracy. Student-test has been inspected and used for development comparisons, so is not untouched; evaluator data was not used. All five models were tested on each training CV feature set, and details are in `results/feature_expansion_summary.json`.

In the separate four-feature classifier-only student-test comparison, Gaussian naive Bayes scored 84.88% top-1 compared with histogram boosting at 84.42%; histogram boosting had lower candidate log loss (0.169 versus 0.386) and higher training-CV top-1. These are approximate-label results, not verified accuracy.

The practical demo configuration is the untuned **four-feature histogram-gradient-boosting model with projected Viterbi**: it uses fewer features than the six-feature alternative and performed better on the inspected development comparison. This is provisional, not proof of final generalization. The four-feature model used in the successful two-laptop test remains local at `outputs/model_work/four_feature_comparison_v1/provisional_four_feature_model.joblib`. The six-feature model and export also remain local and can be used experimentally through the same inference pipeline.

Six-trial Bayesian optimization of the four-feature histogram model chose 15 max leaf nodes, versus the default 31. On the same student-test proxy labels, classifier-only top-1 was 84.42% untuned and 84.68% tuned; candidate F1 was 0.8994 and 0.8984, and candidate binary log loss was 0.16927 and 0.16988. With projected routing on those same trips, untuned and tuned models had 7,846/9,118 (86.05%) and 7,844/9,118 (86.03%) proxy agreements respectively. Training-fold selection and model choices are not independent of these development analyses. Do not claim a final accuracy from this comparison.

The four- and six-feature export scripts preserve the parent snapshot and the model scripts use frozen training-trip folds; use their `--help` arguments to reproduce. Artifacts and raw/derived records remain local and must be shared only through an approved channel. See `src/models/RECONCILIATION.md` for unresolved differences between independent routing reports.
