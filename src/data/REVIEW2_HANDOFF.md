# Data handoff after Review 1

The `review2-v1` export uses the existing monotone reference alignment and the
`distance_m` candidate feature. It contains approximate labels inferred from GPS
and the full ordered reference route. Candidate generation is independent. All
counts for candidate misses use confident inferred labels as the denominator.

| Partition | Trips processed | Observations | Confident | Ambiguous | Unavailable | No candidates | Inferred candidate misses |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training | 80 | 208,417 | 69,372 | 61,133 | 77,912 | 18,230 | 491 / 69,372 (0.71%) |
| Student-test | 10 | 19,160 | 9,118 | 8,529 | 1,513 | 0 | 16 / 9,118 (0.18%) |

Both batches completed with zero processing failures. The training CSV has
307,899 rows from 68,881 observations across 68 trips. There are 68,881 positive
and 239,018 negative candidate rows. Twelve training trips produced no confident
labels. Three hit the alignment memory guard, which withheld 31,662 labels.
Nine were withheld by the disconnected-reference rule, accounting for 28,036
labels. One of the three resource-limited trips also has a reference break, so
ten training references have a detected break in total. Another 18,214
training labels were unavailable because the selected segment was too distant.

## Reproduce the handoff

Run from the repository root with Python 3 and NumPy. Point the data root at
your own authorized student folder and choose fresh output directories.

```powershell
$dataRoot = "C:\Users\suhaas\Downloads\Student folder\Student folder"
python -m src.data.batch_prepare (Join-Path $dataRoot train) --partition training --output-dir outputs/review2-v1/training
python -m src.data.batch_prepare (Join-Path $dataRoot student_test) --partition student-test --output-dir outputs/review2-v1/student-test
python -m src.data.review_alignment (Join-Path $dataRoot train) outputs/review2-v1/training outputs/review2-v1/diagnostics
```

- Training CSV schema: `trip_id,observation_index,arc_id,distance_m,label`.
  A positive row identifies the confident inferred arc among candidates for
  that observation. Every included observation has exactly one positive.
- Each partition has a `manifest.json` with trip outcomes, counts, input
  hashes, configuration, source hashes and a dataset fingerprint. Every
  processed trip has `labels.jsonl`, `summary.json` and `provenance.json`.
  A failed trip is listed with its error and unknown observation count.
- The student-test directory provides proxy evaluation ledgers and has no
  training CSV. Its eligibility flag counts confident, candidate-present
  observations for coverage analysis. Keep student-test out of fitting and
  tuning, and reserved evaluator data outside development.
- The data and outputs are local and Git-ignored. Share the training export
  and full ledgers through the team's approved channel, or reproduce them using
  the commands and verify the hashes. Give both model owners the same snapshot
  and group cross-validation by `trip_id`. Metrics on the training CSV are
  conditional on a confident inferred label being present in the candidate set.

Training dataset SHA-256:
`877e73d2c2283c79cd5d5c4aad0ba449be88fec4d6455d0f2ae482714afb17ba`.

Training CSV SHA-256:
`8a8c08865db51ddddd19588fe244aafad82ccf8ffd8dad31963782c6d5c6a233`.

The original export ran on base commit `4505bc7` with the new batch script in
the working tree. The manifest records the actual script and pipeline hashes,
which make the run reproducible from the committed files in this handoff.

## Diagnostics and next interface decision

The diagnostics command writes local close-up maps and a JSON report. The report
records example IDs, selected and plausible route positions, reasons, threshold
sensitivity, and directed break positions. Threshold comparisons are training
diagnostics and use the same candidate set across settings. Neither section
experiments nor threshold variants feed the frozen export.

- Trip `00000001`, observation 5, has two plausible reference positions (1
  and 2) and a nearby parallel inference candidate (arc 117639). Observation
  1220 borders a GPS gap and has five plausible reference positions.
- Trip `00000018`, observation 178, selects route position 72. Its arc ID
  occurs at positions 72 and 94. Only the earlier occurrence is plausible at
  this observation under the complete-sequence score.
- On six training examples, changing the ambiguity-cost cutoff from 2 to 1
  increased confident labels from 3,256 to 3,983. Raising it to 3 reduced them
  to 2,887. The 35 m label-distance cutoff reduced confidence on trip
  `00000008` by 25 observations. These settings change label eligibility.
  The selected arcs stayed the same because these are classification cutoffs
  applied after path selection.
- Trip `00000002` has a directed break at position 54. GPS anchors support
  fitting the two sections over observations 0–715 and 1027–6690. The trial
  found 2,435 confident inferred labels and 4 candidate misses within those
  windows. It leaves 863 observations outside the windows, including the 311
  between them. This is exploratory evidence for team review. The frozen
  dataset retains the original unavailable-label policy for the entire trip.

Verification covered the existing 19 data tests, a synthetic batch with a
missing input followed by a valid trip, and checks of concatenated CSV
counts, hashes, class counts and train/student-test separation.

For routing, propose `projection_fraction` as candidate metadata outside
`features`. Compute the GPS point's orthogonal projection in the same metric
plane as `distance_m`, divide the along-segment position by segment length, and
clamp to [0, 1]. The fraction is 0 at the from-node and 1 at the to-node.
Reversing the same nondegenerate segment changes a fraction f to 1 − f.
Use null for a zero-length segment. Adithya can choose how to handle a null or
missing value while old candidate records are still supported. Agree on the
field and decoder behavior with Adithya, Pratham and Prithika before adding it
to the shared contract. The training feature list for this snapshot is
`distance_m`.

The supplied instructions still leave official observation-level scoring open.
Ask the organizers whether timestamped road labels or an official alignment
and scoring procedure are available. Until clarified, report pointwise
performance as agreement with inferred labels, with candidate misses and label
availability alongside it.
