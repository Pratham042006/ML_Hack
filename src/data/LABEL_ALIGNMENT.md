# Reference alignment investigation (not verified labels)

## What the supplied files establish

`.track` gives longitude, latitude and timestamp per GPS observation. `.route`
is an ordered sequence of trip-local traversed arc IDs, without timestamps or
observation indices. Row-wise zipping is invalid. The supplied PDFs require
observation-level metrics but do not specify the corresponding alignment.
Suhaas confirmed no further organizer information is available.

Run this diagnostic on a permitted training/student-test trip only:

```powershell
python -m src.data.reference_analysis "C:\Users\suhaas\Downloads\Student folder\Student folder\train\00000001"
```

Actual trip 00000001: 1,315 GPS observations, 525 route entries, 525 unique
reference arc IDs, zero directed endpoint-continuity breaks, and zero adjacent
route pairs lacking a shared node. This supports investigating ordered route
geometry, but does not establish the vehicle's position/time on it. Other trips
may repeat arc IDs or contain discontinuities.

For 1,313 observations, at least one inference candidate appears somewhere in
the reference route. This unordered set overlap ignores occurrence/time and is
NOT pointwise accuracy or candidate recall. The two remaining observations
cannot automatically be called true candidate misses. No labels are produced.

## Initial design and future refinements

1. Build the ordered reference path using arc occurrences, preserving repeated
   IDs as distinct positions. Diagnose direction, disconnected pairs, zero-length
   arcs and absent references first.
2. Project GPS points onto plausible segments of that path, keeping multiple
   possible occurrences at intersections/loops.
3. Use monotone dynamic programming over route position, combining GPS-to-segment
   distance with plausible progress based on timestamps and travel distance.
   Allow stopping and skipping multiple arcs between samples. Do not force
   first/last GPS observations to the route's endpoints without evidence.
4. Flag distant observations, disconnected references, near-tied alternatives,
   shared endpoints, reverse/parallel roads and large GPS gaps. Do not force a
   unique label where evidence is insufficient. Record and validate all chosen
   penalties/confidence criteria on permitted data; none is validated yet.
5. If used, call the result **approximate inferred reference alignment** and
   preserve method version, route occurrence, residual and uncertainty in a
   separate provenance sidecar, never in inference features or the UI.

Reference alignment should not be restricted to inference candidates: that
would hide reference candidate misses. Even independent reference-path alignment
inherits GPS/reference errors and is not verified ground truth. The initial
diagnostic did not create labels; v1 below now implements a simple version.
Speed/progress likelihoods and handling disconnected components remain future work.

## Implemented v1: scoring, uncertainty and outputs

`alignment.py` uses occurrence positions in the FULL ordered reference sequence,
not unique arc IDs. Every observation is compared with every route position;
inference candidates and model scores are not inputs. Distances use the same
approximate local metric projection as the pipeline. This is a reference-label
tool, not the inference HMM owned by Adithya.

For observation t and position j:

```text
E(t,j) = (min(distance_to_segment_m(t,j), 100) / 20)^2
F(t,j) = E(t,j) + min over k <= j [F(t-1,k) + 0.05 * max(j-k-1, 0)]
```

Start/end route positions are free. Staying and moving to the next position
cost zero; skipping pays 0.05 per skipped occurrence. The 100m cap limits extreme
outlier influence. No speed/elapsed-time likelihood is used in v1. These are
untuned heuristic settings, not organizer requirements.

An exact backward pass computes the cheapest COMPLETE path through each state.
An alternative position is plausible if its complete path cost is within 2.0
of the optimum. This detects competing alignments, including repeated arc
occurrences, rather than just checking the nearest two segments.

- Unavailable: selected residual >50m; missing/empty reference; disconnected or
  direction-inconsistent reference; or resource limit exceeded.
- Ambiguous: otherwise, more than one plausible route position; or adjacent to
  a GPS interval >10s or a duplicate timestamp.
- Confident: none of these flags. This is heuristic confidence, not a calibrated
  probability or verified ground truth. Distant/ambiguous tentative arc IDs are
  diagnostic only, not accepted training labels.

V1 conservatively makes an entire trip unavailable if its reference has a
connectivity break; no artificial bridge or reverse arc is introduced. Runtime
and memory are O(observations * route entries), capped at 10,000,000 cells.

Run from the repository root:

```powershell
python -m src.data.align_trips "C:\Users\suhaas\Downloads\Student folder\Student folder\train\00000001" --partition training --export-training
python -m unittest src.data.test_alignment src.data.test_pipeline -v
```

After reviewing the sample results, the SAME code can process every folder in
the verified 80-trip training directory (not run as part of this sample check):

```powershell
python -m src.data.align_trips --partition training --export-training (Get-ChildItem -LiteralPath "C:\Users\suhaas\Downloads\Student folder\Student folder\train" -Directory).FullName
```

This prepares labels, not models. Pratham owns running/comparing the five
models using the inferred-label CSVs, trip-grouped CV, recorded label provenance
and the complete observation ledgers. Do not fit on student-test/evaluator trips.

The caller must declare `training` or `student-test` and verify path permissions;
training CSV export is restricted to `training`. Generated files stay outside
raw data, under git-ignored `outputs/alignment/<trip_id>/`:

- `labels.jsonl`: one record per observation with `trip_id`, `observation_index`,
  `arc_id`, `route_position`, `status`, `reasons`, `distance_m`,
  `alternative_cost_gap`, `plausible_position_count`, `plausible_route_positions`
  (first 10; full count retained), `label_source`, `method_version`,
  `candidate_count`, `candidate_miss`, `training_eligible`.
- `provenance.json`: method/configuration, input and source-code SHA256 hashes,
  partition declaration, UTC creation time, best-path cost, candidate settings,
  and `verified_ground_truth: false`.
- `summary.json`: full status/coverage counts and reason counts.
- Optional `approximate_candidates.csv`: exact columns
  `trip_id,observation_index,arc_id,distance_m,label`, containing only confident
  observations whose inferred arc is present, with exactly one positive each.
  A trip without eligible labels produces a header-only CSV. Keep provenance
  beside the CSV; never call these independently verified labels.

Coverage is assessed AFTER alignment. `candidate_miss` is boolean only for
confident labels, otherwise null. Its denominator includes ALL confident labels,
including misses. A zero denominator means N/A, not a zero rate. Training
eligibility requires confidence AND candidate presence. The complete ledger
retains every excluded observation; the training CSV cannot measure overall accuracy.

### Actual three-trip results (2026-11-18)

| Training trip | Confident | Ambiguous | Unavailable | Misses / confident | Training eligible |
|---|---:|---:|---:|---:|---:|
| 00000001 | 652 | 663 | 0 | 0 / 652 | 652 |
| 00000002 | 0 | 0 | 7,243 | N/A (denominator 0) | 0 |
| 00000003 | 536 | 285 | 0 | 2 / 536 | 534 |
| Total | 1,188 | 948 | 7,243 | 2 / 1,188 | 1,186 |

Trip 00000002 has a reference break entering zero-based position 54: adjacent
arcs share no endpoint. Its 510 no-candidate observations remain recorded but
cannot be called reference candidate misses under this unavailable-label policy.
Trip 00000003's two misses have nonempty candidate sets, but the confidently
inferred reference arc is absent. Exact IDs/residuals remain in local ledgers.
These are proxy labeling/coverage counts, NOT model accuracy. No real model
training was performed. Only these three trips were processed, not all 80.

All 19 unit tests passed, including comparison with exhaustive monotone paths
in tiny synthetic cases, repeated occurrences, stays/skips, distant points,
gaps, missing references and candidate independence. CSV exports are separately
checked for one positive per eligible observation. Threshold sensitivity,
manual map inspection and additional permitted-trip validation remain necessary.

## Pratham handoff and evaluation denominators

The baseline needs only `distance_m`, so inference integration can proceed now.
Pratham's experiment CSV needs exactly one positive candidate per included
observation; it cannot represent unlabeled/ambiguous observations or missing
reference candidates. Do not manufacture positives to satisfy that format.

Before real training, agree on label provenance and a separate observation
ledger: total observations, labels available/ambiguous/unavailable, no candidates,
and aligned reference absent from candidates. Keep trips intact in CV and fit
learned preprocessing inside each training fold only.

State the labeled denominator for pointwise metrics, count no-candidate and
reference-miss observations as failures where applicable, and separately report
unavailable labels against the full observation count. Label conditional metrics
as conditional. Metrics against approximate labels are proxy results, not
organizer accuracy. Define targets/classes consistently before aggregate F1
and log loss, since arc IDs are trip-local.

## Questions retained for future clarification

- Are timestamped/per-observation reference arc IDs supplied separately?
- Is there an official alignment/scoring rule for endpoints, loops, ambiguous
  observations, gaps and reference discontinuities?
- What are the exact pointwise/sequence targets and denominators, and the
  intended class definition for F1/log loss across trip-local road IDs?
- How are reserved references supplied for separate final server-side scoring?

No message has been sent to organizers or teammates. This document is a handoff
for Suhaas to relay when possible. Work proceeds using only supplied files;
organizer-defined pointwise accuracy remains unresolved. Evaluator trips remain
untouched in development, with labels reserved for final server-side scoring.
