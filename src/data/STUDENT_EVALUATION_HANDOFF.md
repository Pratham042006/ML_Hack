# Held-out student-test evaluation handoff

Scope: the ten permitted student-test trips. The diagnostic run uses
`monotone-reference-v1`, the existing 50 m / top-eight candidate policy and
the published 5 m / 10 s heading defaults. Settings were fixed before this
run. Use this partition for evaluation with prefitted models. Keep it outside
fitting, model selection, tuning and future threshold selection. Reserved
evaluator trips remain outside this work.

## Generate and reconcile

From the repository root, with Python 3 and NumPy, use a new output directory.

```sh
python -m src.data.student_evaluation PATH_TO_AUTHORIZED_STUDENT_TEST_ROOT --output-dir outputs/student-evaluation-v1
```

The output contains a batch manifest, ten per-trip ledgers with reason
and candidate-miss flags, per-trip provenance and summaries, and a separate
diagnostic report. Every raw input and emitted ledger is hashed, and the
manifest records the alignment version, configuration and source hashes.
The reference is opened separately for offline alignment and miss analysis.
Keep all generated files local until there is an approved transfer channel.

- Stable dataset fingerprint:
  `f267ab7a04611834e01dc65d0de9c6cd1eb161243347bf579b49ecda28bf3e5c`.
  This matches the local `review2-v1` student-test fingerprint and all coverage counts.
- This run's ledger manifest SHA-256:
  `e036006eec3a8ce5b6239311fcb146e722f8dee600d37b34ad18d6fc2fc07983`.
  This includes the source commit used to run generation. Prefer the stable
  dataset fingerprint, input hashes, settings and individual
  `labels.jsonl` hashes when reconciling machines.
- Pratham's independent ledger and comparison have not been compared
  directly. Check the fingerprint, counts, per-trip input and ledger hashes,
  candidate and alignment settings, then score denominators before
  exchanging records. Same counts alone do not establish row equality.

## Aggregate coverage and error diagnosis

| Measure | Count |
|---|---:|
| Trips processed | 10 / 10 |
| GPS observations | 19,160 |
| Confident inferred labels | 9,118 |
| Ambiguous inferred labels | 8,529 |
| Unavailable inferred labels | 1,513 |
| Observations without candidates | 0 / 19,160 |
| Confident inferred arcs absent from candidates | 16 / 9,118 (0.1755%) |
| Confident inferred labels present among candidates | 9,102 / 9,118 |
| GPS observations with no heading | 1,971 / 19,160 |
| Candidate rows with no heading | 13,640 / 113,650 |

All 16 misses are inside 50 m and rank below eight when all network arcs
are ordered by distance and arc ID. Their ranks range from 9 through 17.
These misses are attributable to the shortlist limit for the inferred arcs.
They do not show how many missed arcs are the actual road the vehicle took.

One trip accounts for all 27 distant unavailable labels. None of those GPS
points is within 50 m of any arc in the supplied reference route, while
every point has a network arc within 50 m. In the largest observed
reference separation, the closest reference arc is about 106 m away
and the closest network arc about 41 m away. The network has candidates
there. The data do not resolve whether the reference diverges from the GPS,
the GPS is off route, or the supplied reference omits some road travel.
A separate trip accounts for the other 1,486 unavailable labels. Its
directed reference breaks between arcs with no shared node and a gap of
about 15 m between the directed endpoints. The existing whole-trip
withholding policy applies. No section repairs were used here.

Among missing GPS headings, 1,934 are short movements, 27 are gaps above
10 s, and 10 are first observations. One gap example spans 210 s despite
only about 5 m of displacement, so it fails the time rule independently of
the distance rule. Among missing candidate values, 13,369 are short
movements, 214 are large gaps, and 57 are first observations. No settings
were changed from these observations. For a later training-data version,
use training-only experiments to compare shortlist sizes and review
reference sections against their GPS evidence.

## Inference parity and scoring

At integrated commit `2856389`, the API requests
`load_trip(..., include_reference=False)` and strips a reference field if
one unexpectedly appears. Only artifacts that require heading trigger
heading enrichment. Training blanks parse as NaN, and inference converts
explicit `None` to NaN for features declared nullable, then passes them into
the saved estimator's imputer. The UI response exposes the whitelisted
observation and prediction fields, and the server metrics hook returns
an empty object. No reference data appears in the inspected UI prediction
path.

The integrated inference helper matched **all 3,808 eligible candidate
rows** of one local training trip exactly, including 13 missing headings.
On a separate student-test trip, an API runner smoke check blocked access
to `.route` and passed with 1,703 candidates, 16 missing headings, and 262
predictions. It used a synthetic training fixture solely to exercise the
saved-artifact branch. This validates the current defaults and missing
handling, not a particular trained artifact or the browser UI. The
heading artifact format currently does not pin its threshold settings.
If the team approves a later feature version with different defaults,
store and enforce its config in the model artifact before deployment.

For an end-to-end proxy top-1 comparison, use the same **9,118** confident
inferred labels for each model and count the 16 misses as incorrect. If
reporting candidate-present top-1 separately, label its **9,102** denominator
as conditional. Carry the 8,529 ambiguous and 1,513 unavailable observations
alongside both results. These are agreement metrics against inferred
labels. Do not describe them as verified route accuracy.

Organizer questions for Pratham: Is there an official GPS-to-arc reference,
or an official method to align the ordered route to GPS timestamps? What
are the denominators and intended treatment of ambiguous/unavailable
observations, candidate misses, repeated arcs and disconnected references?
Which metrics should the server expose for held-out evaluation?
