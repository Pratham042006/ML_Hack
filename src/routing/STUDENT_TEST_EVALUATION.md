# Frozen student-test routing comparison

The runner uses the ten supplied `student_test` trips in ID order. It
records input hashes, scoring and routing settings, per-trip metrics and
pooled counts. The proposed projected settings were carried forward from
development: 5 m backward jitter for positive intervals up to 10 s, otherwise
directed travel. If either fraction is null or missing, the whole transition
uses the midpoint policy. Both decoders retain the 30 s gap and 60 m/s speed
limits. Candidate generation uses the shared 50 m radius and eight-candidate
limit. No parameters were changed after looking at the student-test results.

Run from the repository root with Python 3.12 and the repo dependencies.
Example commands for PowerShell:

```powershell
py -3.12 -m src.routing.heldout_evaluation "data/student_test" --baseline-only --with-proxy
py -3.12 -m src.routing.heldout_evaluation "data/student_test" --model-artifact $trustedHeadingArtifact --artifact-sha256 $artifactHash --training-sha256 $headingCsvHash --with-proxy
```

For the paired run, have Pratham supply the trusted heading artifact, its
SHA-256 and the matching heading-training CSV SHA-256 independently. The
runner verifies the artifact file before loading it, then checks its training
hash and required heading feature metadata. Save the JSON outside Git. Omit
`--with-proxy` for inference-only runs. With it enabled, each trip's
predictions are fixed before loading its reference. The reserved evaluator
split is outside this runner's scope.

The heading artifact and its approved hashes were unavailable locally.
The table below contains the **real distance baseline runs**. A synthetic
nullable-heading artifact exercised the paired batch path in tests.

## Pooled baseline results

There were 19,160 observations and 113,650 candidates. All three methods
matched 19,160 observations with zero unmatched and zero observations
without candidates. Of the 9,118 approximate confident labels, 16 were
absent from the candidate list and count as misses. There were also
8,529 ambiguous labels and 1,513 unavailable labels.

| Distance scorer decoder | Proxy agreement, including misses | Among candidate-present labels | Disconnected / over cap, projected audit | Failed / gap resets | Matched blocks | Sum of decode seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classifier-only | 4,936/9,118 (54.13%) | 4,936/9,102 | 42 / 6,668 | — | — | 1.566 |
| Midpoint Viterbi | 4,430/9,118 (48.59%) | 4,430/9,102 | 3 / 3,902 | 751 / 2 | 763 | 3.378 |
| Projected Viterbi | 7,797/9,118 (85.51%) | 7,797/9,102 | 2 / 122 | 124 / 2 | 136 | 4.263 |

**Reconciliation pending:** The independent results in `src/models/STUDENT_TEST_RESULTS.md`
report 4,972 classifier-only and 4,584 midpoint agreements, while this run
reports 4,936 and 4,430. Both report 7,797 projected agreements. A local
rerun of trips 00000007 and 00000040 reproduced the independent counts, not
the counts in this table. The cause has not been established; do not combine
these runs or attribute the difference to platform or tie handling without
further evidence. Trip 00000052 was inspected during earlier development,
so this student-test partition is not fully untouched.

The common projected audit classifies adjacent assigned arcs under the same
directed distance and speed rules for all three outputs. JSON also records a
common midpoint audit and per-trip timings. Decode timing sums the ten
decoder calls and excludes shared input loading, candidate generation,
scoring, projection preparation, diagnostics audits and proxy evaluation.
Input loading, candidate generation and baseline scoring together totaled
3.973 s. Preparing projected positions totaled 0.874 s. These are observed
timings on one laptop. Rates pool the raw
numerators and denominators. The labels are inferred and may favor aspects of
the matching rules. Verified route accuracy requires stronger ground truth.

| Student-test trip | Matched / unmatched, all methods | Confident / ambiguous / unavailable | Candidate misses | Classifier / midpoint / projected agreements | Failed resets, mid → proj | Blocks, mid → proj |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00000000 | 2,503 / 0 | 1,970 / 533 / 0 | 2 | 1,063 / 1,077 / 1,921 | 118 → 6 | 119 → 7 |
| 00000007 | 1,354 / 0 | 1,036 / 318 / 0 | 0 | 453 / 543 / 934 | 72 → 38 | 73 → 39 |
| 00000009 | 1,049 / 0 | 171 / 878 / 0 | 12 | 80 / 82 / 122 | 26 → 4 | 27 → 5 |
| 00000040 | 2,995 / 0 | 1,969 / 1,026 / 0 | 0 | 1,100 / 762 / 1,883 | 135 → 25 | 136 → 26 |
| 00000052 | 262 / 0 | 115 / 147 / 0 | 0 | 92 / 77 / 115 | 22 → 0 | 23 → 1 |
| 00000056 | 3,958 / 0 | 1,323 / 2,608 / 27 | 0 | 985 / 736 / 1,229 | 202 → 11 | 203 → 12 |
| 00000065 | 768 / 0 | 234 / 534 / 0 | 1 | 187 / 143 / 225 | 29 → 4 | 31 → 6 |
| 00000076 | 1,486 / 0 | 0 / 0 / 1,486 | — | — | 29 → 1 | 31 → 3 |
| 00000077 | 3,911 / 0 | 1,985 / 1,926 / 0 | 1 | 801 / 794 / 1,120 | 83 → 31 | 84 → 32 |
| 00000091 | 874 / 0 | 315 / 559 / 0 | 0 | 175 / 216 / 248 | 35 → 4 | 36 → 5 |

Trip `00000076` has no proxy rate because its reference failed the
directed connectivity check. No-candidate counts are distinct from candidate
misses. Every observation in this split had candidates, while 16 confident
proxy arcs did not appear in their observation's candidate list.

## Example checks

- **Student-test 00000007, observations 22 → 23.** GPS moved 19.10 m in 1 s.
  Midpoint Viterbi chose arcs 4 → 5, both outgoing from node 2 on different
  branches. Arc 4 was 45.32 m from GPS at observation 22 and the projected
  audit estimates 137.41 m for this pair. Projected Viterbi stayed on arc 5
  (2 → 503). Its projected progress was 19.06 m and both points were within
  3.31 m of that arc.
- **Student-test 00000007, observations 472 → 473.** GPS moved 7.68 m in 1 s.
  Midpoint Viterbi stayed on arc 540 (260 → 262), with 7.65 m of projected
  forward progress. Projected Viterbi selected reverse arc 545 (262 → 260)
  on both sides of a reset at observation 473. The two arcs share geometry,
  so GPS proximity cannot distinguish them. Motion is 7.65 m backward on
  545. The directed loop would require 944.29 m in one second. This is a
  regression in local direction and continuity despite lower trip-level resets.
- **Training 00000002, observations 2188 → 2189.** The earlier development
  comparison found 13 → 63 failed resets across the trip. At this pair
  projected Viterbi ended a block on arc 14406, about 50 m from GPS,
  and restarted on 14405. The cross-break projection was 149.39 m in
  one second. See [the development projection comparison](PROJECTION_COMPARISON.md).

## Map handoff

Consume `decode_with_diagnostics` and `matched_blocks` for the API/UI wiring.
The preview renders each selected road geometry independently with its block
ID, a marker at each reset, and a marker for each unmatched observation.
It reads only the trip's track and network. React integration and the
two-laptop demo remain with Prithika pending the shared response field.

```powershell
py -3.12 -m src.routing.map_preview "data/train/00000002" --out "outputs/routing-preview-00000002.html"
```

The output directory must exist. Folium is listed in the repo requirements.
The renderer test asserts that the HTML has an individual polyline for each
matched observation and no connectors across resets or unmatched points.
Output remains matched segments per observation.

Verification passed 23 data tests, 7 model tests and 38 routing tests. The
development preview for trip 00000002 emitted 6,733 independent segment
polylines, 510 unmatched markers, 10 gap markers and 13 failed-transition
markers.
