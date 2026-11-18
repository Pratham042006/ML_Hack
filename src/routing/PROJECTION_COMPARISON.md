# Projected transitions on development trips

Run from the repository root with Python 3.12 and NumPy:

```powershell
py -3.12 -m src.routing.projection_comparison "data/train/00000005"
py -3.12 -m src.routing.projection_comparison "data/train/00000005" --with-proxy
```

Use the first command for inference-only comparison. Use the second with a
permitted labeled training or student-test trip. Its reference is opened
after all three prediction sequences are fixed. It reports agreement with
approximate reference alignment. The experimental section-alignment policy
from the data branch was excluded from this comparison. The four reports
used local copies of the supplied development trips. No evaluator trips
were used.

## Position policy and compatibility

The experiment uses a sidecar keyed by `(observation_index, arc_id)` and the
same local metric plane as candidate distance. All 59,779 sidecar values on
these four trips agreed with Suhaas's `project_candidate_positions` at
`b5858a1` to within 1e-12. A future nullable top-level `projection_fraction`
would carry the same value, outside `features`, from directed start (0) to
directed end (1). A reverse arc is a separate state and has the complementary
fraction. A zero-length arc gives null. If either fraction is missing or null,
use the midpoint approximation for the entire transition. Other invalid
fractions raise an error.

- Between different arcs, add the previous arc's tail, the directed shortest
  path from its end to the next arc's start, and the next arc's prefix.
- On one arc, use forward distance. Following Suhaas's proposal, treat up to
  5 m of backwards displacement as stationary when elapsed time is positive
  and at most 10 seconds. Larger slips or slips over longer intervals
  require a legal directed loop from the arc end back to its start. The
  speed cutoff still applies.
- The existing 30-second gap limit, 60 m/s speed limit, emissions and
  missing-candidate behavior apply to both decoders. The jitter settings
  were taken from the data handoff before the final comparison, without
  selection on proxy scores.

Pratham, Suhaas and Prithika still need to confirm the shared field and
fallback. The experiment has a routing-local projection provider while
Suhaas's helper is on the data branch. Once it is available in the integrated
branch, use it in place of the local provider. The default `decode()`
and the candidate, scorer and API contracts continue with their existing
behavior.

## Comparison

Classifier-only, midpoint Viterbi and projected Viterbi shared the same
candidate list and normalized distance scores. All passed the prediction
contract. Counts are per observation and all timing columns are seconds
from one set of laptop runs. Projection prep is separate from decoder
time. Both exclude the audit and optional proxy calculation.

| Trip | Matched / unmatched, all methods | Candidate rows | Arc choices changed, mid → projected | Classifier / mid / projected decode | Projection prep |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00000001 | 1,315 / 0 | 8,697 | 670 | 1.035 / 1.171 / 1.063 | 0.182 |
| 00000002 | 6,733 / 510 | 46,306 | 3,752 | 0.430 / 1.645 / 1.612 | 0.775 |
| 00000005 | 1,045 / 0 | 3,073 | 224 | 0.040 / 0.086 / 0.100 | 0.070 |
| 00000052 | 262 / 0 | 1,703 | 107 | 0.051 / 0.098 / 0.145 | 0.042 |

The table below audits both output sequences using the **same projected
distance policy**. Reachable includes over-cap pairs. Disconnected means
the directed graph cannot link the two projected positions under this
policy. Pairs with unmatched endpoints or non-positive or over-30-second
intervals are separate. The CLI also emits a midpoint audit for each
sequence. The positional audit measures consistency with the experimental
model, not whether the selected road is correct.

| Trip | Reachable, mid → projected | Disconnected | Over speed cap | Failed-transition resets | Matched blocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00000001 | 1,299 → 1,314 | 15 → 0 | 340 → 34 | 155 → 34 | 156 → 35 |
| 00000002 | 6,685 → 6,685 | 0 → 0 | 461 → 63 | 13 → 63 | 61 → 111 |
| 00000005 | 1,038 → 1,044 | 6 → 0 | 95 → 0 | 89 → 0 | 90 → 1 |
| 00000052 | 261 → 261 | 0 → 0 | 26 → 0 | 22 → 0 | 23 → 1 |

Trip 00000002 also had 10 gap resets and 510 missing-candidate events in
each Viterbi run. None of the runs had a non-positive timestamp reset.

## Examples and offline proxy

On trip 00000005, at observations 1 → 2, the projected decoder takes the
directed turn 176 → 173. The positions are 16.45 m apart along the roads
(16.8 m GPS displacement). The midpoint estimate is 487.41 m, and
midpoint Viterbi entered arc 173 early. At observations 24 → 25 the
projected decoder moves 173 → 172, with 32.22 m of projected travel and
33.5 m GPS displacement, whereas the midpoint estimate is 536.91 m.
Midpoint Viterbi delayed the turn and reset at observation 26. Projected
Viterbi keeps the block through both junctions.

On trip 00000002 at observations 2188 → 2189, midpoint Viterbi stays
on arc 14405, with 3.08 m of projected progress. Projected Viterbi ends
its earlier block on arc 14406, almost 50 m from GPS, then resets onto
14405. The cross-break transition would require 149.39 m in one second.
This is a concrete local regression. More resets on that trip may
also expose impossible motion that the midpoint rule accepted, so the
reset count alone does not tell which road choices were right.

| Trip | Confident denominator | Ambiguous | Unavailable | Candidate misses among confident | Classifier / mid / projected agreements |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00000001 | 652 | 663 | 0 | 0 | 564 / 373 / 609 |
| 00000002 | 0 | 0 | 7,243 | — | — |
| 00000005 | 855 | 190 | 0 | 0 | 813 / 734 / 853 |
| 00000052 | 115 | 147 | 0 | 0 | 92 / 77 / 115 |

Trip 00000002's reference fails the directed connectivity check in the
current alignment method. The agreement rate is null for every method.
The three usable trips are preliminary proxy evidence, with no verified
accuracy claim. The trained-model comparison can reuse the agreed
position policy and frozen labels after the team selects an artifact.

## Map handoff for Prithika

For either Viterbi mode the report includes `runs.<mode>.diagnostics.events`
and `runs.<mode>.diagnostics.matched_blocks`. These use the same events and
boundaries as the [existing routing handoff](DEVELOPMENT_RESULTS.md).
Break before a reset observation and across an unmatched observation.
Render the geometries as matched segments per observation. Even within a
block, a transition may traverse intermediate arcs that are absent from
the output. The experiment adds no reference information to the
inference or map payload.
