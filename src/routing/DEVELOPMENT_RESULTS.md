# Routing development results

Run from the repository root with Python 3.12 and NumPy. The decoder itself uses
the standard library. The scikit-learn dependency is needed for the combined
model tests.

```powershell
python -m src.routing.integration_demo "data/train/00000001"
python -m src.routing.proxy_evaluation "data/train/00000001"
```

The first command reads track and network files and runs the distance scorer.
The second is for labeled development trips offline. It fixes both sets of
predictions before it loads the reference. Replace the argument with your local
trip directory. The runs below used training trips 00000001, 00000002 and
00000005 and student-test trip 00000052.

## Inference comparison

Both methods used identical candidates and distance scores. All four trips
passed count, order, candidate membership and geometry checks for both methods.
The test suite also covers disconnected directed arcs, timestamp resets and
unmatched points.

| Trip | Observations | Candidates | Matched / unmatched, both | Changed choices | Classifier / Viterbi seconds | Pipeline seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00000001 | 1,315 | 8,697 | 1,315 / 0 | 740 | 0.886 / 1.040 | 4.149 |
| 00000002 | 7,243 | 46,306 | 6,733 / 510 | 3,982 | 0.261 / 1.236 | 2.796 |
| 00000005 | 1,045 | 3,073 | 1,045 / 0 | 213 | 0.021 / 0.045 | 0.172 |
| 00000052 | 262 | 1,703 | 262 / 0 | 141 | 0.022 / 0.051 | 0.123 |

These are single laptop runs. Pipeline timing includes loading, generation,
scoring and both decoders, measured before the audit and JSON serialization.

| Trip | Directed disconnected pairs, C → V | Reachable but over speed cap, C → V | Viterbi gap resets | Viterbi failed-transition resets |
| --- | ---: | ---: | ---: | ---: |
| 00000001 | 8 → 3 | 496 → 152 | 0 | 155 |
| 00000002 | 0 → 0 | 622 → 13 | 10 | 13 |
| 00000005 | 13 → 0 | 117 → 89 | 0 | 89 |
| 00000052 | 0 → 0 | 76 → 22 | 0 | 22 |

The audit checks consecutive observations with two assigned arcs and a time
delta in (0, 30] seconds. A pair is disconnected if no directed path exists
between arc midpoints. A reachable pair is over the cap if its estimated travel
speed exceeds 60 m/s. Unmatched endpoints and excluded timestamp intervals have
separate counts. Failed-transition resets include candidates disconnected by
the graph or rejected by the speed cap.

Two cases on training trip 00000005 illustrate why a better continuity count
can conceal poor choices. These are local checks on GPS and road geometry
without reference labels. The projected distances below are diagnostic
calculations and were never used for decoding.

- **Observation 1.** The baseline picked arc 176 (7.14 m from GPS), then
  moved onto arc 173 at observation 2. These arcs meet at node 140 and point
  southwest, consistent with the GPS heading of 240°. Their midpoint
  distance is 487.41 m, but the GPS projects near the end of 176 and the start
  of 173, about 16.4 m apart along the roads. The GPS itself moved 16.8 m.
  Viterbi picked arc 173 early (10.89 m from GPS), avoiding the penalty by
  staying on that arc. The baseline transition looks physically plausible.
- **Observation 25.** Arc 173 ends at node 139, where arc 172 starts.
  The GPS moved 33.5 m with a heading of 239°, close to the arcs' 223°
  bearing. The baseline's move from 173 to 172 covers about 32.2 m using
  the projected positions, versus 536.91 m with midpoints. Viterbi stayed
  on 173 at observation 25, where it was farther from GPS (27.47 m vs
  11.10 m), then reset at 26. The decoder delayed the transition by one
  observation and still created a block break.

## Offline proxy agreement

Exact trip-local arc agreement against Suhaas's monotone reference alignment
(v1). The reported denominator is the number of labels marked confident by
that heuristic. Candidate misses are counted separately. The CLI also reports
the agreement denominator restricted to confident labels present in the
candidate set. When there are zero labels, the rate is null.

| Trip | Confident | Ambiguous | Unavailable | Candidate misses | Classifier agreement | Viterbi agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00000001 | 652 | 663 | 0 | 0 | 564/652 (86.5%) | 373/652 (57.2%) |
| 00000002 | 0 | 0 | 7,243 | — | — | — |
| 00000005 | 855 | 190 | 0 | 0 | 813/855 (95.1%) | 734/855 (85.8%) |
| 00000052 | 115 | 147 | 0 | 0 | 92/115 (80.0%) | 77/115 (67.0%) |

Trip 00000002's reference failed the directed connectivity check. Its 510
observations without candidates still count against inference coverage. The
tested Viterbi setup reduced implausible jumps on all four trips but had
lower proxy agreement on all three trips with usable labels. These proxy
results need team review before use in the report or model selection.

## Method and map handoff

Each hidden state is a candidate directed arc for the current observation.
Log emissions come from locally normalized scorer weights. The transition
penalizes disagreement between estimated road travel and GPS displacement.
Viterbi adds each current state's log emission to the maximum of the previous
state score plus the transition score, then follows backpointers within each
block. The shortest-path cutoff uses the existing speed cap. Exact prediction
and diagnostic equality against unbounded search was checked on trips 00000002,
00000005 and 00000052.

Prithika can call `decode_with_diagnostics(observations, scored_candidates,
network)` to obtain `(predictions, events)` in one pass. The demo emits these
events under `diagnostics.events`. Each event carries `observation_index`
and `event`; timestamp resets also carry `delta_seconds`.
`src.routing.map_diagnostics.matched_blocks(predictions, events)` returns
inclusive first and last observation indices, a matched count and a
`break_before` reason per block. The demo exposes it as
`diagnostics.matched_blocks`. Use prediction order for the endpoints because
observation IDs can be nonconsecutive. Proposed rendering after the API schema
is agreed:

- Show a break before the affected observation for `large_gap_reset`,
  `nonpositive_timestamp_reset` and `unreachable_transition_reset`.
- Show a missing match for `missing_candidates` and break the line on both
  sides of the unmatched observation.
- Treat `all_zero_scores_uniform` as a score-quality notice.
- Draw the returned road-segment geometries individually. The decoder returns
  per-observation segments. An expanded road path would also need the
  intermediate arcs between matches. Keep references in offline evaluation.

The demo reported 156, 61, 90 and 23 matched blocks on trips 00000001,
00000002, 00000005 and 00000052, respectively. Each trip's block counts sum
to its matched observation count.

Suhaas's proposal on `suhaas/data` at `d20242a` puts
`projection_fraction` on the candidate beside `features`. It is a clamped
orthogonal projection in the candidate generator's metric plane, measured
from the directed start node (0) to the directed end node (1). A reversed
nondegenerate arc uses 1 − f. A zero-length arc carries null.

Routing proposes using the midpoint rule for the whole transition if either
candidate omits the field or carries null. Reject other invalid values, and
confirm that the scorer passes this metadata through and keeps the snapshot's
feature list at `distance_m`. For
different arcs, use the remaining fraction of the previous arc, the directed
node-to-node path, and the prefix fraction of the next arc. For the same arc,
forward movement uses the fraction difference. Backward movement requires a
legal return path or an explicit jitter policy. Ask Suhaas, Pratham and
Prithika to confirm the proposal and the backward-jitter policy before
changing the contract. Compare on fixed candidates with forward/reverse,
same-arc and skipped-arc tests. This proposal is pending team agreement.
