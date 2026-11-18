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

## Proposed experiment: approximate sequence alignment

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
inherits GPS/reference errors and is not verified ground truth. This method is
proposed, not implemented or evaluated by the current diagnostic.

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
