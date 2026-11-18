# Routing - owner: Adithya

First-review slice for the PathSnap directed-road decoder. This component consumes
ordered observations, model-scored candidate arcs and the trip-local network from
`docs/INTEGRATION_CONTRACT.md`.

## What works

- `DirectedRoadGraph` validates the network and preserves directed, trip-local arc IDs.
- `decode_classifier_only(...)` is the independent highest-score baseline.
- `decode(...)` is a log-space Viterbi prototype and returns one contract-shaped
  prediction per observation.
- `decode_with_diagnostics(...)` returns the same predictions plus explicit events
  for missing candidates, timestamp/gap resets, all-zero score fallback and failed
  transitions.
- The implementation uses only the Python standard library.

The output is a sequence of **matched segments per observation**, not an expanded
path containing every intermediate road segment.

## Run the synthetic checks

From the repository root:

```text
python -m unittest discover -s src/routing -v
python -m unittest src.routing.test_decoder -v
python -m unittest src.routing.test_integration_demo -v
python -m src.routing.synthetic_demo
```

The synthetic demo is an illustrative unit fixture and is not an accuracy or evaluation result.

## Real-trip integration demo

From the repository root, run the real-trip integration demo on any trip directory:

```text
python -m src.routing.integration_demo "PATH_TO_TRIP"
```

Example (for a trip directory containing `<trip_id>.track`, `<trip_id>.nodes`, and `<trip_id>.arcs`):

```text
python -m src.routing.integration_demo "data/train/00000070"
```

The output is formatted, machine-readable JSON containing validation results, coverage, candidate counts, timings, classifier vs. Viterbi differences, and diagnostics.

### End-to-end pipeline execution

Following `docs/INTEGRATION_CONTRACT.md`, the integration demo executes:
1. `load_trip(path, include_reference=False)`
2. `generate_candidates(trip)`
3. `score_candidates(candidates)`
4. `decode_with_diagnostics(trip["observations"], scored, trip["network"])`

It also runs `decode_classifier_only(...)` to compare the independent choices with Viterbi.

### Validations verified

- **Exactly one prediction per observation**: Output length matches input observations.
- **Input order preserved**: Prediction sequence order matches input observation order.
- **Candidate set membership**: Every non-null selected `arc_id` belongs to that observation's candidate set.
- **Unmatched representation**: Unmatched outputs strictly have `arc_id: None` and `geometry: []`.
- **Matched geometry**: Matched outputs have valid road coordinate geometry `[[lon1, lat1], [lon2, lat2]]`.

### Reported metrics and diagnostics

- **Coverage (not accuracy)**: Matched and unmatched observation counts and coverage rate.
- **Candidate counts**: Total candidates, scored candidates, observations with/without candidates, and per-observation statistics.
- **Timings**: Elapsed seconds for trip loading, candidate generation, candidate scoring, and decoding.
- **Classifier-only vs. Viterbi**: Agreement rate, difference count, and breakdown of observations where independent argmax and Viterbi sequence decoding differ.
- **Diagnostics**: Aggregate event counts for missing candidates, large gap resets, unreachable transition resets, non-positive timestamps, and uniform emission fallbacks.

### Limitations and explicit disclaimers

- **Distance baseline only**: The candidate scorer is an untuned distance baseline (`exp(-distance_m / 20.0)`), not a trained model.
- **Reference labels not loaded**: Reference `.route` files are never loaded or accessed (`include_reference=False`).
- **No accuracy claims**: Matched/unmatched counts represent spatial pipeline coverage, **not** ground-truth route accuracy.

## Score and failure policy

- Finite scores in `[0, 1]` are locally normalized for Viterbi emissions.
- Individual zero scores receive a small log-space floor. If all scores for an
  observation are zero, uniform emissions are used and a diagnostic is emitted.
- Non-numeric, non-finite or out-of-range scores raise `ValueError` because they
  violate the model-to-decoder interface.
- No candidates produces the contract's unmatched record and ends the current
  Viterbi block.
- A non-increasing timestamp or a gap above 30 seconds ends the block and starts a
  new one. The reset is reported by `decode_with_diagnostics`.
- If every transition to the next candidate set is unreachable or faster than the
  default 60 m/s cap, the previous block is finalized and decoding restarts at the
  current observation. The reset is explicitly reported; the two blocks must not
  be drawn as a continuous route.

## Initial transition-distance approximation

The candidate contract currently identifies a segment but not the observation's
projected position along it. The prototype therefore uses a documented midpoint
approximation:

1. Remaining half of the previous directed arc.
2. Directed shortest-path distance from its end node to the next arc's start node.
3. First half of the next directed arc.

Staying on the same directed arc has distance zero. Reverse travel is not silently
allowed: it requires a separate reverse directed arc and a reachable directed turn.
The transition log score penalizes disagreement between this road distance and the
straight-line GPS displacement, and rejects implausible speed. This approximation
should be replaced with projected fractions when Person 1 exposes them in the
agreed contract.
