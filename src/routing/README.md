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
python -m unittest src.routing.test_decoder -v
python -m src.routing.synthetic_demo
```

The demo is explicitly synthetic and is not an accuracy or evaluation result.

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
