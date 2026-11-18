# Snapshot reconciliation and routing follow-up

## Compare the exact experiment CSV

Pratham's frozen experiment CSV has SHA-256
`65486db4e3d3118f200ffd4c84e11ecdffb6b84d5d5e32c366147fad7e57c1b9`.
Keep using that file while reconciling the exports. Suhaas's local
`review2-v1` CSV has SHA-256
`8a8c08865db51ddddd19588fe244aafad82ccf8ffd8dad31963782c6d5c6a233`.
Reassembling all 80 local per-trip files in trip-ID order with one header
reproduced the local CSV byte-for-byte. LF/CRLF, a UTF-8 BOM, final-newline
changes, and tested pandas float formats did not produce Pratham's hash.
The experiment CSV and its combiner are still on Pratham's laptop.

Run this from the repo with both approved local files available. Pass the
experiment file as the second argument to verify its SHA-256 before comparison.

```text
python -m src.data.compare_exports outputs/review2-v1/training/training_candidates.csv PATH_TO_TRAINING_CHECKED_CSV --expected-right-sha256 65486db4e3d3118f200ffd4c84e11ecdffb6b84d5d5e32c366147fad7e57c1b9
```

The tool joins on the exact trip ID and numeric observation and arc IDs,
rejects duplicate keys, and reports byte equality, key order, one-sided keys,
exact decimal distance differences and label differences. It includes counts
by affected trip. Identical parsed rows with different bytes indicate a CSV
serialization or ordering difference. If numeric values differ, report the
number and magnitude even when the change is small. Matching row counts alone
leave key and value equality unknown.

## Exploratory training diagnostics

Use a new output directory for these results.

```powershell
$dataRoot = "C:\Users\suhaas\Downloads\Student folder\Student folder"
python -m src.data.review_alignment (Join-Path $dataRoot train) outputs/review2-v1/training outputs/alignment-followup/full --all-sections --uncertainty-trip 00000054 --uncertainty-trip 00000066 --uncertainty-trip 00000081
```

- Eight of ten disconnected training routes had ordered, nonoverlapping GPS
  anchor windows for every continuous section. The experimental fits found
  10,041 confident inferred labels and 24 misses within those windows.
  For `00000033` the last section had zero distinct anchors. For `00000086`
  the first section had zero. Both were withheld from the section trial.
- On `00000054`, 17,167 points were over 50 m from their selected reference
  arc. Of these, 16,519 had no inference candidates and 320 had some other
  reference arc within 50 m. The first count points to a road-network or GPS
  coverage issue as well as an alignment issue. The second is a subset to
  inspect for route ordering, loops or the effect of the skip penalty.
- On `00000066`, 203 of 390 distant points were within 50 m of some reference
  arc. Increasing the cutoff to 65 m added 173 confident labels and 173
  misses against the same 50 m candidate generator. Keep the cutoff fixed
  for the frozen snapshot.

The section experiment treats every break as a boundary and only fits within
the observed anchor windows. Its counts are dependent on heuristic anchor
and label rules. The generated report is separate from both snapshots and
contains no training export.

## Proposal for Adithya

Would a nullable top-level `projection_fraction` work for the routing scorer?
The local data prototype can supply a clamped fraction from the directed
from-node (0) to the to-node (1), using the same projection as `distance_m`.
Zero-length arcs yield null. A missing or null value can retain the midpoint
fallback. A sidecar keyed by `(observation_index, arc_id)` is also possible
if you prefer to keep the candidate record unchanged.

For a transition across distinct arcs, the fraction lets you use
`(1 - previous_fraction) * previous_length` and
`current_fraction * current_length` for the two partial segments. For two
observations on the same directed arc, please confirm how you want to handle
negative progress from GPS jitter. On training trip `00000001`, the median
position difference from the midpoint was 21.2 m across 8,697 candidates.
The next check is to compare routing choices and resets on the same trips
under both approaches. Agree on the field, fallback and backwards-jitter
rule with Pratham and Prithika before any contract update.
