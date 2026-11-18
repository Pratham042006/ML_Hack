# Heading agreement experiment

`heading_agreement` is the cosine of the angle between the incoming GPS step
and the candidate arc's directed from-node to to-node vector. Use the previous
observation in original trip order, including observations without candidates.
Both vectors are in the same local metric plane as `distance_m`. Their
normalized dot product works across 0°/360° and changes sign on a reversed
arc. The unitless range is [-1, 1], with 1 aligned, 0 perpendicular, and
−1 opposite. Curved roads are represented by their supplied endpoint chord.
A step that crosses a junction may reflect the previous road or the turn.

## Provisional thresholds

Use at least **5 m** of displacement and a strictly positive time gap of at
most **10 s**. On the 80 training trips, among steps with a valid gap up to
10 s, thresholds of 3 m, 5 m, and 10 m would withhold 12,307, 31,640, and
89,989 steps respectively. Gaps over 5 s, 10 s, and 20 s occurred 1,216, 590,
and 204 times respectively. Five metres removes more short, potentially noisy
steps than 3 m while retaining substantially more observations than 10 m.
Ten seconds filters long gaps while retaining more usable steps than 5 s.
These settings are provisional and based on training-data coverage. They have
not been validated as estimates of GPS accuracy or selected by model scores.

## Schema and missing values

```text
trip_id,observation_index,arc_id,distance_m,label,heading_agreement
```

The first observation, stationary/short steps, nonpositive or invalid
timestamp gaps, gaps over 10 s, and zero-length arcs have missing heading.
Store missing as a blank CSV field and return Python `None` in the prototype.
Proposed model preprocessing is to parse blanks/None as NaN and use the same
saved training-fold imputer at inference for estimators that need it. A
heading-missing indicator can be a separately agreed preprocessing option.
Do not use a zero sentinel because zero is a valid perpendicular measurement.
Train and compare on identical observations and trip folds, retaining rows
with missing heading. The shared candidate and scorer interfaces are unchanged.

## Run on Pratham's exact frozen snapshot

Use the exact frozen CSV and its matching manifest. The exporter verifies
the CSV hash before reading rows, keeps original row order and fields, and
adds only the new column. It reads `.track`, `.nodes`, and `.arcs`. Existing
labels are copied from the CSV and never enter the heading computation.

```sh
python -m src.data.export_heading PATH_TO_TRAIN_ROOT outputs/model_work/training_checked.csv outputs/model_work/checked_manifest.json --output-dir outputs/heading-v1-frozen
```

Local reproducible check on Suhaas's `review2-v1` snapshot (PowerShell):

```powershell
$train = "C:\Users\suhaas\Downloads\Student folder\Student folder\train"
python -m src.data.export_heading $train outputs/review2-v1/training/training_candidates.csv outputs/review2-v1/training/manifest.json --output-dir outputs/heading-v1-local
python -m unittest discover -s src/data
```

Use a fresh output directory for each run. The versioned output contains
`training_candidates_heading_v1.csv` and `manifest.json`. The manifest stores
the export/parent CSV hashes, source manifest hash, alignment method version,
feature code hashes, feature settings, training counts and per-trip input hashes.
The `review2-v1` manifest has original inference file hashes, which were verified
for every trip. If a source manifest lacks raw input hashes, the exporter can
record the current hashes and verify the parent CSV, but matching those raw
files to the original requires the team's input provenance. Keep the existing
observation, ambiguity and candidate-miss ledgers with the source snapshot.
Generated files remain local and Git-ignored.

## Actual local results

| Scope | Total | Missing heading |
|---|---:|---:|
| All training GPS observations | 208,417 | 32,310 |
| Eligible labeled GPS observations | 68,881 | 9,100 (all candidate headings absent) |
| Exported candidate rows | 307,899 | 44,676 |

No eligibility or label change. Among missing candidate values, 44,381 were
short steps and 295 were first observations. At training trip `00000001`,
observation 1 moved 43.65 m in 1 s. Arcs 64973 and 117639 scored +0.99974
and −0.99977 respectively. At observation 1177, a 4.91 m step had missing
heading. The first observation also had a blank value. Geometry tests cover
wrap-around, reverse and perpendicular arcs, stationary steps, timestamp
gaps, zero-length roads and preserving blank rows through export.

- Local source CSV SHA-256: `8a8c08865db51ddddd19588fe244aafad82ccf8ffd8dad31963782c6d5c6a233`
- Local heading export SHA-256: `e93edbbdcb32437e9f964c0960e6baf39eea7fbb8c406095d4d6765dbd718b3e`
- Pratham's frozen source CSV SHA-256: `65486db4e3d3118f200ffd4c84e11ecdffb6b84d5d5e32c366147fad7e57c1b9`

The local result is a feature/export verification. Run the first command
on Pratham's exact frozen source before the controlled model comparison.
Alignment labels remain inferred, and experimental section fits are excluded.
Routing projection and jitter settings are a separate experiment.
