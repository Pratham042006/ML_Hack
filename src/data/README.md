# src/data - owner: Suhaas

Implemented in `pipeline.py`, following `docs/INTEGRATION_CONTRACT.md` unchanged:
`load_trip(trip_dir, include_reference=False)` and `generate_candidates(trip)`.
Verified with Pratham's distance baseline at `8f0d557`; no trained models or
decoding are included. Only files in `src/data/` are changed.

## Run from repository root (PowerShell)

```powershell
python -m src.data.example "C:\Users\suhaas\Downloads\Student folder\Student folder\train\00000001"
python -m unittest src.data.test_pipeline -v
git fetch origin pratham/models
python -m src.data.integration_check "C:\Users\suhaas\Downloads\Student folder\Student folder\train\00000001" --model-ref 8f0d557
```

Dependencies: Python 3 with NumPy (already in root requirements). Replace the
example trip path on other machines; the raw dataset remains outside Git.
`--model-ref` reads a trusted, reviewed scorer from the local Git object database
into a temporary file and runs it without checking out/editing model files.
After model integration, omit it to use the working-tree scorer normally.

## Output contract

```python
{
    "trip_id": "00000001",
    "observations": [
        {"observation_index": 0, "longitude": 8.595203,
         "latitude": 49.844053, "timestamp": 1.0}
    ],
    "network": {"nodes": [[8.581679, 49.857880]], "arcs": [[0, 281]]}
}
```

The excerpt shows only the first record of each list, not a complete network.
Coordinates are `[longitude, latitude]`; timestamps are seconds. Node and arc
IDs remain zero-based row indices local to the trip. Arc endpoints retain
their input order; physical directionality still needs agreement with routing.

## Validation and limitations

- Requires `<trip_id>.track`, `.nodes`, and `.arcs` inside the trip directory.
- Rejects missing files, empty track/network files, malformed or blank rows,
  nonfinite/out-of-range coordinates, nonfinite/decreasing timestamps, and
  out-of-range arc node references. Errors identify the file and row where applicable.
- Equal timestamps and positive gaps are retained; the example reports duplicate
  timestamp steps and gaps over one second (a descriptive count, not a cleaning threshold).
- No raw inputs are modified, sorted, removed, imputed or renumbered.
- Default loading does not access `.route`. Explicit `include_reference=True`
  separately adds `reference_route`, validating each arc ID. An empty route
  remains an empty list, indicating no reference entries.
- Reference entries are ordered traversed arcs, not labels aligned to GPS rows.
- Full-dataset validation/EDA and label alignment remain future work. Coordinate
  range checks cannot detect every possible coordinate-order mistake.
- Tests use clearly labeled synthetic fixtures for validation only, not metrics.

## Actual validation (2026-11-18)

Recorded results: `validation_results.json`. Three real training trips only:

| Trip | Observations | Candidates | Observations without candidates |
|---|---:|---:|---:|
| 00000001 | 1,315 | 8,697 | 0 |
| 00000002 | 7,243 | 46,306 | 510 |
| 00000003 | 821 | 6,293 | 0 |

For all three, Pratham's baseline preserved candidate fields/order and normalized
scores per matched observation (maximum sum error `2.220446049250313e-16`).
Complete unmatched observation IDs are retained in the JSON. Brute-force
nearest-segment checks for trip 00000002 observations 716, 6691 and 7000 gave
51.36, 54.16 and 104.20 metres respectively, beyond the 50-metre radius. These
are no-candidate counts, not reference candidate-miss rates, which need aligned
labels. The radius was not enlarged to hide these observations.

Trip 00000001 contains 88,809 nodes and 165,790 arcs, with 25 timestamp gaps over
one second, maximum 16 seconds, and no duplicate timestamp steps. Initial example
candidate generation took about one second. Recorded total load/generation/scorer
check times were about 0.24-1.22 seconds per trip: single local runs, not guarantees.
All 11 unit tests passed. No student-test or evaluator trips were accessed.

## Candidate policy and feature specification

- Initial radius: **50 metres**, inclusive. Maximum **8 candidates per observation**.
  Constants are in `pipeline.py`; these are provisional, untuned integration
  settings, not organizer requirements.
- Only feature: `distance_m`, a finite, nonnegative Python float in metres,
  representing the shortest distance to the closed road segment. Missing
  distances are never imputed; invalid coordinates fail validation, and no
  qualifying segment produces no candidate.
- A local equirectangular projection uses WGS84 radii of curvature at the
  central latitude of the combined network/observation extent. This is an
  approximate local metric, not exact geodesic distance. No formal error bound
  has been established. Spans over 2 degrees in longitude/latitude and latitudes
  outside +/-75 degrees raise errors; antimeridian trips are unsupported.
- Bounding-box filtering considers the entire segment. Exact point-to-segment
  distance in the projected plane follows: project onto the segment and clamp
  to endpoints. It is not distance to a node or midpoint. Zero-length arcs use
  their endpoint. A suitable projected CRS remains a future improvement.
- Candidates follow observation order, then ascending distance and arc ID.
  Each `(observation_index, arc_id)` occurs once. Distinct reverse/duplicate
  geometries retain their distinct arc IDs and may fill the candidate cap.
- Observations without candidates remain in the full observation list; no
  fallback match is invented. Reference fields never affect candidates/features.
- Full-array bounding-box scans cost O(observations * arcs); a spatial index
  can improve larger-trip performance if profiling requires it.

Actual first candidate from trip 00000001:

```python
{"observation_index": 0, "arc_id": 64973,
 "features": {"distance_m": 0.7205448939894391},
 "geometry": [[8.595156, 49.843765], [8.595675, 49.846381]]}
```

## Team handoff and limitations

Pratham receives only the stable `distance_m` feature; no added feature names,
missing-value conventions or inferred labels are introduced. One trip per
scorer call: never join IDs across trips. Candidate coverage is not accuracy.

Adithya receives original `[from_node, to_node]` arcs and endpoint geometry.
Generation ignores travel direction for distance purposes; reverse arcs remain
separate. Physical directionality requires agreement: do not silently make all
roads bidirectional. No intermediate path is reconstructed here.

Prithika can pass unchanged observations/network through integration, including
unmatched observations. See `LABEL_ALIGNMENT.md` for the separate reference
investigation and evaluation-denominator rules. The separate approximate aligner
now exports qualified labels and an optional training CSV with provenance;
labels never enter inference. Details and counts are in `LABEL_ALIGNMENT.md`.

Remaining data work: validate remaining permitted training trips, inspect
connectivity/EDA, investigate candidate radius/coverage and projection error,
and validate/refine approximate reference alignment. No claim of full-dataset readiness.
