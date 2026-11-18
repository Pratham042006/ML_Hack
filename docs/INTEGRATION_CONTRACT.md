# Integration contract v1 (do not change without telling all four people)

Run everything from the repository root. Coordinates are [longitude, latitude]. Timestamps in seconds.
Node and arc IDs are zero-based list indices, local to one trip.

## Data (Suhaas) - src/data/pipeline.py
    load_trip(trip_dir, include_reference=False) -> dict
    generate_candidates(trip) -> list[dict]

trip = {"trip_id": "00000000",
        "observations": [{"observation_index": 0, "longitude": -1.0, "latitude": 48.0, "timestamp": 0.0}],
        "network": {"nodes": [[-1.0, 48.0], [-0.999, 48.0]], "arcs": [[0, 1]]}}
Add "reference_route" only if include_reference=True. The API never sets it.

candidate = {"observation_index": 0, "arc_id": 0, "features": {"distance_m": 5.2},
             "geometry": [[-1.0, 48.0], [-0.999, 48.0]]}
distance_m is mandatory, finite, >= 0. No qualifying arc -> no candidate for that observation.

## Model (Pratham) - src/models/inference.py
    score_candidates(candidates, model_artifact=None) -> list[dict]
Return copies with a "score" in [0, 1]; scores sum to ~1 per observation. None -> documented distance baseline.

## Routing (Adithya) - src/routing/decoder.py
    decode(observations, scored_candidates, network) -> list[dict]
One record per observation, same order:
    {"observation_index": 0, "arc_id": 0, "geometry": [[...]]}
Unmatched: {"observation_index": 0, "arc_id": None, "geometry": []}

## API (Prithika) - src/api/
    GET  /trips   -> {"trip_ids": ["00000000"]}
    POST /predict {"trip_id": "00000000"} ->
      {"trip_id": "...", "observations": [], "predictions": [], "metrics": {}, "mode": "baseline", "warnings": []}
mode is honestly one of mock / baseline / the real integrated mode. Empty metrics = not calculated.
Never return .route or reference labels to the UI.

## Integration check
    trip = load_trip("data/student_test/00000052")
    cands = generate_candidates(trip)
    scored = score_candidates(cands)
    preds = decode(trip["observations"], scored, trip["network"])
    assert len(preds) == len(trip["observations"])

## Integrated optional features and diagnostics (development)

The original candidate and prediction fields remain valid. When a trusted model artifact declares additional feature names, the API adds the requested context from the current trip before scoring. `heading_agreement` uses the artifact's `heading_config`, or heading-v1 defaults with an explicit warning for legacy artifacts. The optional context generator also supports `distance_margin_m`, `projection_fraction`, `candidate_rank` and `segment_length_m`. `heading_agreement` and `projection_fraction` may be null only when listed in the artifact's `nullable_features`; the estimator must contain its training-fitted imputer. Features are computed without reference routes. The distance baseline continues to require only `distance_m`.

Prediction responses additionally provide `decoder` (`midpoint`, `projected`, or `mock`) and `diagnostics` with `events` and `matched_blocks`. Blocks describe per-observation matches; they are not expanded paths. A reset starts a new block, and an unmatched observation separates blocks. The UI must not infer missing intermediate road geometry. `PATHSNAP_DECODER` selects the decoder, while `PATHSNAP_MODEL_ARTIFACT` selects scoring mode.
