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
