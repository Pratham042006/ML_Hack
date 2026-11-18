# data/ (local only, not committed)

Put the supplied dataset here on every laptop so the layout is:

    data/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}         (80 trips)
    data/student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}  (10 trips)

Verified facts about the supplied files:
- Whitespace/tab separated, no headers, Windows (CRLF) line endings. Parse robustly.
- .track = longitude, latitude, timestamp (s); .nodes = longitude, latitude;
  .arcs = from_node, to_node (row index = arc_id); .route = ordered arc IDs.
- IDs are local to each trip. Never join across trips.

Evaluator trips (final 10) are supplied later; keep them out of git and out of training.
