"""Clearly labeled synthetic integration fixture for the routing contract."""

from __future__ import annotations

import json

from src.routing.decoder import (
    decode_classifier_only,
    decode_with_diagnostics,
)


def main() -> None:
    network = {
        "nodes": [
            [0.0000, 0.0000],
            [0.0010, 0.0000],
            [0.0020, 0.0000],
            [0.0000, 0.0010],
            [0.0010, 0.0010],
            [0.0020, 0.0010],
        ],
        "arcs": [[0, 1], [1, 2], [3, 4], [4, 5]],
    }
    observations = [
        {
            "observation_index": 0,
            "longitude": 0.0005,
            "latitude": 0.0005,
            "timestamp": 1.0,
        },
        {
            "observation_index": 1,
            "longitude": 0.0015,
            "latitude": 0.0005,
            "timestamp": 2.0,
        },
        {
            "observation_index": 2,
            "longitude": 0.0020,
            "latitude": 0.0005,
            "timestamp": 3.0,
        },
    ]
    scored_candidates = [
        {"observation_index": 0, "arc_id": 0, "score": 0.60},
        {"observation_index": 0, "arc_id": 2, "score": 0.40},
        {"observation_index": 1, "arc_id": 1, "score": 0.30},
        {"observation_index": 1, "arc_id": 3, "score": 0.70},
        # Observation 2 deliberately has no candidates.
    ]

    classifier_only = decode_classifier_only(
        observations, scored_candidates, network
    )
    viterbi, diagnostics = decode_with_diagnostics(
        observations, scored_candidates, network, max_speed_mps=200.0
    )
    print(
        json.dumps(
            {
                "fixture": "SYNTHETIC - NOT AN EVALUATION RESULT",
                "classifier_only": classifier_only,
                "viterbi": viterbi,
                "diagnostics": diagnostics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
