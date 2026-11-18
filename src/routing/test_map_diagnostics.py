import unittest

from src.routing.map_diagnostics import matched_blocks


class MapDiagnosticsTests(unittest.TestCase):
    def test_reset_and_unmatched_split_blocks_but_score_notice_does_not(self):
        predictions = [
            {"observation_index": i, "arc_id": arc}
            for i, arc in [(10, None), (12, 0), (15, 1), (17, 2), (20, None), (25, 3)]
        ]
        events = [
            {"observation_index": 15, "event": "all_zero_scores_uniform"},
            {"observation_index": 17, "event": "unreachable_transition_reset"},
        ]

        self.assertEqual(matched_blocks(predictions, events), [
            {"first_observation_index": 12, "last_observation_index": 15,
             "matched_count": 2, "break_before": "missing_candidates"},
            {"first_observation_index": 17, "last_observation_index": 17,
             "matched_count": 1, "break_before": "unreachable_transition_reset"},
            {"first_observation_index": 25, "last_observation_index": 25,
             "matched_count": 1, "break_before": "missing_candidates"},
        ])

    def test_both_timestamp_reset_events_start_a_new_block(self):
        predictions = [{"observation_index": i, "arc_id": 1} for i in range(3)]
        events = [
            {"observation_index": 1, "event": "large_gap_reset"},
            {"observation_index": 2, "event": "nonpositive_timestamp_reset"},
        ]
        self.assertEqual(
            [b["break_before"] for b in matched_blocks(predictions, events)],
            [None, "large_gap_reset", "nonpositive_timestamp_reset"],
        )
        self.assertEqual(matched_blocks([], []), [])
