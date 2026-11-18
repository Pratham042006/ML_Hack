"""Prepare decoding-block boundaries for consumers of routing diagnostics."""


_RESETS = {
    "large_gap_reset",
    "nonpositive_timestamp_reset",
    "unreachable_transition_reset",
}


def matched_blocks(predictions, events):
    """Describe contiguous decoder blocks in prediction order.

    Match geometries remain individual road segments within each block.
    """
    resets = {
        event["observation_index"]: event["event"]
        for event in events if event["event"] in _RESETS
    }
    blocks = []
    block = None
    break_before = None
    for prediction in predictions:
        index = prediction["observation_index"]
        if prediction["arc_id"] is None:
            block = None
            break_before = "missing_candidates"
            continue
        if index in resets:
            block = None
            break_before = resets[index]
        if block is None:
            block = {
                "first_observation_index": index,
                "last_observation_index": index,
                "matched_count": 0,
                "break_before": break_before,
            }
            blocks.append(block)
        block["last_observation_index"] = index
        block["matched_count"] += 1
    return blocks
