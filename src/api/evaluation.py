"""Server-side evaluation hook (kept separate from inference on purpose).

Returns {} = "not calculated". It must NEVER return reference labels or routes, only allowed
aggregate numbers. A defensible observation-level evaluation needs the organizers' answer on how
labels are supplied (see the team brief), so nothing is computed yet. Do not put placeholder
numbers here: an empty dict is the honest answer.
"""


def compute_metrics(trip_id: str, predictions: list) -> dict:
    return {}
