"""Offline comparison against approximate labels for permitted development trips."""

import argparse
from collections import Counter
import json
from pathlib import Path

from src.data.alignment import align_reference, assess_candidate_coverage
from src.data.pipeline import generate_candidates, load_trip
from src.models.inference import score_candidates
from src.routing.decoder import decode, decode_classifier_only
from src.routing.integration_demo import validate_pipeline_outputs


def summarize_proxy(ledger, predictions):
    """Measure exact arc agreement on confident labels, with and without misses."""
    by_index = {p["observation_index"]: p["arc_id"] for p in predictions}
    confident = [label for label in ledger["labels"] if label["status"] == "confident"]
    available = [label for label in confident if label["candidate_miss"] is False]

    def agreement(labels):
        hits = sum(by_index[label["observation_index"]] == label["arc_id"] for label in labels)
        return {"agreements": hits, "denominator": len(labels),
                "rate": hits / len(labels) if labels else None}

    return {
        "all_confident": agreement(confident),
        "confident_with_label_in_candidates": agreement(available),
    }


def evaluate_development_trip(path):
    trip = load_trip(path, include_reference=False)
    candidates = generate_candidates(trip)
    scored = score_candidates(candidates)
    observations, network = trip["observations"], trip["network"]
    runs = {
        "classifier_only": decode_classifier_only(observations, scored, network),
        "viterbi": decode(observations, scored, network),
    }
    for name, predictions in runs.items():
        if not validate_pipeline_outputs(observations, candidates, predictions)["all_checks_passed"]:
            raise ValueError(f"{name} predictions failed the routing contract")

    # Reference data is loaded only after both prediction sets are fixed.
    reference_trip = load_trip(path, include_reference=True)
    ledger = assess_candidate_coverage(align_reference(reference_trip), candidates)
    exclusion_reasons = Counter(
        reason for label in ledger["labels"] if label["status"] != "confident"
        for reason in label["reasons"]
    )
    return {
        "trip_id": trip["trip_id"],
        "metric": "exact trip-local arc agreement with approximate reference alignment",
        "provenance": ledger["provenance"],
        "label_counts": ledger["counts"],
        "exclusion_reason_counts": dict(exclusion_reasons),
        "proxy_agreement": {
            name: summarize_proxy(ledger, predictions) for name, predictions in runs.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip", type=Path, help="A labeled train or student_test trip, offline only")
    args = parser.parse_args()
    if args.trip.resolve().parent.name not in {"train", "student_test"}:
        parser.error("expected a trip under the train or student_test split")
    print(json.dumps(evaluate_development_trip(args.trip), indent=2))


if __name__ == "__main__":
    main()
