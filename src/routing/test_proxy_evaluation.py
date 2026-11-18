import unittest

from src.routing.proxy_evaluation import summarize_proxy


class ProxyAgreementTests(unittest.TestCase):
    def test_ambiguous_and_unavailable_labels_are_excluded(self):
        ledger = {"labels": [
            {"observation_index": 0, "status": "confident", "arc_id": 4, "candidate_miss": False},
            {"observation_index": 1, "status": "confident", "arc_id": 5, "candidate_miss": True},
            {"observation_index": 2, "status": "ambiguous", "arc_id": 6, "candidate_miss": None},
            {"observation_index": 3, "status": "unavailable", "arc_id": None, "candidate_miss": None},
        ]}
        predictions = [
            {"observation_index": i, "arc_id": arc}
            for i, arc in enumerate([4, None, 6, None])
        ]
        result = summarize_proxy(ledger, predictions)
        self.assertEqual(result["all_confident"], {"agreements": 1, "denominator": 2, "rate": 0.5})
        self.assertEqual(result["confident_with_label_in_candidates"],
                         {"agreements": 1, "denominator": 1, "rate": 1.0})

    def test_no_defensible_labels_means_no_rate(self):
        result = summarize_proxy({"labels": []}, [])
        self.assertIsNone(result["all_confident"]["rate"])
        self.assertIsNone(result["confident_with_label_in_candidates"]["rate"])
