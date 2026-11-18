"""Checks search behavior on a tiny synthetic dataset, not model performance."""
import contextlib
import io
import unittest
import numpy as np
import pandas as pd
from src.models.bayesian_tuning import search
from src.models.experiments import evaluate_fold


class TuningTests(unittest.TestCase):
    def test_search_uses_disjoint_trip_folds_and_finite_trials(self):
        rows = []
        for trip in range(6):
            for observation in range(3):
                for arc, distance in enumerate((1 + trip, 10 + observation)):
                    rows.append({'trip_id': str(trip), 'observation_index': observation,
                                 'arc_id': arc, 'distance_m': distance, 'label': int(arc == 0)})
        data = pd.DataFrame(rows)
        with contextlib.redirect_stdout(io.StringIO()):
            best, trials = search(data, trials=5, seed=42)
        self.assertEqual(len(trials), 5)
        self.assertIn(best, [trial['log10_var_smoothing'] for trial in trials])
        self.assertTrue(all(np.isfinite(trial['mean_candidate_log_loss']) for trial in trials))
        for trial in trials:
            groups = [set(fold['validation_trips']) for fold in trial['fold_results']]
            self.assertEqual(set.union(*groups), set(data.trip_id))
            self.assertFalse(any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i)))

    def test_rejects_too_few_trials(self):
        with self.assertRaisesRegex(ValueError, 'four trials'):
            search(pd.DataFrame(), trials=3, seed=42)


if __name__ == '__main__':
    unittest.main()
