"""Check API routing diagnostics without reference inputs or network calls."""
import unittest
from pathlib import Path
from unittest.mock import patch
from src.api.runner import PipelineError, run_prediction


class DiagnosticsTests(unittest.TestCase):
    def test_reset_blocks_are_exposed_without_reference(self):
        obs = [{'observation_index': 0, 'longitude': 0, 'latitude': 0, 'timestamp': 1},
               {'observation_index': 1, 'longitude': 1, 'latitude': 0, 'timestamp': 40}]
        trip = {'observations': obs, 'network': {'nodes': [[0, 0], [1, 0]], 'arcs': [[0, 1]]}}
        predictions = [{'observation_index': i, 'arc_id': 0, 'geometry': [[0,0],[1,0]]} for i in range(2)]
        events = [{'observation_index': 1, 'event': 'large_gap_reset', 'delta_seconds': 39}]
        def loader(path, include_reference=False):
            self.assertFalse(include_reference)
            return trip
        components = {'loader': loader, 'candidates': lambda t: [], 'scorer': lambda c, model_artifact=None: [],
                      'decoder': lambda o, s, n: predictions}
        with patch('src.api.runner.resolve_components', return_value=(components, [], [])), \
             patch('src.api.config.DECODER', 'midpoint'), \
             patch('src.routing.decoder.decode_with_diagnostics', return_value=(predictions, events)):
            result = run_prediction(Path('synthetic'), None)
        self.assertEqual(result['diagnostics']['events'], events)
        self.assertEqual(len(result['diagnostics']['matched_blocks']), 2)
        self.assertEqual(result['decoder'], 'midpoint')
        self.assertEqual(result['mode'], 'baseline')
        self.assertNotIn('reference', str(result))

    def test_missing_model_is_actionable(self):
        with patch('src.api.runner.resolve_components', return_value=({'x': 1}, [], [])):
            with self.assertRaisesRegex(PipelineError, 'not found'):
                run_prediction(Path('synthetic'), '/nonexistent/pathsnap/model.joblib')

if __name__ == '__main__': unittest.main()
