"""Checks missing-heading behavior through feature generation and scoring."""
import unittest
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from src.data.heading import add_heading_features
from src.models.inference import score_candidates


class HeadingIntegrationTests(unittest.TestCase):
    def test_four_feature_artifact_scores_null_projection(self):
        model = make_pipeline(SimpleImputer(), LogisticRegression()).fit(
            [[1., np.nan, 0., np.nan], [3., 1., 2., .5], [1., 1., 0., .2], [4., -1., 3., .8]],
            [1, 0, 1, 0])
        artifact = {'format_version': 1,
                    'feature_names': ['distance_m', 'heading_agreement', 'distance_margin_m', 'projection_fraction'],
                    'nullable_features': ['heading_agreement', 'projection_fraction'], 'estimator': model}
        candidates = [{'observation_index': 0, 'arc_id': 0, 'features': {'distance_m': 1., 'heading_agreement': None,
                         'distance_margin_m': 0., 'projection_fraction': None}, 'geometry': []},
                      {'observation_index': 0, 'arc_id': 1, 'features': {'distance_m': 3., 'heading_agreement': 1.,
                         'distance_margin_m': 2., 'projection_fraction': .5}, 'geometry': []}]
        result = score_candidates(candidates, artifact)
        self.assertAlmostEqual(sum(item['score'] for item in result), 1.)

    def test_missing_and_present_heading_score_with_fold_fitted_imputer(self):
        trip = {'observations': [{'observation_index': 0, 'longitude': 0., 'latitude': 0., 'timestamp': 1.},
                                 {'observation_index': 1, 'longitude': .001, 'latitude': 0., 'timestamp': 2.}],
                'network': {'nodes': [[0., 0.], [.002, 0.]], 'arcs': [[0, 1], [1, 0]]}}
        candidates = [{'observation_index': i, 'arc_id': j, 'features': {'distance_m': 1.}, 'geometry': []}
                      for i in (0, 1) for j in (0, 1)]
        enriched = add_heading_features(trip, candidates)
        self.assertIsNone(enriched[0]['features']['heading_agreement'])
        self.assertGreater(enriched[2]['features']['heading_agreement'], .99)
        self.assertLess(enriched[3]['features']['heading_agreement'], -.99)
        self.assertNotIn('heading_agreement', candidates[0]['features'])
        model = make_pipeline(SimpleImputer(), LogisticRegression()).fit(
            [[1., np.nan], [2., np.nan], [1., 1.], [2., -1.]], [1, 0, 1, 0])
        artifact = {'format_version': 1, 'feature_names': ['distance_m', 'heading_agreement'],
                    'nullable_features': ['heading_agreement'], 'estimator': model}
        scored = score_candidates(enriched, artifact)
        self.assertAlmostEqual(scored[0]['score'] + scored[1]['score'], 1.)
        self.assertGreater(scored[2]['score'], scored[3]['score'])
        with self.assertRaisesRegex(ValueError, 'heading_agreement'):
            score_candidates(candidates, artifact)

if __name__ == '__main__': unittest.main()
