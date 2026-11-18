"""Check direction, relative distances and missing geometry features."""
import unittest
from src.data.candidate_features import add_candidate_features


class CandidateFeatureTests(unittest.TestCase):
 def test_features_keep_inputs_and_handle_reverse_and_zero_length(self):
  trip={'observations':[{'observation_index':0,'longitude':0.00025,'latitude':0.,'timestamp':1.}, {'observation_index':1,'longitude':0.0005,'latitude':0.,'timestamp':2.}], 'network':{'nodes':[[0.,0.],[.001,0.]],'arcs':[[0,1],[1,0],[0,0]]}}
  cs=[{'observation_index':i,'arc_id':j,'features':{'distance_m':float(j+1)},'geometry':[]} for i in (0,1) for j in range(3)]
  out=add_candidate_features(trip,cs)
  self.assertAlmostEqual(out[0]['features']['projection_fraction'],.25)
  self.assertAlmostEqual(out[1]['features']['projection_fraction'],.75)
  self.assertIsNone(out[2]['features']['projection_fraction'])
  self.assertEqual(out[0]['features']['distance_margin_m'],0.)
  self.assertEqual(out[2]['features']['distance_margin_m'],2.)
  self.assertEqual(out[0]['features']['candidate_rank'],1)
  self.assertEqual(out[2]['features']['candidate_rank'],3)
  self.assertGreater(out[0]['features']['segment_length_m'],100.)
  self.assertEqual(out[2]['features']['segment_length_m'],0.)
  self.assertIsNone(out[0]['features']['heading_agreement'])
  self.assertNotIn('distance_margin_m',cs[0]['features'])

if __name__=='__main__':unittest.main()
