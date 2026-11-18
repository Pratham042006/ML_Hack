"""Compare trusted tuned and untuned heading artifacts on student-test proxies."""
import argparse, hashlib, json, time
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import f1_score, log_loss
from src.data.heading import HeadingConfig, add_heading_features
from src.data.pipeline import load_trip, generate_candidates


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--ledger',type=Path,required=True);p.add_argument('--untuned',type=Path,required=True);p.add_argument('--tuned',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 ledger=json.loads((a.ledger/'manifest.json').read_text());
 if ledger['partition']!='student-test' or ledger['failed_trips']:raise ValueError('Expected complete student-test ledger')
 artifacts={n:joblib.load(path) for n,path in [('untuned',a.untuned),('tuned',a.tuned)]}
 if artifacts['untuned']['training_csv_sha256']!=artifacts['tuned']['training_csv_sha256']:raise ValueError('Training snapshot mismatch')
 features=artifacts['untuned']['feature_names']
 if features not in [['distance_m','heading_agreement'],['distance_m','heading_agreement','distance_margin_m','projection_fraction'],['distance_m','heading_agreement','distance_margin_m','projection_fraction','candidate_rank','segment_length_m']]:raise ValueError('Unexpected feature schema')
 for v in artifacts.values():
  if v['feature_names']!=features or v.get('heading_config')!={'min_displacement_m':5.0,'max_gap_seconds':10.0}:raise ValueError('Unexpected feature configuration')
 out={k:{'correct':0,'confident':0,'eligible':0,'labels':[],'scores':[],'seconds':0.0} for k in artifacts}
 for summary in ledger['trips']:
  tid=summary['trip_id']; directory=a.root/tid
  for ext in ('track','nodes','arcs','route'):
   if sha(directory/(tid+'.'+ext))!=summary['input_sha256'][ext]:raise ValueError('Input mismatch')
  trip=load_trip(directory,include_reference=False)
  if len(features)>2:
   from src.data.candidate_features import add_candidate_features
   candidates=add_candidate_features(trip,generate_candidates(trip),HeadingConfig())
  else:candidates=add_heading_features(trip,generate_candidates(trip),HeadingConfig())
  x=np.array([[np.nan if c['features'][feature] is None else c['features'][feature] for feature in features] for c in candidates]);probs={}
  for name,art in artifacts.items():
   start=time.perf_counter();probs[name]=art['estimator'].predict_proba(x)[:,1];out[name]['seconds']+=time.perf_counter()-start
   if not np.isfinite(probs[name]).all():raise ValueError('Nonfinite probabilities')
  # Read offline labels only after both score arrays are fixed.
  labels=[json.loads(line) for line in (a.ledger/tid/'labels.jsonl').read_text().splitlines()];group={}
  for i,c in enumerate(candidates):group.setdefault(c['observation_index'],[]).append(i)
  for name,scores in probs.items():
   t=out[name]
   for l in labels:
    if l['status']!='confident':continue
    t['confident']+=1
    if l['candidate_miss'] is not False:continue
    indices=group[l['observation_index']];t['eligible']+=1;t['correct']+=candidates[indices[int(np.argmax(scores[indices]))]]['arc_id']==l['arc_id'];t['labels'].extend(int(candidates[i]['arc_id']==l['arc_id']) for i in indices);t['scores'].extend(float(scores[i]) for i in indices)
  print('Compared',tid,flush=True)
 report={'scope':'student-test development; inferred labels, not verified accuracy','coverage':ledger['totals'],'models':{}}
 for name,t in out.items():
  y=t.pop('labels');s=t.pop('scores');report['models'][name]={**t,'top1_including_misses':t['correct']/t['confident'],'candidate_macro_f1':float(f1_score(y,np.array(s)>=.5,average='macro')),'candidate_binary_log_loss':float(log_loss(y,s,labels=[0,1])),'artifact_sha256':sha(getattr(a,name))}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
