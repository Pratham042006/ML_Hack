"""Compare frozen tuned and untuned histogram settings on training-trip folds."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from src.models.experiments import make_models


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv',type=Path);p.add_argument('--heading-manifest',type=Path,required=True);p.add_argument('--fold-manifest',type=Path,required=True);p.add_argument('--search',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 manifest=json.loads(a.heading_manifest.read_text());folds=json.loads(a.fold_manifest.read_text());search=json.loads(a.search.read_text());h=hashlib.sha256(a.csv.read_bytes()).hexdigest()
 is_six=manifest.get('version')=='six-features-v1'
 is_four=manifest.get('version')=='four-features-v1'
 is_context=is_six or is_four
 expected=manifest['csv_sha256'] if is_context else manifest['export_csv_sha256']
 source=manifest['source_snapshot_sha256'] if is_context else manifest['source_snapshot']['csv_sha256']
 if h!=expected or h!=search['training_csv_sha256'] or source!=folds['dataset_csv_sha256']:raise ValueError('Snapshot mismatch')
 data=pd.read_csv(a.csv,dtype={'trip_id':str});data.sort_values(['trip_id','observation_index','distance_m','arc_id'],inplace=True,ignore_index=True);features=['distance_m','heading_agreement']+(['distance_margin_m','projection_fraction'] if is_context else [])+(['candidate_rank','segment_length_m'] if is_six else []);results=[]
 for i,group in enumerate(folds['validation_trips_by_fold'],1):
  train=data[~data.trip_id.isin(group)];val=data[data.trip_id.isin(group)]
  for name,leaf in [('untuned',31),('tuned',search['best'])]:
   model=make_models(42)['hist_gradient_boosting'];model.set_params(histgradientboostingclassifier__max_leaf_nodes=leaf);model.fit(train[features].to_numpy(),train.label.to_numpy());prob=model.predict_proba(val[features].to_numpy())[:,1]
   if not np.isfinite(prob).all():raise ValueError('Nonfinite probabilities')
   results.append({'fold':i,'model':name,'max_leaf_nodes':leaf,'candidate_macro_f1':float(f1_score(val.label,prob>=.5,average='macro')),'candidate_binary_log_loss':float(log_loss(val.label,prob,labels=[0,1]))});print(i,name,flush=True)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'scope':'training-trip CV on inferred candidate-present labels','training_csv_sha256':h,'results':results},indent=2)+'\n')

if __name__=='__main__':main()
