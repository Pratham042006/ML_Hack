"""Four-feature grouped training comparison using the frozen heading folds."""
import argparse, hashlib, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from src.models.experiments import make_models
from src.models.review_comparison import measure


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv',type=Path);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--fold-manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 meta=json.loads(a.manifest.read_text());folds=json.loads(a.fold_manifest.read_text());sha=hashlib.sha256(a.csv.read_bytes()).hexdigest()
 if sha!=meta['csv_sha256'] or meta['source_snapshot_sha256']!=folds['dataset_csv_sha256']:raise ValueError('Snapshot mismatch')
 if a.output_dir.exists():raise ValueError('Use a new output directory')
 data=pd.read_csv(a.csv,dtype={'trip_id':str});keys=['trip_id','observation_index'];features=['distance_m','heading_agreement','distance_margin_m','projection_fraction']
 if len(data)!=meta['rows'] or data.duplicated(keys+['arc_id']).any() or not data.groupby(keys).label.sum().eq(1).all():raise ValueError('Invalid candidate data')
 for name in ['heading_agreement','projection_fraction']:
  low,high=(-1,1) if name=='heading_agreement' else (0,1)
  if not data[name].dropna().between(low,high).all():raise ValueError('Invalid '+name)
 if not (data.distance_margin_m>=0).all():raise ValueError('Invalid distance margin')
 data.sort_values(keys+['distance_m','arc_id'],inplace=True,ignore_index=True)
 counts=json.loads(Path('outputs/model_work/checked_manifest.json').read_text())['trip_counts'];templates=make_models(42);templates['logistic_regression'].set_params(logisticregression__solver='liblinear');records=[]
 for fold,validation in enumerate(folds['validation_trips_by_fold'],1):
  val=set(validation);train=data[~data.trip_id.isin(val)];test=data[data.trip_id.isin(val)];sizes=test.groupby(keys,sort=False).size().to_numpy();starts=np.r_[0,np.cumsum(sizes)[:-1]];y=test.label.to_numpy();den=sum(counts[t]['confident'] for t in val);miss=sum(counts[t]['candidate_misses'] for t in val);assert len(starts)+miss==den
  for name,template in templates.items():
   model=clone(template).fit(train[features].to_numpy(),train.label.to_numpy());raw,scores,winners,_=measure(model,test[features].to_numpy(),starts,sizes)
   records.append({'fold':fold,'model':name,'correct':int(y[winners].sum()),'confident':den,'eligible':len(starts),'candidate_macro_f1':float(f1_score(y,raw>=.5,average='macro'))});print('Fold',fold,name,records[-1]['correct'],'/',den,flush=True)
 summary=[]
 for name in templates:
  rows=[r for r in records if r['model']==name];summary.append({'model':name,'top1_including_misses':sum(r['correct'] for r in rows)/sum(r['confident'] for r in rows),'conditional_top1':sum(r['correct'] for r in rows)/sum(r['eligible'] for r in rows),'candidate_macro_f1_mean':float(np.mean([r['candidate_macro_f1'] for r in rows]))})
 best=max(summary,key=lambda r:r['top1_including_misses']);model=clone(templates[best['model']]).fit(data[features].to_numpy(),data.label.to_numpy());a.output_dir.mkdir(parents=True,exist_ok=False)
 joblib.dump({'format_version':1,'feature_names':features,'nullable_features':meta['nullable_features'],'heading_config':meta['heading_config'],'estimator':model,'model_name':best['model']+'_four_features','training_csv_sha256':sha,'verified_ground_truth':False},a.output_dir/'provisional_four_feature_model.joblib')
 (a.output_dir/'results.json').write_text(json.dumps({'scope':'trip-grouped training CV, approximate labels, candidate misses included','csv_sha256':sha,'records':records,'summary':summary,'selected':best['model']},indent=2)+'\n');print('Provisional model',best['model'],flush=True)

if __name__=='__main__':main()
