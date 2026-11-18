"""Training-fold ablation of candidate distance margin; no evaluator data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.models.experiments import make_models
from src.models.review_comparison import measure


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv',type=Path); p.add_argument('--heading-manifest',type=Path,required=True)
    p.add_argument('--fold-manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); m=json.loads(a.heading_manifest.read_text()); folds=json.loads(a.fold_manifest.read_text())
    h=hashlib.sha256(a.csv.read_bytes()).hexdigest()
    if h != m['export_csv_sha256'] or m['source_snapshot']['csv_sha256'] != folds['dataset_csv_sha256']: raise ValueError('Snapshot mismatch')
    data=pd.read_csv(a.csv,dtype={'trip_id':str}); keys=['trip_id','observation_index']
    data.sort_values(keys+['distance_m','arc_id'],inplace=True,ignore_index=True)
    data['distance_margin_m']=data.distance_m-data.groupby(keys).distance_m.transform('min')
    results=[]
    for i, validation in enumerate(folds['validation_trips_by_fold'],1):
        train=data[~data.trip_id.isin(validation)]; test=data[data.trip_id.isin(validation)]
        groups=test.groupby(keys,sort=False).size().to_numpy(); starts=np.r_[0,np.cumsum(groups)[:-1]]
        for columns in (['distance_m','heading_agreement'], ['distance_m','heading_agreement','distance_margin_m']):
            model=make_models(42)['hist_gradient_boosting']; model.fit(train[columns].to_numpy(),train.label.to_numpy())
            raw,scores,winners,_=measure(model,test[columns].to_numpy(),starts,groups)
            results.append({'fold':i,'features':columns,'correct':int(test.label.to_numpy()[winners].sum()),'observations':len(starts)})
            print('Fold',i,'features',len(columns),'correct',results[-1]['correct'],flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'scope':'training-trip folds; candidate-present approximate labels only','csv_sha256':h,'results':results},indent=2)+'\n')

if __name__=='__main__':main()
