"""Bayesian search for histogram-boosting leaf count on frozen training trips."""
import argparse, hashlib, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import log_loss
from src.models.experiments import make_models


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv',type=Path);p.add_argument('--heading-manifest',type=Path,required=True);p.add_argument('--fold-manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--trials',type=int,default=6);a=p.parse_args()
 manifest=json.loads(a.heading_manifest.read_text());folds=json.loads(a.fold_manifest.read_text());h=hashlib.sha256(a.csv.read_bytes()).hexdigest()
 is_six=manifest.get('version')=='six-features-v1'
 is_four=manifest.get('version')=='four-features-v1'
 is_context=is_six or is_four
 source_hash=manifest['source_snapshot_sha256'] if is_context else manifest['source_snapshot']['csv_sha256']
 expected_hash=manifest['csv_sha256'] if is_context else manifest['export_csv_sha256']
 if h!=expected_hash or source_hash!=folds['dataset_csv_sha256']:raise ValueError('Training snapshot mismatch')
 if a.trials<4:raise ValueError('At least four trials are required')
 if a.output_dir.exists():raise ValueError('Use a new output directory')
 data=pd.read_csv(a.csv,dtype={'trip_id':str});data.sort_values(['trip_id','observation_index','distance_m','arc_id'],inplace=True,ignore_index=True);features=['distance_m','heading_agreement']+(['distance_margin_m','projection_fraction'] if is_context else [])+(['candidate_rank','segment_length_m'] if is_six else []);tested=[];losses=[];records=[]; choices=np.arange(7,64)
 for i in range(a.trials):
  if i<4: leaf=[7,15,31,63][i]
  else:
   gp=GaussianProcessRegressor(kernel=Matern(length_scale=10.,length_scale_bounds='fixed',nu=2.5)+WhiteKernel(noise_level=.0001,noise_level_bounds='fixed'),optimizer=None,normalize_y=False)
   gp.fit(np.array(tested).reshape(-1,1),np.array(losses)); points=choices.reshape(-1,1);cov=gp.kernel_(points,gp.X_train_);mu=np.einsum('ij,j->i',cov,gp.alpha_);v=solve_triangular(gp.L_,cov.T,lower=True);sd=np.sqrt(np.maximum(gp.kernel_.diag(points)-np.einsum('ij,ij->j',v,v),0));imp=min(losses)-mu-.0001;z=np.divide(imp,sd,out=np.zeros_like(mu),where=sd>0);ei=imp*norm.cdf(z)+sd*norm.pdf(z);ei[np.isin(choices,tested)]=-np.inf;leaf=int(choices[np.argmax(ei)])
  vals=[]
  for group in folds['validation_trips_by_fold']:
   train=data[~data.trip_id.isin(group)];val=data[data.trip_id.isin(group)];model=make_models(42)['hist_gradient_boosting'];model.set_params(histgradientboostingclassifier__max_leaf_nodes=leaf);model.fit(train[features].to_numpy(),train.label.to_numpy());prob=model.predict_proba(val[features].to_numpy())[:,1]
   if not np.isfinite(prob).all():raise ValueError('Non-finite prediction')
   vals.append(float(log_loss(val.label,prob,labels=[0,1])))
  tested.append(leaf);losses.append(float(np.mean(vals)));records.append({'trial':i+1,'max_leaf_nodes':leaf,'fold_log_loss':vals,'mean_log_loss':losses[-1]});print('Trial',i+1,'leaf',leaf,'loss',round(losses[-1],5),flush=True)
 best=tested[int(np.argmin(losses))];final=make_models(42)['hist_gradient_boosting'];final.set_params(histgradientboostingclassifier__max_leaf_nodes=best);final.fit(data[features].to_numpy(),data.label.to_numpy());a.output_dir.mkdir(parents=True,exist_ok=False)
 joblib.dump({'format_version':1,'feature_names':features,'nullable_features':(['heading_agreement','projection_fraction'] if is_context else ['heading_agreement']),'heading_config':(manifest['heading_config'] if is_context else manifest['config']),'estimator':final,'model_name':'tuned_hist_gradient_boosting','training_csv_sha256':h,'verified_ground_truth':False},a.output_dir/'model.joblib')
 (a.output_dir/'search.json').write_text(json.dumps({'scope':'training-trip CV, candidate-present approximate labels','objective':'mean candidate binary log loss','search':'Gaussian-process expected improvement','parameter':'max_leaf_nodes','range':[7,63],'trials':records,'best':best,'training_csv_sha256':h},indent=2)+'\n');print('Best max_leaf_nodes:',best,flush=True)

if __name__=='__main__':main()
