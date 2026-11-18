"""Add rank and segment length to the frozen four-feature training export."""
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from src.data.candidate_features import add_candidate_features
from src.data.pipeline import load_trip


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('csv',type=Path);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--heading-manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();dest=a.output_dir.resolve();m=json.loads(a.manifest.read_text());heading=json.loads(a.heading_manifest.read_text())
 if dest.exists() or dest.is_relative_to(root):raise ValueError('Use a new output directory outside raw data')
 if sha(a.csv)!=m['csv_sha256'] or m['source_heading_sha256']!=heading['export_csv_sha256']:raise ValueError('Frozen snapshot mismatch')
 rows=[];by_trip=defaultdict(list)
 with a.csv.open(newline='') as f:
  reader=csv.DictReader(f);fields=reader.fieldnames
  expected=['trip_id','observation_index','arc_id','distance_m','label','heading_agreement','distance_margin_m','projection_fraction']
  if fields!=expected:raise ValueError('Unexpected source schema')
  for row in reader:rows.append(row);by_trip[row['trip_id']].append(row)
 for tid,group in by_trip.items():
  directory=root/tid
  for ext,expected_hash in heading['trips'][tid]['inference_input_sha256'].items():
   if sha(directory/(tid+'.'+ext))!=expected_hash:raise ValueError(f'Input mismatch {tid}.{ext}')
  trip=load_trip(directory,include_reference=False)
  candidates=[{'observation_index':int(r['observation_index']),'arc_id':int(r['arc_id']),'features':{'distance_m':float(r['distance_m'])}} for r in group]
  enriched=add_candidate_features(trip,candidates)
  for row,c in zip(group,enriched):row['candidate_rank']=c['features']['candidate_rank'];row['segment_length_m']=c['features']['segment_length_m']
  print('Features',tid,flush=True)
 dest.parent.mkdir(parents=True,exist_ok=True)
 with TemporaryDirectory(dir=dest.parent) as tmp:
  stage=Path(tmp)/'export';stage.mkdir();out=stage/'training_candidates_six_v1.csv'
  with out.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields+['candidate_rank','segment_length_m']);w.writeheader();w.writerows(rows)
  report={'version':'six-features-v1','source_four_sha256':m['csv_sha256'],'source_snapshot_sha256':m['source_snapshot_sha256'],'csv_sha256':sha(out),'rows':len(rows),'partition':'training','verified_ground_truth':False,'heading_config':m['heading_config'],'nullable_features':m['nullable_features'],'eligibility_change':0,'feature_code_sha256':{name:sha(Path(__file__).with_name(name)) for name in ['candidate_features.py','candidate_projection.py','heading.py','export_six_features.py']}}
  (stage/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');stage.rename(dest)
 print('Export SHA-256',report['csv_sha256'],flush=True)

if __name__=='__main__':main()
