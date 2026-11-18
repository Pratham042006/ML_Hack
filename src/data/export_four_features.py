"""Create a four-feature export from the frozen heading-v1 snapshot."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from src.data.candidate_projection import project_candidate_positions
from src.data.pipeline import load_trip


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('csv',type=Path);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();dest=a.output_dir.resolve();meta=json.loads(a.manifest.read_text())
    if dest.exists() or dest.is_relative_to(root):raise ValueError('Use a new directory outside raw data')
    if digest(a.csv)!=meta['export_csv_sha256'] or meta['partition']!='training':raise ValueError('Training export hash or partition mismatch')
    rows=[];by_trip=defaultdict(list)
    with a.csv.open(newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames
        if fields!=['trip_id','observation_index','arc_id','distance_m','label','heading_agreement']:raise ValueError('Unexpected source schema')
        for row in reader:rows.append(row);by_trip[row['trip_id']].append(row)
    for tid,group in by_trip.items():
        directory=root/tid
        for ext,expected in meta['trips'][tid]['inference_input_sha256'].items():
            if digest(directory/(tid+'.'+ext))!=expected:raise ValueError(f'Input mismatch {tid}.{ext}')
        trip=load_trip(directory,include_reference=False)
        candidates=[{'observation_index':int(r['observation_index']),'arc_id':int(r['arc_id'])} for r in group]
        positions=project_candidate_positions(trip,candidates);closest={}
        for r in group:
            key=int(r['observation_index']);d=float(r['distance_m']);closest[key]=min(closest.get(key,float('inf')),d)
        for r in group:
            key=int(r['observation_index']);r['distance_margin_m']=float(r['distance_m'])-closest[key]
            r['projection_fraction']=positions[(key,int(r['arc_id']))]
        print('Features',tid,flush=True)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        stage=Path(tmp)/'export';stage.mkdir();out=stage/'training_candidates_four_v1.csv'
        with out.open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=fields+['distance_margin_m','projection_fraction']);w.writeheader();w.writerows(rows)
        report={'version':'four-features-v1','source_heading_sha256':meta['export_csv_sha256'],'source_snapshot_sha256':meta['source_snapshot']['csv_sha256'],'csv_sha256':digest(out),'rows':len(rows),'partition':'training','verified_ground_truth':False,'heading_config':meta['config'],'nullable_features':['heading_agreement','projection_fraction'],'missing_projection':sum(r['projection_fraction'] is None for r in rows),'eligibility_change':0}
        (stage/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');stage.rename(dest)
    print('Export SHA-256',report['csv_sha256'],flush=True)

if __name__=='__main__':main()
