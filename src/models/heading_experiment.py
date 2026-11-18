"""Compare distance and heading on identical frozen observations and trip folds."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from src.models.experiments import make_models
from src.models.review_comparison import measure


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv', type=Path)
    p.add_argument('--heading-manifest', type=Path, required=True)
    p.add_argument('--fold-manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    heading = json.loads(a.heading_manifest.read_text())
    folds = json.loads(a.fold_manifest.read_text())
    if digest(a.csv) != heading['export_csv_sha256'] or heading['source_snapshot']['csv_sha256'] != folds['dataset_csv_sha256']:
        raise ValueError('Heading export or frozen source hash does not match the manifests')
    data = pd.read_csv(a.csv, dtype={'trip_id': str})
    keys = ['trip_id', 'observation_index']
    if data.duplicated(keys + ['arc_id']).any() or not data.groupby(keys).label.sum().eq(1).all():
        raise ValueError('Invalid candidate labels or duplicate candidates')
    if not data.heading_agreement.dropna().between(-1, 1).all():
        raise ValueError('Invalid heading')
    data.sort_values(keys + ['distance_m', 'arc_id'], inplace=True, ignore_index=True)
    counts = json.loads(Path('outputs/model_work/checked_manifest.json').read_text())['trip_counts']
    templates = make_models(42)
    templates['logistic_regression'].set_params(logisticregression__solver='liblinear')
    records = []
    all_trips = set(counts)
    for fold, trip_list in enumerate(folds['validation_trips_by_fold'], 1):
        test = set(trip_list)
        if not test <= all_trips: raise ValueError('Unknown validation trips')
        train, valid = data[~data.trip_id.isin(test)], data[data.trip_id.isin(test)]
        if train.trip_id.nunique() + valid.trip_id.nunique() != data.trip_id.nunique(): raise ValueError('Overlapping trips')
        group_sizes = valid.groupby(keys, sort=False).size().to_numpy()
        starts = np.r_[0, np.cumsum(group_sizes)[:-1]]
        y = valid.label.to_numpy()
        confident = sum(counts[t]['confident'] for t in test)
        misses = sum(counts[t]['candidate_misses'] for t in test)
        assert len(starts) + misses == confident
        for name, template in {'distance_baseline': None, **templates}.items():
            variants = ['distance'] if template is None else ['distance', 'distance_heading']
            for variant in variants:
                features = ['distance_m'] + (['heading_agreement'] if variant == 'distance_heading' else [])
                model = None if template is None else clone(template)
                start = time.perf_counter()
                if model is not None: model.fit(train[features].to_numpy(), train.label.to_numpy())
                raw, scores, winners, elapsed = measure(model, valid[features].to_numpy(), starts, group_sizes)
                correct = int(y[winners].sum())
                records.append({'fold': fold, 'model': name, 'variant': variant, 'features': features,
                                'correct': correct, 'confident': confident, 'candidate_misses': misses,
                                'conditional_observations': len(starts),
                                'candidate_macro_f1': float(f1_score(y, raw >= .5, average='macro')) if model is not None else None,
                                'conditional_log_loss': float(-np.log(scores[y == 1].clip(1e-15, 1)).mean()),
                                'seconds': time.perf_counter()-start})
                print(f'fold {fold}: {name} {variant}, {correct}/{confident}', flush=True)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    summary = []
    for name in ['distance_baseline', *templates]:
        for variant in (['distance'] if name == 'distance_baseline' else ['distance', 'distance_heading']):
            rows = [r for r in records if r['model'] == name and r['variant'] == variant]
            summary.append({'model': name, 'variant': variant,
                            'top1_including_misses': sum(r['correct'] for r in rows)/sum(r['confident'] for r in rows),
                            'conditional_top1': sum(r['correct'] for r in rows)/sum(r['conditional_observations'] for r in rows),
                            'candidate_macro_f1_mean': float(np.mean([r['candidate_macro_f1'] for r in rows])) if name != 'distance_baseline' else None})
    report = {'scope': 'approximate labels; confident observations; misses counted as wrong; not verified accuracy',
              'heading_csv_sha256': digest(a.csv), 'source_csv_sha256': folds['dataset_csv_sha256'],
              'missing_headings': int(data.heading_agreement.isna().sum()), 'folds': records, 'summary': summary}
    (a.output_dir/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    best = max((r for r in summary if r['variant'] == 'distance_heading'), key=lambda r:r['top1_including_misses'])
    model = clone(templates[best['model']]).fit(data[['distance_m','heading_agreement']].to_numpy(), data.label.to_numpy())
    joblib.dump({'format_version': 1, 'feature_names': ['distance_m','heading_agreement'],
                 'nullable_features': ['heading_agreement'], 'heading_config': heading['config'], 'estimator': model,
                 'model_name': best['model'], 'training_csv_sha256': digest(a.csv),
                 'label_source': 'approximate_reference_alignment', 'verified_ground_truth': False},
                a.output_dir/'provisional_heading_model.joblib')
    print('Provisional heading model:', best['model'], flush=True)

if __name__ == '__main__': main()
