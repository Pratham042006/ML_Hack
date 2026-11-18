"""Fit five heading classifiers and evaluate independently on student-test proxies."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score
from src.data.heading import HeadingConfig, add_heading_features
from src.data.pipeline import generate_candidates, load_trip
from src.models.experiments import make_models


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('training_csv', type=Path)
    p.add_argument('--heading-manifest', type=Path, required=True)
    p.add_argument('--student-root', type=Path, required=True)
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    meta = json.loads(a.heading_manifest.read_text())
    if sha(a.training_csv) != (meta['csv_sha256'] if meta.get('version') in ('four-features-v1', 'six-features-v1') else meta['export_csv_sha256']):
        raise ValueError('Training export hash mismatch')
    ledger = json.loads((a.ledger/'manifest.json').read_text())
    if ledger['partition'] != 'student-test' or ledger['failed_trips']:
        raise ValueError('Expected complete student-test ledger')
    a.output.mkdir(parents=True, exist_ok=False)
    train = pd.read_csv(a.training_csv, dtype={'trip_id': str})
    version = meta.get('version')
    contextual = version in ('four-features-v1', 'six-features-v1')
    data_config = meta['heading_config'] if contextual else meta['config']
    features = ['distance_m', 'heading_agreement'] + (['distance_margin_m', 'projection_fraction'] if contextual else []) + (['candidate_rank', 'segment_length_m'] if version == 'six-features-v1' else [])
    train.sort_values(['trip_id', 'observation_index', 'distance_m', 'arc_id'], inplace=True, ignore_index=True)
    models = make_models(42)
    models['logistic_regression'].set_params(logisticregression__solver='liblinear')
    artifact_hashes = {}
    for name, model in models.items():
        start = perf_counter()
        model.fit(train[features].to_numpy(), train.label.to_numpy())
        path = a.output/(name+'.joblib')
        joblib.dump({'format_version': 1, 'feature_names': features, 'nullable_features': (['heading_agreement', 'projection_fraction'] if contextual else ['heading_agreement']),
                     'heading_config': data_config, 'estimator': model, 'model_name': name,
                     'training_csv_sha256': sha(a.training_csv), 'verified_ground_truth': False}, path)
        artifact_hashes[name] = {'sha256': sha(path), 'training_seconds': perf_counter()-start}
        print('Fitted', name, flush=True)
    totals = {n: {'correct': 0, 'confident': 0, 'eligible': 0, 'candidate_labels': [], 'candidate_scores': [],
                  'prediction_seconds': 0.0} for n in models}
    for summary in ledger['trips']:
        tid = summary['trip_id']; trip_dir = a.student_root/tid
        for ext in ('track', 'nodes', 'arcs', 'route'):
            if sha(trip_dir/(tid+'.'+ext)) != summary['input_sha256'][ext]:
                raise ValueError(f'Input mismatch {tid}.{ext}')
        trip = load_trip(trip_dir, include_reference=False)
        if contextual:
            from src.data.candidate_features import add_candidate_features
            candidates = add_candidate_features(trip, generate_candidates(trip), HeadingConfig(**data_config))
        else:
            candidates = add_heading_features(trip, generate_candidates(trip), HeadingConfig(**data_config))
        x = np.array([[np.nan if c['features'][name] is None else c['features'][name] for name in features] for c in candidates])
        # Calculate all predictions before reading offline labels.
        scored = {}
        for name, model in models.items():
            start = perf_counter(); scored[name] = model.predict_proba(x)[:, list(model.classes_).index(1)]
            if not np.isfinite(scored[name]).all(): raise ValueError('Nonfinite model probabilities')
            totals[name]['prediction_seconds'] += perf_counter()-start
        labels = [json.loads(line) for line in (a.ledger/tid/'labels.jsonl').read_text().splitlines()]
        confident = {z['observation_index']: z for z in labels if z['status'] == 'confident'}
        groups = {}
        for i, c in enumerate(candidates): groups.setdefault(c['observation_index'], []).append(i)
        for name, scores in scored.items():
            t = totals[name]; t['confident'] += len(confident)
            for index, label in confident.items():
                if label['candidate_miss'] is not False: continue
                ids = groups[index]; t['eligible'] += 1
                winner = ids[int(np.argmax(scores[ids]))]
                t['correct'] += candidates[winner]['arc_id'] == label['arc_id']
                t['candidate_labels'].extend(int(candidates[i]['arc_id']==label['arc_id']) for i in ids)
                t['candidate_scores'].extend(float(scores[i]) for i in ids)
        print('Evaluated', tid, flush=True)
    report = {'scope': 'student-test development comparison; inferred reference labels, not verified accuracy',
              'training_csv_sha256': sha(a.training_csv), 'student_dataset_sha256': ledger['dataset_sha256'],
              'coverage': ledger['totals'], 'training_artifacts': artifact_hashes, 'models': {}}
    for name, t in totals.items():
        y, scores = t.pop('candidate_labels'), t.pop('candidate_scores')
        report['models'][name] = {**t, 'top1_including_misses': t['correct']/t['confident'] if t['confident'] else None,
                                  'conditional_top1': t['correct']/t['eligible'] if t['eligible'] else None,
                                  'candidate_macro_f1': float(f1_score(y, np.array(scores)>=.5, average='macro')),
                                  'candidate_positive_precision': float(precision_score(y, np.array(scores)>=.5, zero_division=0)),
                                  'candidate_positive_recall': float(recall_score(y, np.array(scores)>=.5, zero_division=0)),
                                  'candidate_confusion_matrix': confusion_matrix(y, np.array(scores)>=.5, labels=[0,1]).tolist(),
                                  'candidate_binary_log_loss': float(log_loss(y, scores, labels=[0,1]))}
    (a.output/'student_results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Saved report and private models:', a.output, flush=True)

if __name__ == '__main__': main()
