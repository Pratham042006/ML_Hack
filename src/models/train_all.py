"""Fit and save the five candidate classifiers on a frozen training export.

This fits on all supplied rows; it does not evaluate performance. Use only an
explicitly authorized training export and preserve the provenance manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
from src.models.experiments import load_candidate_data, make_models


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory, outside raw data')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    digest = sha256(args.csv)
    if digest != args.expected_sha256.lower():
        raise ValueError('Training CSV does not match the expected SHA-256')
    data = load_candidate_data(args.csv, ['distance_m'])
    if set(data.label.unique()) != {0, 1}:
        raise ValueError('Training data must contain both candidate classes')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    models = make_models(args.seed)
    # Match the configuration used in the documented five-model comparison.
    models['logistic_regression'].set_params(logisticregression__solver='liblinear')
    manifest = {'training_csv_sha256': digest, 'features': ['distance_m'],
                'training_rows': len(data), 'training_observations': len(data.groupby(['trip_id','observation_index'])),
                'training_trips': sorted(data.trip_id.unique().tolist()), 'seed': args.seed,
                'label_source': 'approximate_reference_alignment', 'verified_ground_truth': False,
                'scope': 'candidate-present observations with heuristic confident inferred labels; no evaluation',
                'models': {}}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(data[['distance_m']].to_numpy(), data.label.to_numpy())
        path = args.output_dir / (name + '.joblib')
        joblib.dump({'format_version': 1, 'feature_names': ['distance_m'], 'estimator': model,
                     'model_name': name, 'training_csv_sha256': digest,
                     'label_source': manifest['label_source'], 'verified_ground_truth': False}, path)
        manifest['models'][name] = {'artifact': path.name, 'artifact_sha256': sha256(path),
                                    'training_seconds': time.perf_counter() - start,
                                    'configuration': repr(model)}
        print(f'Trained {name}', flush=True)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved five models to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
