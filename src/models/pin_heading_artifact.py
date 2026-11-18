"""Copy a trusted legacy heading artifact and pin its feature configuration."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
from src.models.inference import _load_artifact


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('artifact', type=Path)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--heading-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source_hash = digest(a.artifact)
    if source_hash != a.expected_sha256.lower():
        raise ValueError('Source artifact SHA-256 mismatch')
    if a.output.exists() or a.output.resolve() == a.artifact.resolve():
        raise ValueError('Use a new output file')
    manifest = json.loads(a.heading_manifest.read_text())
    artifact = _load_artifact(a.artifact)
    if artifact['feature_names'] != ['distance_m', 'heading_agreement']:
        raise ValueError('Unexpected feature schema')
    if artifact.get('training_csv_sha256') != manifest['export_csv_sha256']:
        raise ValueError('Artifact training CSV does not match heading manifest')
    if artifact.get('nullable_features') != ['heading_agreement']:
        raise ValueError('Unexpected nullable feature policy')
    settings = manifest['config']
    if settings != {'min_displacement_m': 5.0, 'max_gap_seconds': 10.0}:
        raise ValueError('Source model was evaluated only with heading-v1 settings')
    artifact = {**artifact, 'heading_config': settings,
                'source_artifact_sha256': source_hash,
                'heading_code_sha256': manifest['feature_code_sha256']['heading.py']}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, a.output)
    print('Pinned artifact SHA-256:', digest(a.output))

if __name__ == '__main__': main()
