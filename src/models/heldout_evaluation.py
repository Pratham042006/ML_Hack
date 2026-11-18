"""Run a pre-registered paired comparison on the student-test partition only."""
import argparse
import hashlib
import json
from pathlib import Path
from src.routing.projection_comparison import compare_trip


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--artifact', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.root.resolve().name != 'student_test' or plan['partition'] != 'student_test':
        raise ValueError('This experiment is restricted to the student_test partition')
    if sha(args.artifact) != plan['artifact_sha256']:
        raise ValueError('Artifact differs from frozen plan')
    for trip, files in plan['inputs'].items():
        for ext, expected in files.items():
            if sha(args.root / trip / (trip + '.' + ext)) != expected:
                raise ValueError(f'Input mismatch: {trip}.{ext}')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for trip in sorted(plan['inputs']):
        result = compare_trip(args.root / trip, with_proxy=True, model_artifact=args.artifact)
        (args.output_dir / (trip + '.json')).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print('Completed', trip, flush=True)

if __name__ == '__main__': main()
