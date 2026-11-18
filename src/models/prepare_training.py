"""Combine permitted alignment exports and preserve a coverage/hash manifest."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('alignment_dir', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--manifest', required=True, type=Path)
    args = p.parse_args()
    folders = sorted(x for x in args.alignment_dir.iterdir() if x.is_dir())
    if not folders:
        raise ValueError('No alignment exports found')
    fields = ['trip_id', 'observation_index', 'arc_id', 'distance_m', 'label']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts, hashes, methods, sources = {}, {}, set(), set()
    with args.output.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for folder in folders:
            provenance = json.loads((folder / 'provenance.json').read_text())
            summary = json.loads((folder / 'summary.json').read_text())
            if provenance['partition'] != 'training' or provenance['verified_ground_truth']:
                raise ValueError(f'{folder}: expected approximate training labels')
            if provenance['trip_id'] != folder.name or summary['trip_id'] != folder.name:
                raise ValueError(f'{folder}: inconsistent trip identity')
            methods.add(provenance['method_version'])
            sources.add(json.dumps(provenance['source_sha256'], sort_keys=True))
            counts[folder.name] = summary['counts']
            source = folder / 'approximate_candidates.csv'
            hashes[folder.name] = sha256(source)
            with source.open(newline='') as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != fields:
                    raise ValueError(f'{source}: unexpected columns')
                for row in reader:
                    if row['trip_id'] != folder.name:
                        raise ValueError(f'{source}: unexpected trip ID')
                    writer.writerow(row)
    if len(methods) != 1 or len(sources) != 1:
        raise ValueError('Exports use different alignment code or methods; regenerate consistently')
    manifest = {'label_source': 'approximate_reference_alignment', 'verified_ground_truth': False,
                'partition': 'training', 'method_version': next(iter(methods)),
                'source_sha256': json.loads(next(iter(sources))), 'trip_counts': counts,
                'export_sha256': hashes, 'combined_sha256': sha256(args.output)}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Combined {len(folders)} training exports into {args.output}')


if __name__ == '__main__':
    main()
