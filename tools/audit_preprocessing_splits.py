"""Audit whether shared correction-fit entities overlap later evaluation folds."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True, help='Root of the historical research workspace.')
    p.add_argument('--models', type=Path, default=Path(__file__).resolve().parents[1] / 'manifests/models.json')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = []
    for model in json.loads(args.models.read_text()):
        for kind in ('compound', 'gene'):
            directory = args.data_root / model[kind + '_data_dir']
            shared = (directory / (kind + '_mocop_replicate_features.npy')).resolve(strict=True).parent
            filename = 'splits_' + kind + '_mocop.json'
            original = json.loads((shared / filename).read_text())
            current = json.loads((directory / filename).read_text())
            # Comparing row indices is valid only for the same entity table.
            entities = kind + '_mocop_entities.parquet'
            if (directory / entities).read_bytes() != (shared / entities).read_bytes():
                raise ValueError('Entity tables differ: cannot compare row indices for ' + str(directory))
            split = 'cold_' + kind
            fit = set(original[split]['train'])
            rows.append(dict(kind=kind, seed=model['seed'], normalization_split_seed=original.get('seed'),
                             fit_entities=len(fit), test_entities=len(current[split]['test']),
                             test_entities_in_normalization_fit=len(fit.intersection(current[split]['test'])),
                             val_entities_in_normalization_fit=len(fit.intersection(current[split]['val']))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows, indent=2))
    if any(r['test_entities_in_normalization_fit'] or r['val_entities_in_normalization_fit'] for r in rows):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
