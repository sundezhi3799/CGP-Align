"""Recompute one seed's profiles from uncorrected common-feature inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgp_align.strict_preprocessing import fit_transform


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(16 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--kind', choices=('compound', 'gene'), required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists(): raise SystemExit('Refusing to overwrite output directory: ' + str(a.output_dir))
    if (a.raw_dir / 'profile_correction_audit.json').exists():
        raise SystemExit('Input must precede project-level profile correction')
    prefix = a.kind + '_mocop'
    names = [prefix + '_entities.parquet', prefix + '_replicates.parquet']
    entities, reps = [pd.read_parquet(a.raw_dir / name) for name in names]
    for name in names:
        if sha(a.raw_dir / name) != sha(a.split_dir / name):
            raise ValueError('Raw and split directory metadata differ: ' + name)
    split_file = a.split_dir / ('splits_' + prefix + '.json')
    payload = json.loads(split_file.read_text())
    if payload['seed'] != a.seed: raise ValueError('Seed mismatch')
    folds = payload['cold_' + a.kind]
    sets = {k: set(v) for k, v in folds.items()}
    for key, values in folds.items():
        if len(sets[key]) != len(values) or not sets[key] <= set(range(len(entities))):
            raise ValueError('Invalid or duplicate split indices')
    for first, second in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if sets[first] & sets[second]: raise ValueError('Entity split overlap')
        if a.kind == 'gene':
            symbols = entities['gene_symbol'].fillna('').astype(str).str.upper()
            left = set(symbols.iloc[folds[first]]) - {''}
            right = set(symbols.iloc[folds[second]]) - {''}
            if left & right: raise ValueError('Gene symbol overlaps across modalities/folds')
    feature_path = a.raw_dir / (prefix + '_replicate_features.npy')
    x = np.load(feature_path, mmap_mode='r')
    if x.shape != (len(reps), 3180): raise ValueError('Expected aligned 3180-feature profiles')
    if not np.array_equal(reps.feature_index.to_numpy(), np.arange(len(reps))):
        raise ValueError('Feature-index order requires explicit realignment')
    fit_rows = np.flatnonzero(reps.entity_index.isin(sets['train']).to_numpy())
    for fold in ('val', 'test'):
        if reps.iloc[fit_rows].entity_index.isin(sets[fold]).any(): raise ValueError('Fit leakage')
    a.output_dir.mkdir(parents=True)
    print('Fitting ' + a.kind + ' seed ' + str(a.seed), flush=True)
    corrected, params, steps = fit_transform(x, reps, fit_rows, a.kind)
    output = a.output_dir / feature_path.name
    np.save(output, corrected)
    np.savez_compressed(a.output_dir / 'fitted_preprocessing.npz', **params)
    for name in names: shutil.copy2(a.raw_dir / name, a.output_dir / name)
    shutil.copy2(split_file, a.output_dir / split_file.name)
    for name in ('feature_columns.json', 'dataset_summary.json'):
        if (a.raw_dir / name).exists():
            shutil.copy2(a.raw_dir / name, a.output_dir / ('dataset_summary.base.json' if name == 'dataset_summary.json' else name))
    (a.output_dir / 'dataset_summary.json').write_text(json.dumps(dict(
        dataset_name=prefix + '_strict_train_corrected', seed=a.seed, feature_dim=3180,
        num_entities=len(entities), num_replicates=len(reps),
        split_sizes={key: len(value) for key, value in sets.items()}), indent=2) + '\n')
    audit = dict(schema='strict_train_fit_v1', kind=a.kind, seed=a.seed, raw_dir=str(a.raw_dir.resolve()),
                 split_sha256=sha(split_file), input_sha256=sha(feature_path), output_sha256=sha(output),
                 parameters_sha256=sha(a.output_dir / 'fitted_preprocessing.npz'),
                 entity_table_sha256=sha(a.raw_dir / names[0]), fit_feature_rows=len(fit_rows),
                 fit_entities=len(sets['train']), fit_val_overlap=0, fit_test_overlap=0,
                 gene_group_disjoint=True if a.kind == 'gene' else None,
                 correction_order='modality zscore (compound pooled), plate then well, two passes',
                 unseen_label_policy='zero correction; never fit held-out rows', steps=steps,
                 scope='project-level corrections only; upstream public profile processing is unchanged')
    (a.output_dir / 'profile_correction_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__': main()
