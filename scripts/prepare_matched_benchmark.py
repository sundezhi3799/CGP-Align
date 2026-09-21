"""Build seed-specific entity-mean targets with verified fixed molecular features."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

METHODS = ('morgan', 'molformer_xl', 'chemberta')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(16 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--strict-root', type=Path, required=True)
    p.add_argument('--feature-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=(31, 37, 41), required=True)
    a = p.parse_args()
    source = a.strict_root / f'seed{a.seed}/data/compound'
    audit = json.loads((source / 'profile_correction_audit.json').read_text())
    assert audit['seed'] == a.seed and audit['fit_val_overlap'] == audit['fit_test_overlap'] == 0
    split_path = source / 'splits_compound_mocop.json'
    assert sha(split_path) == audit['split_sha256']
    folds = json.loads(split_path.read_text())['cold_compound']
    entities = pd.read_parquet(source / 'compound_mocop_entities.parquet')
    reps = pd.read_parquet(source / 'compound_mocop_replicates.parquet')
    assert entities.compound_id.astype(str).is_unique
    assert np.array_equal(entities.entity_index, np.arange(len(entities)))
    assert np.array_equal(reps.feature_index, np.arange(len(reps)))
    assert entities.has_structure.all() and entities.canonical_smiles.fillna('').ne('').all()
    sets = [set(folds[k]) for k in ('train', 'val', 'test')]
    assert set.union(*sets) == set(range(len(entities)))
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    a.output.mkdir(parents=True, exist_ok=False)
    audit_out = dict(seed=a.seed, protocol='strict_entity_mean_v1', fit_audit=audit,
                     profile_definition='arithmetic mean of strict corrected replicates, then encode once',
                     split_sha256=sha(split_path), methods={})
    for method in METHODS:
        old = a.feature_root / ('raw3180_entity_mean_' + method)
        table = pd.read_parquet(old / 'compounds.parquet')
        assert table.compound_id.astype(str).tolist() == entities.compound_id.astype(str).tolist()
        assert table.canonical_smiles.fillna('').tolist() == entities.canonical_smiles.fillna('').tolist()
        x = np.load(old / 'compound_structure_features.npy', mmap_mode='r')
        assert len(x) == len(entities)
        for start in range(0, len(x), 4096):
            part = x[start:start + 4096]
            assert np.isfinite(part).all() and np.all(np.abs(part).sum(axis=1) > 0)
        dest = a.output / method
        dest.mkdir()
        for name in ('compounds.parquet', 'genes.parquet', 'profiles.parquet', 'compound_target_edges.parquet'):
            shutil.copy2(old / name, dest / name)
        assert pd.read_parquet(dest / 'compound_target_edges.parquet').empty
        profiles = pd.read_parquet(dest / 'profiles.parquet')
        assert len(profiles) == len(entities)
        assert profiles.profile_id.astype(str).tolist() == table.compound_profile_id.astype(str).tolist()
        assert np.array_equal(profiles.feature_index, np.arange(len(entities)))
        profiles['source_dataset'] = f'strict_seed{a.seed}_entity_mean'
        profiles.to_parquet(dest / 'profiles.parquet', index=False)
        (dest / 'compound_structure_features.npy').symlink_to((old / 'compound_structure_features.npy').resolve())
        (dest / 'profile_features.npy').symlink_to((a.output / 'profile_features.npy').resolve())
        write(dest / 'dataset_summary.json', dict(profile_dim=3180, feature_dim_by_source=None,
              seed=a.seed, profile_source='strict training-fitted corrected replicate entity mean'))
        write(dest / 'splits_intrinsic_entity.json', dict(seed=a.seed, compound=folds,
              gene=dict(train=[], val=[], test=[]), summary=dict(compound_split_sizes={k:len(v) for k,v in folds.items()})))
        audit_out['methods'][method] = dict(feature_source=str(old.resolve()), shape=list(x.shape),
            feature_sha256=sha(old / 'compound_structure_features.npy'), compound_table_sha256=sha(old / 'compounds.parquet'),
            missing_ids=0, reordered_ids=0, nonfinite_rows=0, zero_rows=0)
    raw_path = source / 'compound_mocop_replicate_features.npy'
    assert sha(raw_path) == audit['output_sha256']
    features = np.load(raw_path, mmap_mode='r')
    means = np.lib.format.open_memmap(a.output / 'profile_features.npy', mode='w+', dtype='float32', shape=(len(entities), features.shape[1]))
    groups = reps.groupby('entity_index', sort=True).feature_index.apply(lambda s: s.to_numpy(dtype=np.int64))
    assert len(groups) == len(entities)
    for entity, indices in groups.items():
        means[int(entity)] = features[indices].mean(axis=0)
    means.flush()
    assert np.isfinite(means).all()
    # Validate several means independently, including train/validation/test entities.
    for fold in ('train', 'val', 'test'):
        for entity in folds[fold][:3]:
            np.testing.assert_allclose(means[entity], features[groups[entity]].mean(axis=0), rtol=0, atol=0)
    audit_out['entity_mean_sha256'] = sha(a.output / 'profile_features.npy')
    audit_out['entity_table_sha256'] = sha(source / 'compound_mocop_entities.parquet')
    audit_out['test_ids'] = entities.iloc[folds['test']].compound_id.astype(str).tolist()
    # Frozen negative candidates, reused byte-for-byte by every model and direction.
    n = len(folds['test'])
    bank = np.lib.format.open_memmap(a.output / 'negative_candidates.npy', mode='w+', dtype='int32', shape=(10, n, 1000))
    assert n > 1000
    rng = np.random.default_rng(a.seed + 20260919)
    for repeat in range(10):
        for query in range(n):
            sampled = rng.choice(n - 1, 1000, replace=False)
            bank[repeat, query] = sampled + (sampled >= query)
    bank.flush()
    audit_out['negative_candidates_sha256'] = sha(a.output / 'negative_candidates.npy')
    audit_out['negative_sampling'] = dict(ratios=[100,1000], repeats=10, replacement=False, seed=a.seed + 20260919,
                                        nested=True, shared_across_models_and_directions=True)
    write(a.output / 'audit.json', audit_out)
    print(json.dumps({k:v for k,v in audit_out.items() if k not in ('test_ids','fit_audit')}), flush=True)


if __name__ == '__main__': main()
