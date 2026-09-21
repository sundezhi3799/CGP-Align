"""Validate profile inputs and align their rows to the matched query identities."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def query_rows(entities, test_rows, test_ids):
    ids = entities.compound_id.astype(str)
    wanted = list(map(str, test_ids))
    if ids.duplicated().any() or len(wanted) != len(set(wanted)):
        raise ValueError('Compound identities must be unique')
    test_rows = np.asarray(test_rows, dtype=np.int64)
    if len(test_rows) != len(set(test_rows.tolist())) or (test_rows < 0).any() or (test_rows >= len(ids)).any():
        raise ValueError('Invalid source test rows')
    if set(ids.iloc[test_rows]) != set(wanted):
        raise ValueError('Source and matched test compounds differ')
    lookup = {value: index for index, value in enumerate(ids)}
    return np.asarray([lookup[value] for value in wanted], dtype=np.int64)


def validated_profile_input(source, audit, manifest=None):
    source = Path(source)
    split = json.loads((source / 'splits_compound_mocop.json').read_text())
    if int(split['seed']) != int(audit['seed']):
        raise ValueError('Profile input seed differs from matched audit')
    compact = split.get('split_index_type') == 'compact_test_entities'
    names = ['compound_mocop_entities.parquet', 'splits_compound_mocop.json',
             'compound_mocop_replicate_features.npy', 'compound_mocop_replicates.parquet']
    if compact:
        manifest = Path(manifest) if manifest else Path(__file__).resolve().parents[1] / 'configs/downloads/compact_main_inputs_20260921.json'
        records = json.loads(manifest.read_text())
        expected = {f['path']: f['sha256'] for g in records['groups'] for f in g['files']}
        expected = {name: expected[f'data/seed{audit["seed"]}/compound/{name}'] for name in names}
    else:
        expected = {names[0]: audit['entity_table_sha256'], names[1]: audit['split_sha256'],
                    names[2]: audit['fit_audit']['output_sha256']}
    hashes = {name: sha256(source / name) for name in names}
    for name, digest in expected.items():
        if hashes[name] != digest:
            raise ValueError(f'Profile input checksum mismatch: {source / name}')
    entities = pd.read_parquet(source / names[0])
    # The compact entity_index column retains full-table provenance. Replicate
    # ownership and GraphStore use positional rows in the supplied table.
    rows = query_rows(entities, split['cold_compound']['test'], audit['test_ids'])
    return entities, rows, hashes, 'compact' if compact else 'full'
