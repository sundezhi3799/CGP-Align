"""Check entity identities before analyses requiring the full candidate universe."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def identity_digest(table, columns):
    values = table[columns].fillna('').astype(str).values.tolist()
    return hashlib.sha256(json.dumps(values, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def require_full_universe(directory, compound=True, metadata=False, config=None):
    directory = Path(directory)
    config = Path(config) if config else Path(__file__).resolve().parents[1] / 'configs/analysis_universe.json'
    expected = json.loads(config.read_text())
    for branch in (['compound', 'gene'] if compound else ['gene']):
        info = expected[branch]
        table = pd.read_parquet(directory / f'{branch}_entities.parquet')
        if len(table) != info['rows'] or identity_digest(table, info['columns']) != info['identity_sha256']:
            raise ValueError(f'{directory}: {branch} candidate universe differs from the full downstream inputs. '
                             'Use downstream-embedding-cache; embeddings from compact test inputs cannot replace it.')
        matrix = np.load(directory / f'{branch}.npy', mmap_mode='r')
        if matrix.ndim != 2 or len(matrix) != len(table):
            raise ValueError(f'{branch} embedding rows do not match entity identities')
        if branch == 'gene' and metadata:
            gene_csv = pd.read_csv(directory / 'gene_metadata.csv')
            if identity_digest(gene_csv, info['columns']) != info['identity_sha256']:
                raise ValueError('Gene metadata order differs from the full gene embeddings')
