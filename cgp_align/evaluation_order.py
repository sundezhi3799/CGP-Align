"""Restore the recorded query order for compact held-out compound inputs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def restore_compact_test_order(splits, entity_table, config_root=None):
    if splits.get('split_index_type') != 'compact_test_entities':
        return splits
    root = Path(config_root) if config_root is not None else Path(__file__).resolve().parents[1] / 'configs'
    metadata = json.loads((root / f"seed{int(splits['seed'])}" / 'compact_test_order.json').read_text())
    if hashlib.sha256(Path(entity_table).read_bytes()).hexdigest() != metadata['entity_table_sha256']:
        raise ValueError('Compact compound table does not match the recorded evaluation order')
    key = metadata['split']
    order = metadata['test']
    current = splits[key]['test']
    if len(order) != len(set(order)) or sorted(order) != sorted(current):
        raise ValueError('Recorded evaluation order must be a permutation of the test rows')
    result = deepcopy(splits)
    result[key]['test'] = order
    return result
