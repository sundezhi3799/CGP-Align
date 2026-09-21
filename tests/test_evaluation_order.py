"""Compact packaging must preserve negative-sampling order and cohort identity."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgp_align.evaluation_order import restore_compact_test_order


class EvaluationOrderTest(unittest.TestCase):
    def test_restore_without_mutating_input(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); table = root / 'entities.parquet'; table.write_bytes(b'cohort')
            config = root / 'seed31'; config.mkdir()
            meta = {'split': 'cold_compound', 'test': [2, 0, 1], 'entity_table_sha256': hashlib.sha256(table.read_bytes()).hexdigest()}
            file = config / 'compact_test_order.json'; file.write_text(json.dumps(meta))
            split = {'seed': 31, 'split_index_type': 'compact_test_entities', 'cold_compound': {'test': [0, 1, 2], 'train': [], 'val': []}}
            self.assertEqual(restore_compact_test_order(split, table, root)['cold_compound']['test'], [2, 0, 1])
            self.assertEqual(split['cold_compound']['test'], [0, 1, 2])
            meta['test'] = [2, 0, 0]; file.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'permutation'): restore_compact_test_order(split, table, root)
            table.write_bytes(b'other cohort')
            with self.assertRaisesRegex(ValueError, 'table does not match'): restore_compact_test_order(split, table, root)

    def test_full_data_unchanged(self):
        split = {'cold_compound': {'test': [8, 2, 7]}}
        self.assertIs(restore_compact_test_order(split, 'unused'), split)


if __name__ == '__main__': unittest.main()
