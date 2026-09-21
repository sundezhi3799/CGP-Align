"""Regression: relocate checkpoint data and load full weights without initializers."""
from argparse import Namespace
from pathlib import Path
import sys
import tempfile
import pathlib
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import torch
import eval_raw3180_branch_retrieval_from_checkpoints as evaluation
from eval_cgp_align_crossmodal_bridge import namespace_from_config
from cgp_align.checkpoint_paths import configure_inference
from cgp_align.checkpoint_io import load_checkpoint


class SavedLinuxPath:
    def __reduce__(self):
        return pathlib.PosixPath, ('/training/compound',)


class SavedWindowsPath:
    def __reduce__(self):
        return pathlib.WindowsPath, ('C:/training/compound',)


class PortableInference(unittest.TestCase):
    def test_foreign_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'paths.pt'
            torch.save({'linux': SavedLinuxPath(), 'windows': SavedWindowsPath()}, p)
            value = load_checkpoint(p)
            self.assertEqual(value['linux'], pathlib.PurePosixPath('/training/compound'))
            self.assertEqual(value['windows'], pathlib.PureWindowsPath('C:/training/compound'))

    def test_relocated_full_checkpoint(self):
        torch.set_num_threads(1)
        config = dict(compound_data_dir='missing/compound', gene_data_dir='missing/gene',
                      gene_protein_embedding_dir='missing/protein', embed_dim=8,
                      gnn_hidden_dim=8, gnn_layers=1, gene_hidden_dims='8',
                      profile_hidden_dims='8', modality_context_dim=0,
                      split_gene_modality_branches=True, dropout=0.0)
        for key in ('init_compound_checkpoint', 'init_gene_checkpoint',
                    'init_orf_gene_checkpoint', 'init_crispr_gene_checkpoint'):
            config[key] = 'missing/initializer.pt'
        data = {'profile_dim': 6, 'compound': {'norm': {}},
                'gene': {'protein_embeddings': np.zeros((2, 4)), 'norm': {}}}
        args = configure_inference(namespace_from_config(config))
        original, _ = evaluation.cgp.build_model(args, data, torch.device('cpu'))
        original.eval()
        overrides = Namespace(compound_data_dir=Path('relocated/compound'),
                              gene_data_dir=Path('relocated/gene'),
                              gene_protein_embedding_dir=Path('relocated/protein'))
        def build_data(received):
            for key, value in vars(overrides).items(): self.assertEqual(getattr(received, key), value)
            return data
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'complete.pt'
            payload = {'config': config, 'model_state_dict': original.state_dict(),
                       'compound_profile_normalizer': {'sentinel': 42}}
            torch.save(payload, path)
            with patch.object(evaluation.cgp, 'build_data', side_effect=build_data):
                restored, loaded, _, _ = evaluation.load_model_and_data(path, torch.device('cpu'), 2, overrides)
            self.assertEqual(loaded['compound']['norm'], {'sentinel': 42})
            protein = torch.randn(2, 4)
            modalities = torch.tensor([0, 1])
            self.assertTrue(torch.equal(original.encode_gene(protein, modalities), restored.encode_gene(protein, modalities)))
            # Incomplete weights must still fail, not silently initialize missing branches.
            payload['model_state_dict'].pop(next(iter(payload['model_state_dict'])))
            torch.save(payload, path)
            with patch.object(evaluation.cgp, 'build_data', side_effect=build_data):
                with self.assertRaises(RuntimeError):
                    evaluation.load_model_and_data(path, torch.device('cpu'), 2, overrides)


if __name__ == '__main__':
    unittest.main()
