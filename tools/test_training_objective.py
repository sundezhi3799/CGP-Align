"""Compare production loss values/gradients with the author's submission reference."""
import ast
from pathlib import Path
import runpy
import unittest
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'scripts/train_cgp_align_replicate.py').read_text(encoding='utf-8-sig')
# Execute the production functions unchanged without importing dataset/RDKit loaders.
tree = ast.parse(SOURCE)
names = {'multipositive_contrastive_loss', 'modality_balanced_contrastive_loss_parts'}
ns = dict(torch=torch, F=F, List=List, Dict=Dict, Tuple=Tuple, MODALITY_TO_ID={'orf': 0, 'crispr': 1})
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), '<production-loss-functions>', 'exec'), ns)
reference = runpy.run_path(str(ROOT / 'cgp_align/losses.py'))['cgp_align_objective']

class ObjectiveTest(unittest.TestCase):
    def check_case(self, n_crispr):
        torch.manual_seed(19)
        sizes = [4, 6, 3, 5, n_crispr, n_crispr + 1 if n_crispr else 0]
        z = [F.normalize(torch.randn(n, 7, dtype=torch.float64), dim=1).requires_grad_() for n in sizes]
        ids = [torch.arange(n) // 2 for n in sizes]
        expected = reference(z[0], z[1], ids[0], ids[1], z[2], z[3], ids[2], ids[3], z[4], z[5], ids[4], ids[5])['total']
        qm = torch.cat([torch.zeros(sizes[2]), torch.ones(sizes[4])]).long()
        gm = torch.cat([torch.zeros(sizes[3]), torch.ones(sizes[5])]).long()
        args = (torch.cat([z[2], z[4]]), torch.cat([z[3], z[5]]), torch.cat([ids[2], ids[4]]), torch.cat([ids[3], ids[5]]), qm, gm, .07)
        gene, parts, _ = ns['modality_balanced_contrastive_loss_parts'](*args)
        actual = ns['multipositive_contrastive_loss'](z[0], z[1], ids[0], ids[1], .07) + gene
        torch.testing.assert_close(actual, expected)
        active = [x for x in z if x.numel()]
        for a, b in zip(torch.autograd.grad(actual, active, retain_graph=True), torch.autograd.grad(expected, active)):
            torch.testing.assert_close(a, b)
        legacy, _, _ = ns['modality_balanced_contrastive_loss_parts'](*args, reduction='mean')
        torch.testing.assert_close(legacy, gene / len(parts))

    def test_both_modalities(self): self.check_case(2)
    def test_absent_modality(self): self.check_case(0)
    def test_training_wiring(self):
        self.assertIn('reduction=args.gene_branch_loss_reduction', SOURCE)
        self.assertIn('choices=["sum", "mean"], default="sum"', SOURCE)
        wrapper = (ROOT / 'tools/train_primary.py').read_text()
        self.assertIn("'gene_branch_loss_reduction': 'sum'", wrapper)
        self.assertIn("'compound_loss_weight': 1.0, 'gene_loss_weight': 1.0", wrapper)

if __name__ == '__main__': unittest.main()
