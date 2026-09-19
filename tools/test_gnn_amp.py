"""Regression check for message aggregation across autocast dtypes."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import torch
from cgp_gnn import GGNNCompoundEncoder, graph_from_smiles, batch_graphs, atom_feature_dim, bond_feature_dim

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    device = torch.device(a.device)
    torch.manual_seed(31)
    torch.set_num_threads(1)
    model = GGNNCompoundEncoder(atom_feature_dim(), bond_feature_dim(), hidden_dim=32,
                                embed_dim=16, num_layers=6, dropout=0.0).to(device)
    graphs = [graph_from_smiles(s) for s in ('CCO', 'CCN', 'c1ccccc1', '[Na+]')]
    tensors = batch_graphs(graphs, device)
    for enabled in (False, True):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=enabled,
                            dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
            output = model(*tensors)
            loss = (output.float() - torch.arange(16, device=device).float() / 16).square().mean()
        assert torch.isfinite(output).all() and torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) for g in grads)
        print('PASS', str(device), 'autocast=' + str(enabled), 'finite forward/backward')

if __name__ == '__main__': main()
