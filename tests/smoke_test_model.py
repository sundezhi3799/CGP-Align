"""Exercise the original four-branch model and loss, using synthetic data only."""
from pathlib import Path
import io
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import torch
import train_cgp_align_replicate as cgp
from cgp_gnn import graph_from_smiles

def main():
    torch.set_num_threads(1)
    torch.manual_seed(31)
    kwargs = dict(profile_dim=3180, protein_dim=16, embed_dim=256,
                  gnn_hidden_dim=32, gnn_layers=2, gene_hidden_dims=(32,),
                  profile_hidden_dims=(32,), modality_context_dim=0, dropout=0.0,
                  profile_source_adapters=False, profile_source_embedding=True,
                  profile_source_dropout=0.0, num_profile_sources=3,
                  profile_input_layernorm=False, profile_feature_dropout=0.0,
                  profile_mlp_norm=False, split_gene_modality_branches=True)
    model = cgp.ReplicateCGPAlign(**kwargs)
    graphs = [graph_from_smiles(s) for s in ('CCO', 'CCN', 'c1ccccc1', 'CC(=O)O')]
    compounds = model.encode_compound_graphs(graphs, torch.device('cpu'))
    modalities = torch.tensor([0, 0, 1, 1])
    protein = torch.randn(4, 16)
    genes = model.encode_gene(protein, modalities)
    profile_input = torch.randn(8, 3180)
    profiles = model.encode_profile(profile_input, torch.zeros(8, dtype=torch.long))
    codes = torch.arange(4)
    gallery_codes = codes.repeat_interleave(2)
    loss = cgp.multipositive_contrastive_loss(compounds, profiles, codes, gallery_codes, 0.07)
    # Exercise ORF and CRISPR branches with their corresponding profile source IDs.
    gene_profiles = model.encode_profile(torch.randn(4, 3180), modalities + 1)
    loss = loss + cgp.multipositive_contrastive_loss(genes, gene_profiles, codes, codes, 0.07)
    assert torch.isfinite(loss)
    for values in (compounds, genes, profiles):
        assert values.shape[1] == 256 and torch.isfinite(values).all()
        assert torch.allclose(values.norm(dim=1), torch.ones(len(values)), atol=1e-5)
    loss.backward()
    for branch in (model.compound_encoder, model.gene_encoder, model.crispr_gene_encoder, model.profile_encoder):
        gradients = [p.grad for p in branch.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g).item() for g in gradients)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = cgp.ReplicateCGPAlign(**kwargs)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    model.eval(); restored.eval()
    assert torch.allclose(model.encode_gene(protein, modalities), restored.encode_gene(protein, modalities))
    print(json.dumps({'status': 'PASS', 'device': 'cpu', 'synthetic_loss': float(loss.detach()),
                      'checks': ['four branches', 'unit norm', 'multi-positive loss', 'finite nonzero gradients', 'checkpoint round-trip'],
                      'scope': 'synthetic implementation check; reduced hidden sizes; not paper performance'}, indent=2))

if __name__ == '__main__':
    main()
