"""Evaluate frozen CGP or molecular-profile models with an explicit, shared replicate aggregation protocol."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from cgp_align.checkpoint_io import load_checkpoint
from cgp_align.matched_retrieval import evaluate
from cgp_common import CGPAlignModel
from prepare_matched_benchmark import sha, write, METHODS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--method', choices=('cgp',) + METHODS, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--aggregation', choices=('encode_mean', 'mean_encoded'), default='encode_mean',
                   help='Historical default retained; adopted protocol uses mean_encoded explicitly.')
    p.add_argument('--replicate-data', type=Path)
    p.add_argument('--artifacts', type=Path, help='Released final artifacts; replaces archived CGP input paths.')
    a = p.parse_args()
    audit = json.loads((a.prepared / 'audit.json').read_text())
    profiles = np.load(a.prepared / 'profile_features.npy', mmap_mode='r')
    split = json.loads((a.prepared / 'morgan/splits_intrinsic_entity.json').read_text())
    rows = np.asarray(split['compound']['test'], dtype=np.int64)
    table = pd.read_parquet(a.prepared / 'morgan/compounds.parquet')
    assert table.iloc[rows].compound_id.astype(str).tolist() == audit['test_ids']
    assert sha(a.prepared / 'negative_candidates.npy') == audit['negative_candidates_sha256']
    assert sha(a.prepared / 'profile_features.npy') == audit['entity_mean_sha256']
    payload = load_checkpoint(a.checkpoint)
    cfg = payload['config']
    assert int(cfg['seed']) == audit['seed']
    device = torch.device(a.device)
    if a.method == 'cgp':
        from eval_raw3180_branch_retrieval_from_checkpoints import load_model_and_data
        assert cfg['profile_norm'] == 'none'
        overrides = None
        if a.artifacts is not None:
            if not (ROOT / 'RUNTIME_VERSION.json').exists():
                p.error('Use a staged final-architecture runtime with --artifacts')
            base = a.artifacts.resolve()
            overrides = dict(compound_data_dir=str(base / f'data/seed{audit["seed"]}/compound'),
                             gene_data_dir=str(base / f'data/seed{audit["seed"]}/gene'),
                             gene_protein_embedding_dir=str(base / 'protein'))
        source = Path(overrides['compound_data_dir'] if overrides else cfg['compound_data_dir'])
        assert sha(source / 'compound_mocop_entities.parquet') == audit['entity_table_sha256']
        assert sha(source / 'splits_compound_mocop.json') == audit['split_sha256']
        model, data, payload, args = load_model_and_data(a.checkpoint, device, 512,
            data_overrides=argparse.Namespace(**overrides) if overrides else None)
        assert np.array_equal(data['compound']['test_entities'], rows)
    else:
        assert sha(a.prepared / a.method / 'compound_structure_features.npy') == audit['methods'][a.method]['feature_sha256']
        model = CGPAlignModel(cfg['structure_dim'], cfg['protein_dim'], cfg['profile_dim'],
                             embed_dim=cfg['embed_dim'], dropout=cfg['dropout'],
                             feature_dim_by_source=cfg.get('feature_dim_by_source')).to(device)
        model.load_state_dict(payload['model_state_dict'], strict=True)
        structure = np.load(a.prepared / a.method / 'compound_structure_features.npy', mmap_mode='r')
    model.eval()
    zc, zp = [], []
    with torch.no_grad():
        for start in range(0, len(rows), 512):
            ids = rows[start:start + 512]
            if a.method == 'cgp':
                c = model.encode_compound_graphs(data['compound']['graph_store'].get_many(ids.tolist()), device)
            else:
                c = model.encode_compound(torch.as_tensor(np.array(structure[ids]), device=device))
            profile = model.encode_profile(torch.as_tensor(np.array(profiles[ids]), device=device),
                                            torch.zeros(len(ids), device=device, dtype=torch.long))
            zc.append(c.float().cpu().numpy()); zp.append(profile.float().cpu().numpy())
    zc, zp = np.concatenate(zc), np.concatenate(zp)
    aggregation_provenance = {}
    if a.aggregation == 'mean_encoded':
        if a.replicate_data is None:
            raise ValueError('--replicate-data is required for mean_encoded')
        from cgp_align.profile_aggregation import encode_replicate_mean
        source = a.replicate_data
        assert sha(source / 'compound_mocop_entities.parquet') == audit['entity_table_sha256']
        assert sha(source / 'splits_compound_mocop.json') == audit['split_sha256']
        feature_hash = sha(source / 'compound_mocop_replicate_features.npy')
        assert feature_hash == audit['fit_audit']['output_sha256']
        reps = pd.read_parquet(source / 'compound_mocop_replicates.parquet')
        raw = np.load(source / 'compound_mocop_replicate_features.npy', mmap_mode='r')
        lookup = np.full(len(table), -1, dtype=np.int64)
        lookup[rows] = np.arange(len(rows))
        selected = reps.loc[reps.entity_index.isin(rows)]
        owners = lookup[selected.entity_index.to_numpy(dtype=np.int64)]
        feature_rows = selected.feature_index.to_numpy(dtype=np.int64)
        def encode(batch):
            with torch.no_grad():
                return model.encode_profile(torch.as_tensor(batch, device=device),
                    torch.zeros(len(batch), device=device, dtype=torch.long)).float().cpu().numpy()
        zp, counts = encode_replicate_mean(raw, feature_rows, owners, len(rows), encode)
        aggregation_provenance = dict(replicate_features_sha256=feature_hash,
            replicate_table_sha256=sha(source / 'compound_mocop_replicates.parquet'),
            replicate_count=int(counts.sum()), min_replicates=int(counts.min()),
            max_replicates=int(counts.max()), statistics_fit_on_test=False)
    bank = np.load(a.prepared / 'negative_candidates.npy', mmap_mode='r')
    output = dict(seed=audit['seed'], method=a.method, protocol=('strict_entity_latent_mean_v1' if a.aggregation == 'mean_encoded' else 'strict_entity_mean_v1'),
                  aggregation=a.aggregation, aggregation_provenance=aggregation_provenance,
                  checkpoint=str(a.checkpoint), checkpoint_sha256=sha(a.checkpoint), checkpoint_epoch=payload['epoch'],
                  selection_metric=payload.get('selection_metric', cfg.get('selection_metric', 'val_hmean_Top10')),
                  test_count=len(rows), prepared_audit_sha256=sha(a.prepared / 'audit.json'),
                  negative_candidates_sha256=audit['negative_candidates_sha256'],
                  profile_definition=('Encode each strictly corrected replicate; arithmetic mean by compound; L2 normalize'
                      if a.aggregation == 'mean_encoded' else audit['profile_definition']), tie_policy='ascending gallery row index',
                  directions={})
    ranks = {}
    for direction, query, gallery in [('compound_to_profile', zc, zp), ('profile_to_compound', zp, zc)]:
        metrics, full, sampled = evaluate(query, gallery, bank, device=device)
        output['directions'][direction] = metrics
        ranks[direction + '_full'] = full
        for k, values in sampled.items(): ranks[direction + '_' + str(k)] = values
    a.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(a.output / 'query_ranks.npz', **ranks)
    np.savez_compressed(a.output / 'test_embeddings.npz', compound=zc, profile=zp, compound_ids=np.asarray(audit['test_ids']))
    write(a.output / 'metrics.json', output)
    print(json.dumps(output), flush=True)


if __name__ == '__main__': main()
