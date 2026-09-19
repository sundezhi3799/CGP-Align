"""Evaluate frozen CGP or molecular-profile models on identical entity-mean targets."""
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
        source = Path(cfg['compound_data_dir'])
        assert sha(source / 'compound_mocop_entities.parquet') == audit['entity_table_sha256']
        assert sha(source / 'splits_compound_mocop.json') == audit['split_sha256']
        model, data, payload, args = load_model_and_data(a.checkpoint, device, 512)
        assert np.array_equal(data['compound']['test_entities'], rows)
    else:
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
    bank = np.load(a.prepared / 'negative_candidates.npy', mmap_mode='r')
    output = dict(seed=audit['seed'], method=a.method, protocol='strict_entity_mean_v1',
                  checkpoint=str(a.checkpoint), checkpoint_sha256=sha(a.checkpoint), checkpoint_epoch=payload['epoch'],
                  selection_metric=payload.get('selection_metric', cfg.get('selection_metric', 'val_hmean_Top10')),
                  test_count=len(rows), prepared_audit_sha256=sha(a.prepared / 'audit.json'),
                  negative_candidates_sha256=audit['negative_candidates_sha256'],
                  profile_definition=audit['profile_definition'], tie_policy='ascending gallery row index',
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
