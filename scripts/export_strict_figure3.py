"""Recompute Figure 3A-E/H source records from strict frozen models; no plotting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import train_cgp_align_replicate as cgp
from cgp_align.checkpoint_io import load_checkpoint
from eval_cgp_align_crossmodal_bridge import namespace_from_config
from strict_pca_helpers import encode_stage, l2_normalize


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def random_hit(positives, gallery, k):
    m = np.asarray(positives, dtype=np.float64)
    n = np.asarray(gallery, dtype=np.float64)
    miss = np.ones_like(m)
    for j in range(k): miss *= np.maximum(n-m-j, 0)/(n-j)
    return float(np.mean(1-miss))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--strict-root', type=Path, required=True)
    p.add_argument('--matched-root', type=Path, required=True)
    p.add_argument('--encoded-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--joint-directory', default=None)
    p.add_argument('--metric-pattern', default='strict_seed{seed}_joint_test_metrics.json')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(a.device)
    all_runs, controls, branches, sources = [], [], [], []
    definitions = [('compound', 'compound_to_profile', 'Compound -> profile', 'C-P'),
                   ('compound', 'profile_to_compound', 'Profile -> compound', 'P-C'),
                   ('gene', 'gene_to_profile', 'Gene -> profile', 'G-P'),
                   ('gene', 'profile_to_gene', 'Profile -> gene', 'P-G')]
    for seed in (31, 37, 41):
        previous = json.loads((a.encoded_root/f'seed{seed}/cgp/metrics.json').read_text())
        checkpoint = Path(previous['checkpoint'])
        assert sha(checkpoint) == previous['checkpoint_sha256']
        payload = load_checkpoint(checkpoint)
        args = namespace_from_config(payload['config'])
        for key in ('init_compound_checkpoint', 'init_gene_checkpoint', 'init_orf_gene_checkpoint', 'init_crispr_gene_checkpoint'):
            value = payload['config'].get(key)
            setattr(args, key, Path(value) if value else None)
        args.smoke_test = False
        assert args.profile_norm == 'none'
        assert int(args.eval_max_replicates_per_entity) == 0
        cgp.set_seed(seed)
        data = cgp.build_data(args)
        joint = a.joint_directory or ('joint_ampfix01' if seed == 31 else 'joint')
        metric_paths = list((a.strict_root/f'seed{seed}'/joint).rglob(a.metric_pattern.format(seed=seed)))
        assert len(metric_paths) == 1, metric_paths
        metrics = json.loads(metric_paths[0].read_text())
        write(a.output/f'strict_seed{seed}_test_metrics.json', metrics)
        positive_arrays = {}
        provenance = dict(seed=seed, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                          epoch=payload['epoch'], metrics_sha256=sha(metric_paths[0]), data={})
        for kind in ('compound', 'gene'):
            d = data[kind]; rows = d['test_entities']
            counts = np.array([len(d['entity_to_reps'][i]) for i in rows], dtype=np.int64)
            codes = d['codes'][rows]
            # Aggregate duplicate entity codes exactly as the evaluation gallery does.
            count_map = {int(c): int(counts[codes == c].sum()) for c in np.unique(codes)}
            entity_map = {int(c): int((codes == c).sum()) for c in np.unique(codes)}
            ep = np.array([count_map[int(c)] for c in codes])
            pe = np.repeat([entity_map[int(c)] for c in codes], counts)
            positive_arrays[kind] = (ep, pe)
            assert len(rows) == metrics[kind]['num_entities']
            assert int(counts.sum()) == metrics[kind]['num_profile_replicates']
            pd.DataFrame(dict(entity_index=rows, entity_code=codes, replicate_count=counts,
                              positive_profile_count=ep)).to_csv(a.output/f'seed{seed}_{kind}_test_counts.csv', index=False)
            provenance['data'][kind] = dict(entities=len(rows), replicates=int(counts.sum()),
                                            positive_counts_sha256=sha(a.output/f'seed{seed}_{kind}_test_counts.csv'))
        sample_values, sample_random = {1: [], 10: []}, {1: [], 10: []}
        for kind, direction, label, short in definitions:
            forward = direction.endswith('_to_profile')
            positives = positive_arrays[kind][0 if forward else 1]
            sampled = metrics[kind]['mocop_protocol_sampled'][direction]['1:100']
            full = metrics[kind]['full_gallery'][direction]
            assert len(positives) == sampled['num_queries']
            assert abs(float(positives.mean())-sampled['positive_count_mean']) < 1e-10
            route = 'Compound-profile' if kind == 'compound' else 'Gene-profile'
            all_runs.append(dict(setting='CGP-Align', independent_run=seed, direction=label,
                                 short=short, top10=sampled['Top10_accuracy_mean'], route=route))
            for k in (1, 10):
                observed = sampled[f'Top{k}_accuracy_mean']
                expected = random_hit(positives, positives+100, k)
                sample_values[k].append(observed); sample_random[k].append(expected)
                controls.append(dict(independent_run=seed, direction=label.replace(' -> ', ' to '), short=short,
                    branch=route.lower(), candidate_setting='1:100 sampled', metric=f'Top{k}',
                    observed=observed, random_ranking=expected, fold_enrichment=observed/expected,
                    positive_count_mean=float(positives.mean()), negative_count=100,
                    num_queries=len(positives), num_gallery=sampled['num_gallery']))
            for k in (1, 5, 10):
                expected = random_hit(positives, full['num_gallery'], k)
                observed = full[f'Recall@{k}']
                controls.append(dict(independent_run=seed, direction=label.replace(' -> ', ' to '), short=short,
                    branch=route.lower(), candidate_setting='full gallery', metric=f'Recall@{k}',
                    observed=observed, random_ranking=expected, fold_enrichment=observed/expected,
                    positive_count_mean=float(positives.mean()), negative_count=full['num_gallery']-float(positives.mean()),
                    num_queries=len(positives), num_gallery=full['num_gallery']))
        for k in (1, 10):
            for summary in ('Mean', 'HMean'):
                aggregate = lambda v: float(np.mean(v)) if summary == 'Mean' else float(len(v)/np.sum(1/np.array(v)))
                branches.append(dict(seed=seed, metric=f'{summary} Top-{k}', value=aggregate(sample_values[k]),
                                     random=aggregate(sample_random[k]), summary_type=summary))
        sources.append(provenance)
        if seed == 41:
            print('Exporting seed41 pre-joint and joint PCA inputs', flush=True)
            # Reconstruct the actual pre-joint initialization: pretrained entity encoders,
            # freshly initialized shared profile encoder, with the training seed.
            model, init = cgp.build_model(args, data, device)
            branch_provenance = {}
            for key in ('init_compound_checkpoint', 'init_orf_gene_checkpoint', 'init_crispr_gene_checkpoint'):
                path = getattr(args, key); assert path is not None
                branch_provenance[key] = dict(path=str(path), sha256=sha(path), epoch=load_checkpoint(path)['epoch'])
            c_rows = np.sort(np.random.default_rng(41).choice(data['compound']['test_entities'], 1200, replace=False))
            g_rows = np.sort(np.random.default_rng(58).choice(data['gene']['test_entities'], 1200, replace=False))
            pca_frames, diagnostics, reduction = [], [], {}
            for stage, label in [('branch_pretrained', 'Before joint training'), ('joint_aligned', 'After joint training')]:
                if stage == 'joint_aligned': model.load_state_dict(payload['model_state_dict'], strict=True)
                model.eval()
                x, meta, diag = encode_stage(model, data, args, c_rows, g_rows, device, 512)
                reducer = PCA(n_components=2, svd_solver='full').fit(x)
                coords = reducer.transform(x)
                meta['stage_key'] = stage; meta['stage_label'] = label
                meta['pca1'] = coords[:, 0]; meta['pca2'] = coords[:, 1]
                pca_frames.append(meta)
                diagnostics.append(dict(stage=stage, **diag))
                reduction[stage] = dict(scope='stage-shared', explained_variance_ratio=reducer.explained_variance_ratio_.tolist())
                np.savez_compressed(a.output/f'{stage}_pca_inputs.npz', embeddings=x, compound_rows=c_rows, gene_rows=g_rows)
            pd.concat(pca_frames).to_csv(a.output/'figure2ab_four_branch_main_pca_source_data.csv', index=False)
            write(a.output/'figure2ab_four_branch_main_pca_metadata.json', dict(
                checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), checkpoint_epoch=payload['epoch'],
                checkpoint_best_score=payload.get('best_score'), seed=41, split='test', num_compounds=1200,
                num_genes=1200, pre_joint_definition='Pretrained entity encoders and seed41-initialized shared profile encoder',
                branch_pretrained_checkpoints=branch_provenance, init_summaries=init,
                reduction_info=reduction, diagnostics=diagnostics))
            del model
        del data, payload
    pd.DataFrame(all_runs).to_csv(a.output/'figure2c_direction_top10_run_values.csv', index=False)
    control = pd.DataFrame(controls)
    control.to_csv(a.output/'figureS5_random_ranking_negative_control.csv', index=False)
    random = control[(control.candidate_setting == '1:100 sampled') & (control.metric == 'Top10')]
    random = random.groupby('direction', sort=False).random_ranking.mean().reset_index()
    random.columns = ['direction', 'random_ranking_top10']
    random.direction = random.direction.str.replace(' to ', ' -> ', regex=False)
    random.to_csv(a.output/'figure2c_random_ranking_top10.csv', index=False)
    branch = pd.DataFrame(branches)
    branch.to_csv(a.output/'branch_summary_run_values.csv', index=False)
    summary = branch.groupby(['metric', 'summary_type'], sort=False).agg(
        mean=('value', 'mean'), sd=('value', 'std'), random_ranking_mean=('random', 'mean'), random_ranking_sd=('random', 'std')).reset_index()
    summary.insert(0, 'setting', 'CGP-Align')
    summary.to_csv(a.output/'figure2d_branch_summary_mean_sd.csv', index=False)
    write(a.output/'strict_sources.json', sources)
    print('Exporting H', flush=True)
    prepared = a.matched_root/'seed41/data'
    audit = json.loads((prepared/'audit.json').read_text())
    assert sha(prepared/'profile_features.npy') == audit['entity_mean_sha256']
    assert sha(prepared/'morgan/compound_structure_features.npy') == audit['methods']['morgan']['feature_sha256']
    rows = np.array(json.loads((prepared/'morgan/splits_intrinsic_entity.json').read_text())['compound']['test'])
    emb = np.load(a.encoded_root/'seed41/cgp/test_embeddings.npz')
    assert emb['compound_ids'].astype(str).tolist() == audit['test_ids']
    raw = l2_normalize(np.array(np.load(prepared/'profile_features.npy', mmap_mode='r')[rows]))
    structure = l2_normalize(np.array(np.load(prepared/'morgan/compound_structure_features.npy', mmap_mode='r')[rows]))
    vectors = {'Structure feature baseline': (structure, structure),
               'CGP-Align entity': (emb['compound'], emb['compound']),
               'CGP entity -> profile anchor': (emb['compound'], emb['profile'])}
    n = len(rows); positives = np.empty((n,10), dtype=np.int32)
    def scores(q, g, start, end):
        result = (torch.as_tensor(q[start:end], device=device) @ torch.as_tensor(g, device=device).T).cpu().numpy()
        result[np.arange(end-start), np.arange(start,end)] = -np.inf
        return result
    for start in range(0,n,256):
        end=min(start+256,n); sim=scores(raw,raw,start,end)
        positives[start:end] = np.argsort(-sim,axis=1,kind='stable')[:,:10]
    records, per_query, saved = [], [], {'positive_indices': positives}
    for method,(q,g) in vectors.items():
        hits=[]; best=[]; top=np.empty((n,50),dtype=np.int32)
        for start in range(0,n,256):
            end=min(start+256,n); sim=scores(q,g,start,end)
            order=np.argsort(-sim,axis=1,kind='stable');top[start:end]=order[:,:50]
            inverse=np.empty_like(order);np.put_along_axis(inverse,order,np.arange(n)[None,:],axis=1)
            ranks=np.take_along_axis(inverse,positives[start:end],axis=1)+1
            hits.extend((ranks<=50).sum(axis=1).tolist());best.extend(ranks.min(axis=1).tolist())
        hit=np.asarray(hits); saved[method]=top
        record=dict(scope='compound_to_compound', method=method,k=50,n_query=n,n_gallery=n-1,
                    hit_at_k=float((hit>0).mean()), recall_at_k=float((hit/10).mean()),
                    precision_enrichment=float((hit/50).mean()/(10/(n-1))),mean_best_positive_rank=float(np.mean(best)))
        records.append(record)
        per_query.extend(dict(compound_id=str(i),method=method,hits_at_50=int(h),best_positive_rank=int(r))
                         for i,h,r in zip(emb['compound_ids'],hits,best))
        print(record,flush=True)
    pd.DataFrame(records).to_csv(a.output/'hidden_phenotype_neighbour_recovery_compact.csv',index=False)
    pd.DataFrame(records).to_csv(a.output/'figure3h_hidden_phenotype_neighbour_recovery.csv',index=False)
    pd.DataFrame(per_query).to_csv(a.output/'hidden_query_metrics.csv',index=False)
    np.savez_compressed(a.output/'hidden_rank_indices.npz',**saved)
    write(a.output/'hidden_input_metadata.json',dict(checkpoint=sources[-1]['checkpoint'],checkpoint_sha256=sources[-1]['checkpoint_sha256'],
        checkpoint_epoch=sources[-1]['epoch'],split='test',compound_entities=n,seed=41,entity_mean_sha256=audit['entity_mean_sha256'],
        profile_latent_definition='L2-normalized mean of individually encoded corrected replicates',
        structure_features_sha256=audit['methods']['morgan']['feature_sha256']))
    write(a.output/'hidden_analysis_metadata.json',dict(positive_topn=10,eval_ks=[50],seed=41,
        positive_definition='Ten nearest other compounds by cosine of entity-mean strict corrected profiles',
        tie_policy='ascending gallery index',n_query=n,n_gallery=n-1))
    write(a.output/'status.json',dict(status='complete',scope='Figure3 A-E/H',seed_representative=41))


if __name__ == '__main__': main()
