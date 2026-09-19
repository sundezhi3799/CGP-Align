"""Independently check strict Figure 3 saved sources; no model inference."""
import csv,json,math,statistics
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'revision_candidates/figure3_strict_20260920'
FIG=ROOT/'figures/figure3'
def rows(p):
 with p.open(newline='') as f: return list(csv.DictReader(f))
def close(a,b):
 assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-12),(a,b)
controls=rows(DATA/'figureS5_random_ranking_negative_control.csv')
for seed in (31,37,41):
 metrics=json.loads((DATA/f'strict_seed{seed}_test_metrics.json').read_text())
 for kind,prefix in [('compound','C'),('gene','G')]:
  counts=rows(DATA/f'seed{seed}_{kind}_test_counts.csv')
  entities=Counter(r['entity_code'] for r in counts)
  forward=Counter(int(r['positive_profile_count']) for r in counts); reverse=Counter()
  for r in counts: reverse[entities[r['entity_code']]]+=int(r['replicate_count'])
  for r in controls:
   if int(r['independent_run'])!=seed or prefix not in r['short']: continue
   is_forward=r['short'].endswith('-P'); distribution=forward if is_forward else reverse
   sampled=r['candidate_setting']=='1:100 sampled'
   k=int(r['metric'][3:]) if sampled else int(r['metric'].split('@')[-1])
   expected=0
   for positive,frequency in distribution.items():
    gallery=positive+100 if sampled else int(r['num_gallery'])
    expected+=frequency*(1-math.comb(gallery-positive,k)/math.comb(gallery,k))
   expected/=sum(distribution.values())
   close(expected,r['random_ranking']); close(float(r['observed'])/expected,r['fold_enrichment'])
   direction=f'{kind}_to_profile' if is_forward else f'profile_to_{kind}'
   observed=metrics[kind]['mocop_protocol_sampled'][direction]['1:100'][f'Top{k}_accuracy_mean'] if sampled else metrics[kind]['full_gallery'][direction][f'Recall@{k}']
   close(observed,r['observed'])
runs=rows(DATA/'branch_summary_run_values.csv')
for r in runs:
 selected=[x for x in controls if x['independent_run']==r['seed'] and x['candidate_setting']=='1:100 sampled' and x['metric']=='Top'+r['metric'].split('-')[-1]]
 assert len(selected)==4
 aggregate=statistics.mean if r['summary_type']=='Mean' else statistics.harmonic_mean
 close(aggregate(float(x['observed']) for x in selected),r['value'])
 close(aggregate(float(x['random_ranking']) for x in selected),r['random'])
for r in rows(FIG/'source_data/figure2d_branch_summary_mean_sd.csv'):
 selected=[x for x in runs if x['metric']==r['metric']]; assert len(selected)==3
 close(statistics.mean(float(x['value']) for x in selected),r['mean'])
 close(statistics.stdev(float(x['value']) for x in selected),r['sd'])
queries=rows(DATA/'hidden_query_metrics.csv')
for r in rows(FIG/'source_data/figure3h_hidden_phenotype_neighbour_recovery.csv'):
 q=[x for x in queries if x['method']==r['method']]
 assert len(q)==len({x['compound_id'] for x in q})==11578
 assert (int(r['n_gallery']),int(r['k']))==(11577,50)
 hits=[int(x['hits_at_50']) for x in q]; assert all(0<=h<=10 for h in hits)
 close(statistics.mean(hits)/10,r['recall_at_k'])
 close(statistics.mean(hits)/50/(10/11577),r['precision_enrichment'])
 close(statistics.mean(h>0 for h in hits),r['hit_at_k'])
 close(statistics.mean(int(x['best_positive_rank']) for x in q),r['mean_best_positive_rank'])
 assert all((int(x['best_positive_rank'])<=50)==(int(x['hits_at_50'])>0) for x in q)
inputs=json.loads((FIG/'provenance/hidden_input_metadata.json').read_text())
pca=json.loads((FIG/'source_data/figure2ab_four_branch_main_pca_metadata.json').read_text())
assert inputs['checkpoint_epoch']==pca['checkpoint_epoch']==60
assert inputs['checkpoint_sha256']==pca['checkpoint_sha256']
assert inputs['split']==pca['split']=='test'
assert pca['num_compounds']==pca['num_genes']==1200
for p in (FIG/'source_data').iterdir():
 if (DATA/p.name).exists(): assert p.read_bytes()==(DATA/p.name).read_bytes(),p
print('PASS: strict Figure 3 C-E random controls, raw metrics, summaries, H query aggregates and A/B/H provenance')
