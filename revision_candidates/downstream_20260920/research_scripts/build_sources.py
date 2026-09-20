from pathlib import Path
import json,shutil,math
import numpy as np,pandas as pd
R=Path('reproducibility_audits/downstream_release_20260920');B=R/'plots';S=B/'source_data';S.mkdir(parents=True,exist_ok=True);(B/'scripts').mkdir(exist_ok=True)
OLD=Path('public_repository/CGP-Align/paper_snapshot/source_data')
for pat in ['figure4*.csv','figure5*.csv','figure6*.csv','figureS2*.csv']:
 for p in OLD.glob(pat):shutil.copy2(p,S/p.name)
for p in (R/'figure4').glob('figure4*.csv'):shutil.copy2(p,S/p.name)
names={'dgidb':'DGIdb','drugrep':'DrugRep','hetionet':'Hetionet','openbiolink':'OpenBioLink','pharmebinet':'PharMeBINet'};order={s:i+2 for i,s in enumerate(names)}
e=pd.read_csv(R/'figure4/enrichment_runs.csv');d=e.groupby('source',as_index=False).agg(mean_lift50=('Mean_Lift@50','mean'),mapped_edges_mean=('mapped_edges','mean'));d['name']=d.source.map(names);d['source_order']=d.source.map(order);d['above_random_ranking']=d.mean_lift50>1;d.to_csv(S/'figure4b_target_transfer_lift50_main.csv',index=False)
a=pd.concat([pd.read_csv(R/f'figure4/{g}/gene_union_latent_link_classifier_summary.csv').assign(graph=g) for g in ['broad','strict']]);a=a[a.fold.eq('test')].copy();a['source']=a.split_mode.str.replace('source_holdout_','');d=a.groupby('source',as_index=False)[['auprc','auroc','cosine_auprc','cosine_auroc']].mean().rename(columns={'auprc':'leave_one_source_out_auprc','auroc':'leave_one_source_out_auroc'});d['name']=d.source.map(names);d['source_order']=d.source.map(order)
for m in ['auprc','auroc']:d['delta_'+m]=d['leave_one_source_out_'+m]-d['cosine_'+m]
d.to_csv(S/'figure4c_target_transfer_leave_one_source_out_classifier_main.csv',index=False)
gain=pd.read_csv(OLD/'figure4d_target_transfer_metric_gain_main.csv');gain['gain']=[float(d.set_index('source').loc[r.source,'delta_'+('auprc' if r.metric=='AUPRC gain' else 'auroc')]) for r in gain.itertuples()];gain.to_csv(S/'figure4d_target_transfer_metric_gain_main.csv',index=False)
# Current CGP rows, unchanged matched non-CGP comparator tables.
f=pd.read_csv(R/'figure5/random_cgp_folds.csv');end=f.groupby('task',as_index=False).agg(auprc=('auprc','mean'),auroc=('auroc','mean'));heat=pd.read_csv(OLD/'figure5b_sr5_endpoint_auprc_heatmap.csv')
for i,row in heat[heat.display.eq('CGP-Align')].iterrows():
 x=end.set_index('task').loc[row.task];heat.loc[i,['auprc','auroc']]=[x.auprc,x.auroc]
heat.to_csv(S/'figure5b_sr5_endpoint_auprc_heatmap.csv',index=False)
mean=heat.groupby('display',as_index=False).agg(auprc_mean=('auprc','mean'),auroc_mean=('auroc','mean'),auprc_sd=('auprc','std'),auroc_sd=('auroc','std'),n_task=('task','nunique'));mean['auprc_se']=mean.auprc_sd/np.sqrt(mean.n_task);mean['auroc_se']=mean.auroc_sd/np.sqrt(mean.n_task);mean.to_csv(S/'figure5a_sr5_representation_mean_metrics.csv',index=False)
f=pd.read_csv(R/'figure5/scaffold_cgp_folds.csv');sc=pd.read_csv(OLD/'figure5c_sr5_scaffold_split_endpoint_dots.csv');end=f.groupby('task').agg(auprc_mean=('auprc','mean'),auprc_sd=('auprc','std'),auroc_mean=('auroc','mean'),auroc_sd=('auroc','std'))
for i,row in sc[sc.display.eq('CGP-Align')].iterrows():sc.loc[i,end.columns]=end.loc[row.task].values
sc.to_csv(S/'figure5c_sr5_scaffold_split_endpoint_dots.csv',index=False)
sm=sc.groupby(['display','representation'],as_index=False).agg(n_task=('task','nunique'),mean_auprc=('auprc_mean','mean'),sd_endpoint_auprc=('auprc_mean','std'),mean_auroc=('auroc_mean','mean'),sd_endpoint_auroc=('auroc_mean','std'));sm.to_csv(S/'figure5c_sr5_scaffold_split_mean_metrics.csv',index=False)
a=pd.read_csv(R/'figure5/alerts_metrics.csv');a=a.groupby(['seed','method'],as_index=False).agg(tasks=('task','nunique'),run_mean_auprc=('auprc','mean'),run_mean_lift_at_top50=('lift_at_top50','mean')).rename(columns={'seed':'independent_run'});a['method_label']=a.method.map({'structure_alert_extratrees':'Expert alerts','cgp_latent_extratrees':'CGP-Align latent','cgp_plus_alert_extratrees':'CGP + alerts'});a.to_csv(S/'figure5d_sr5_expert_alert_complementarity_by_run.csv',index=False)
a.groupby('method_label',as_index=False).agg(n_independent_runs=('independent_run','nunique'),tasks=('tasks','first'),mean_auprc=('run_mean_auprc','mean'),sem_auprc=('run_mean_auprc',lambda x:x.std(ddof=1)/np.sqrt(len(x))),mean_lift_at_top50=('run_mean_lift_at_top50','mean'),sem_lift_at_top50=('run_mean_lift_at_top50',lambda x:x.std(ddof=1)/np.sqrt(len(x)))).to_csv(S/'figure5d_expert_alert_complementarity_summary.csv',index=False)
# PRISM response matrices and metadata are fixed; only CGP similarity changed.
f=R/'figure6';mapping={'prism_high_confidence_hit_summary.csv':'figure6_panel_b_functional_neighbour_enrichment_source.csv','prism_continuous_metric_summary.csv':'figure6_panel_d_pancancer_source.csv','mechanism/pair_gene_mechanism_summary.csv':'figure6_panel_e_gene_mechanism_summary_source.csv','mechanism/pair_mechanism_category_summary.csv':'figure6_panel_e_mechanism_category_summary_source.csv','prism_high_confidence_cgp_cases_tan020_corr030_top50.csv':'figure6_panel_f_high_confidence_cases_source.csv'}
for src,dest in mapping.items():shutil.copy2(f/src,S/dest)
a=pd.read_csv(f/'prism_u2os_single_cell_query_metrics.csv');a=a[np.isclose(a.low_tanimoto,.2)&np.isclose(a.active_threshold,-1)&a.topk.eq(50)&a.method.isin(['CGP-Align','RDKit2D'])];q=a.pivot(index='query_idx',columns='method',values='top_active_fraction').reset_index();q['cgp_better']=q['CGP-Align']>q.RDKit2D;q['same']=q['CGP-Align']==q.RDKit2D;q['n_query']=1;q.to_csv(S/'figure6_panel_c_cgp_vs_rdkit2d_source.csv',index=False)
meta=pd.read_csv(f/'prism_cgp_overlap_compounds.csv');ki={k:i for i,k in enumerate(meta.inchikey)};mech=pd.read_csv(f/'mechanism/pair_gene_mechanism_metrics.csv').query('method == "CGP-Align"').set_index(['pair_a','pair_b']);case=pd.read_csv(f/'prism_high_confidence_cgp_cases_tan020_corr030_top50.csv');rows=[]
for row in case.to_dict('records'):
 i,j=sorted([ki[row['query_inchikey']],ki[row['retrieved_inchikey']]]);m=mech.loc[(i,j)].to_dict();m.update(row);rows.append(m)
case=pd.DataFrame(rows).sort_values(['shared_top_gene_count','prism_response_corr'],ascending=False);case.to_csv(S/'figure6_panel_f_mechanism_cases_source.csv',index=False)
# S2 displays seed 41; all seeds stay in the audit source records.
for src,dest in [('composition','figureS2a_dataset_composition'),('splits','figureS2c_split_composition'),('r2','figureS2d_post_correction_r2')]:
 d=pd.read_csv(R/f's2/{src}.csv');d=d[d.seed.eq(41)].drop(columns='seed');d.to_csv(S/(dest+'.csv'),index=False)
pd.read_csv(R/'s2/replicate_distribution.csv').rename(columns={'replicate_count':'num_replicates'}).to_csv(S/'figureS2b_entity_replicate_distribution.csv',index=False)
pd.read_csv(R/'s2/overlap.csv').rename(columns={'entity_overlap':'overlap_entities'}).to_csv(S/'figureS2c_split_overlap_check.csv',index=False)
# S3 paired seed-level results, including the unchanged primary model references.
rows=[]
for variant in ['Full model','No source indicator','No pretraining']:
 for seed in [31,37,41]:
  if variant=='Full model':p=Path('reproducibility_audits/final_architecture_20260920')/f'final_arch_seed{seed}_joint_test_metrics.json'
  else:
   key='no_source_indicator' if variant=='No source indicator' else 'no_pretraining';p=next((R/'ablations'/key/f'seed{seed}').rglob('*test_metrics.json'))
  m=json.loads(p.read_text());row=dict(variant=variant,seed=seed,epoch=m['checkpoint_epoch'],**m['top10_100']);rows.append(row)
pd.DataFrame(rows).to_csv(S/'figureS3_run_metrics.csv',index=False)
print('source tables prepared')

a=json.loads((R/'s2/preprocessing_audits.json').read_text());pd.DataFrame([{k:v for k,v in x.items() if k!='steps'} for x in a]).to_csv(S/'figureS2e_normalization_audit.csv',index=False)
