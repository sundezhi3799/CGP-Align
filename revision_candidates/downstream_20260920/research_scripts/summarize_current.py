from pathlib import Path
import pandas as pd,numpy as np,json
R=Path(__file__).resolve().parent;S=R/'plots/source_data'
out={}
for n in ['figure4b_target_transfer_lift50_main','figure4c_target_transfer_leave_one_source_out_classifier_main','figure5a_sr5_representation_mean_metrics','figure5c_sr5_scaffold_split_mean_metrics','figure5d_expert_alert_complementarity_summary']:
 out[n]=pd.read_csv(S/(n+'.csv')).to_dict('records')
u=pd.read_csv(R/'figure6/prism_u2os_single_cell_paired_stats.csv');out['u2os_paired']=u[(u.setting=='low_morgan_lt_0.20')&(u.active_threshold==-1)&(u.topk==50)&(u.baseline=='RDKit2D')&(u.metric=='top_active_fraction')].to_dict('records')
q=pd.read_csv(S/'figure6_panel_c_cgp_vs_rdkit2d_source.csv');out['u2os_counts']={'wins':int((q['CGP-Align']>q.RDKit2D).sum()),'ties':int((q['CGP-Align']==q.RDKit2D).sum()),'losses':int((q['CGP-Align']<q.RDKit2D).sum())}
a=pd.read_csv(R/'figure5/alerts_metrics.csv');a=a.groupby(['task','method'])[['auprc','lift_at_top50']].mean().unstack('method');rng=np.random.default_rng(20260920);ix=rng.integers(0,5,size=(20000,5));out['alerts_paired']={}
for metric in ['auprc','lift_at_top50']:
 d=(a[metric]['cgp_plus_alert_extratrees']-a[metric]['cgp_latent_extratrees']).to_numpy();out['alerts_paired'][metric]={'delta':float(d.mean()),'ci95':np.quantile(d[ix].mean(1),[.025,.975]).tolist(),'endpoint_deltas':d.tolist()}
h=pd.read_csv(R/'figure6/prism_high_confidence_hit_summary.csv');out['hits_columns']=list(h.columns);out['hits']=h[(h.low_tanimoto.isin([.2,.15]))&(h.min_response_corr==.3)&(h.topk==50)].to_dict('records') if 'min_response_corr' in h else h.head().to_dict('records')
for p in (R/'figure4').glob('*upset*.csv'):out[p.stem]=pd.read_csv(p).to_dict('records')
(R/'current_summary.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
