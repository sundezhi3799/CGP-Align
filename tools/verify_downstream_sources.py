"""Verify current figure tables against the archived September 20 run records."""
from pathlib import Path
import csv,json,math,statistics
ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'figures/downstream_final/source_data'
D=ROOT/'revision_candidates/downstream_20260920'
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-10,rel_tol=1e-8),(a,b)
def main():
    e=rows(D/'figure4/enrichment_runs.csv')
    assert len(e)==30
    for r in rows(S/'figure4b_target_transfer_lift50_main.csv'):
        close(r['mean_lift50'],statistics.mean(float(x['Mean_Lift@50']) for x in e if x['source']==r['source']))
    a=rows(D/'figure5/random_cgp_folds.csv')
    for r in rows(S/'figure5b_sr5_endpoint_auprc_heatmap.csv'):
        if r['display']=='CGP-Align':
            for metric in ['auprc','auroc']:close(r[metric],statistics.mean(float(x[metric]) for x in a if x['task']==r['task']))
    a=rows(D/'figure5/alerts_metrics.csv');assert len(a)==75
    for r in rows(S/'figure5d_sr5_expert_alert_complementarity_by_run.csv'):
        q=[x for x in a if x['seed']==r['independent_run'] and x['method']==r['method']];assert len(q)==5
        close(r['run_mean_auprc'],statistics.mean(float(x['auprc']) for x in q))
        close(r['run_mean_lift_at_top50'],statistics.mean(float(x['lift_at_top50']) for x in q))
    a=rows(S/'figure6_panel_b_functional_neighbour_enrichment_source.csv')
    q=[x for x in a if float(x['low_tanimoto'])==.2 and float(x['corr_threshold'])==.3 and int(x['topk'])==50]
    assert next(int(x['directed_hits']) for x in q if x['method']=='CGP-Align')==1965
    q=rows(S/'figure6_panel_c_cgp_vs_rdkit2d_source.csv');assert len(q)==187
    close(statistics.mean(float(x['CGP-Align'])-float(x['RDKit2D']) for x in q),.1567914438502674)
    s3=rows(S/'figureS3_run_metrics.csv');assert len(s3)==9
    for r in s3:
        if r['variant']=='Full model':p=ROOT/f"revision_candidates/final_architecture_20260920/figure3/strict_seed{r['seed']}_test_metrics.json"
        else:
            variant={'No source indicator':'no_source_indicator','No pretraining':'no_pretraining'}[r['variant']]
            p=next((D/'ablations'/variant/f"seed{r['seed']}").rglob('*test_metrics.json'))
        m=json.loads(p.read_text());close(r['hmean'],m['top10_100']['hmean']);close(r['epoch'],m['checkpoint_epoch'])
    audit=json.loads((D/'portable_evaluation_audit.json').read_text());assert audit['status']=='PASS'
    assert all(abs(d['values'][1]-d['values'][2])<=audit['pass_absolute_tolerance'] for d in audit['differences'])
    print('PASS: Figure 4 enrichment, Figure 5 metrics, Figure 6 query results, S3 checkpoints and packaged three-seed evaluation')
if __name__=='__main__':main()
