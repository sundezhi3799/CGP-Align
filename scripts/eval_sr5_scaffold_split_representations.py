import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')
except Exception as e:
    raise RuntimeError('RDKit is required for scaffold split') from e

SR5 = ['Endocrine Disruption_SR-ARE','Endocrine Disruption_SR-MMP','Endocrine Disruption_SR-HSE','Endocrine Disruption_SR-p53','Endocrine Disruption_SR-ATAD5']
REPS = ['CGP_sep_k32','RDKit2D','NYAN_latent','Morgan2048','AtomPair2048','Avalon1024']
DISP = {'CGP_sep_k32':'CGP-Align','RDKit2D':'RDKit2D','NYAN_latent':'NYAN','Morgan2048':'Morgan','AtomPair2048':'AtomPair','Avalon1024':'Avalon'}

def canon(s):
    m = Chem.MolFromSmiles(str(s or '').strip())
    return Chem.MolToSmiles(m, canonical=True) if m is not None else ''

def scaffold(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return 'invalid'
    sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    return sc if sc else s

def load_task(toxric_dir, task):
    raw = pd.read_csv(toxric_dir / f'{task}.csv')
    df = raw[['Canonical SMILES','Toxicity Value']].copy()
    df['smiles'] = df['Canonical SMILES'].map(canon)
    df['label'] = pd.to_numeric(df['Toxicity Value'], errors='coerce')
    df = df[(df.smiles!='') & df.label.isin([0,1])].copy()
    df['label'] = df.label.astype(int)
    df = df.drop_duplicates(['smiles','label']).reset_index(drop=True)
    bad = set(df.groupby('smiles').label.nunique().loc[lambda x: x>1].index)
    if bad: df = df[~df.smiles.isin(bad)].copy().reset_index(drop=True)
    df['scaffold'] = df.smiles.map(scaffold)
    return df

def smiles_index(json_path):
    smiles = json.loads(Path(json_path).read_text(encoding='utf-8'))
    idx = {}
    for i, s in enumerate(smiles):
        ss = canon(s)
        if ss and ss not in idx: idx[ss] = i
        if str(s) not in idx: idx[str(s)] = i
    return idx

def load_nyan(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    idx = {}
    for i, (s, ok) in enumerate(zip(z['smiles'], z['valid'])):
        if not bool(ok): continue
        ss = canon(str(s))
        if ss and ss not in idx: idx[ss] = i
    return idx, z['latents'].astype(np.float32)

def get_features(task, rep, smiles, main_dir, atad5_dir, main_idx, atad5_idx, nyan_idx, nyan_x):
    if rep == 'NYAN_latent':
        rows = [nyan_idx[s] for s in smiles]
        return nyan_x[np.asarray(rows, int)]
    if task == 'Endocrine Disruption_SR-ATAD5':
        base, idx = atad5_dir, atad5_idx
    else:
        base, idx = main_dir, main_idx
    rows = [idx[s] for s in smiles]
    arr = np.load(base / 'feature_cache' / f'{rep}.npy', mmap_mode='r')
    return np.asarray(arr[np.asarray(rows, int)], dtype=np.float32)

def make_splits(y, groups, n_splits, seed):
    if StratifiedGroupKFold is None:
        raise RuntimeError('StratifiedGroupKFold is unavailable in this sklearn version')
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups=groups))

def eval_one(x, y, groups, task, rep, n_splits, seed, n_estimators):
    out = []
    for fold, (tr, te) in enumerate(make_splits(y, groups, n_splits, seed)):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        clf = ExtraTreesClassifier(n_estimators=n_estimators, max_features='sqrt', min_samples_leaf=1,
                                   class_weight='balanced_subsample', random_state=seed, n_jobs=1)
        clf.fit(x[tr], y[tr])
        score = clf.predict_proba(x[te])[:,1]
        out.append(dict(task=task, representation=rep, display=DISP[rep], model='extratrees_scaffold', fold=fold,
                        n_train=len(tr), n_test=len(te), positive_train=int(y[tr].sum()), positive_test=int(y[te].sum()),
                        n_train_scaffolds=len(set(groups[tr])), n_test_scaffolds=len(set(groups[te])),
                        scaffold_overlap=len(set(groups[tr]) & set(groups[te])), auroc=roc_auc_score(y[te], score),
                        auprc=average_precision_score(y[te], score)))
    return out

def ci(vals, n_boot, seed):
    vals = np.asarray(vals, float); rng = np.random.default_rng(seed)
    boot = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return float(np.quantile(boot,.025)), float(np.quantile(boot,.975))

def stats(folds, n_boot, seed):
    endpoint = folds.groupby(['task','display','representation'], as_index=False).agg(n=('n_test','sum'), positive_test=('positive_test','sum'),
        scaffolds_test=('n_test_scaffolds','sum'), auprc_mean=('auprc','mean'), auprc_sd=('auprc','std'), auroc_mean=('auroc','mean'), auroc_sd=('auroc','std'))
    mean = endpoint.groupby(['display','representation'], as_index=False).agg(n_task=('task','nunique'), mean_auprc=('auprc_mean','mean'),
        sd_endpoint_auprc=('auprc_mean','std'), mean_auroc=('auroc_mean','mean'), sd_endpoint_auroc=('auroc_mean','std')).sort_values('mean_auprc', ascending=False)
    key=['task','fold']; cgp=folds[folds.representation.eq('CGP_sep_k32')][key+['auprc','auroc']].rename(columns={'auprc':'cgp_auprc','auroc':'cgp_auroc'})
    rows=[]
    for rep in [r for r in REPS if r!='CGP_sep_k32']:
        b=folds[folds.representation.eq(rep)][key+['auprc','auroc']].rename(columns={'auprc':'base_auprc','auroc':'base_auroc'})
        m=cgp.merge(b,on=key,how='inner')
        if m.empty: continue
        for metric in ['auprc','auroc']:
            d=(m[f'cgp_{metric}']-m[f'base_{metric}']).to_numpy(float)
            lo,hi=ci(d,n_boot,seed+len(rep)+len(metric))
            rows.append(dict(baseline=DISP[rep], metric=metric.upper(), n_paired_folds=len(d), mean_delta=float(d.mean()),
                             median_delta=float(np.median(d)), bootstrap_ci95_low=lo, bootstrap_ci95_high=hi,
                             cgp_wins=int((d>0).sum()), cgp_losses=int((d<0).sum())))
    return endpoint, mean, pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--toxric_dir',type=Path,default=Path('output/cgp_align/paper/raw_results/tox_np_mechanism/raw_download_probe/toxric/toxric_30_datasets'))
    ap.add_argument('--main_dir',type=Path,default=Path('output/cgp_align/paper/raw_results/tox_np_mechanism/toxric22_fixed_extratrees_representation_benchmark'))
    ap.add_argument('--atad5_dir',type=Path,default=Path('output/cgp_align/paper/raw_results/tox_np_mechanism/check_sr_atad5_extratrees_representation_benchmark'))
    ap.add_argument('--nyan_cache',type=Path,default=Path('output/cgp_align/paper/raw_results/toxric_auto_search/pheno5_official_nyan_checkpoint_et_f5/official_nyan_latents_cache.npz'))
    ap.add_argument('--output_dir',type=Path,default=Path('output/cgp_align/paper/manuscript_cgp_align_profile/supplementary_experiments/sr5_scaffold_split_v1'))
    ap.add_argument('--seed',type=int,default=41); ap.add_argument('--n_splits',type=int,default=5); ap.add_argument('--n_estimators',type=int,default=160); ap.add_argument('--bootstrap',type=int,default=5000)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    main_idx=smiles_index(a.main_dir/'toxric_benchmark_unique_smiles.json'); atad5_idx=smiles_index(a.atad5_dir/'toxric_benchmark_unique_smiles.json'); nyan_idx,nyan_x=load_nyan(a.nyan_cache)
    rows=[]; task_info=[]
    for task in SR5:
        df=load_task(a.toxric_dir,task); y=df.label.to_numpy(int); groups=df.scaffold.to_numpy(str); smiles=df.smiles.tolist()
        task_info.append(dict(task=task,n=len(df),positive=int(y.sum()),negative=int((y==0).sum()),positive_rate=float(y.mean()),n_scaffolds=len(set(groups)),largest_scaffold_size=int(pd.Series(groups).value_counts().iloc[0])))
        for rep in REPS:
            missing=[s for s in smiles if (rep=='NYAN_latent' and s not in nyan_idx) or (rep!='NYAN_latent' and s not in (atad5_idx if task.endswith('SR-ATAD5') else main_idx))]
            if missing: raise RuntimeError(f'{task} {rep} missing {len(missing)} smiles; first={missing[:3]}')
            x=get_features(task,rep,smiles,a.main_dir,a.atad5_dir,main_idx,atad5_idx,nyan_idx,nyan_x)
            rows += eval_one(x,y,groups,task,rep,a.n_splits,a.seed,a.n_estimators)
            print(task, rep, 'done')
    folds=pd.DataFrame(rows); endpoint,mean,paired=stats(folds,a.bootstrap,a.seed)
    pd.DataFrame(task_info).to_csv(a.output_dir/'sr5_scaffold_task_summary.csv',index=False)
    folds.to_csv(a.output_dir/'sr5_scaffold_fold_metrics.csv',index=False)
    endpoint.to_csv(a.output_dir/'sr5_scaffold_endpoint_metrics.csv',index=False)
    mean.to_csv(a.output_dir/'sr5_scaffold_mean_metrics.csv',index=False)
    paired.to_csv(a.output_dir/'sr5_scaffold_cgp_vs_baseline_paired_stats.csv',index=False)
    meta=dict(method='Bemis-Murcko scaffold grouped StratifiedGroupKFold',tasks=SR5,reps=REPS,seed=a.seed,n_splits=a.n_splits,n_estimators=a.n_estimators,outputs=sorted(p.name for p in a.output_dir.glob('*.csv')))
    (a.output_dir/'sr5_scaffold_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(mean.to_string(index=False)); print(paired.to_string(index=False))
if __name__=='__main__': main()
