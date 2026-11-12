"""Nested scaffold tuning of frozen molecular representations."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import sys,json,hashlib,time,threading,argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd,sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score,average_precision_score
from scaffold_split import make_splits
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'tox30'
BASE=None
INPUT=None
REPS=['CGP','RDKit','NYAN','CGP_RDKit','NYAN_RDKit','ChemBERTa','ChemBERTa_RDKit','MoLFormer','MoLFormer_RDKit']
METHODS=REPS
CONFIGS=[dict(n_estimators=160,min_samples_leaf=2,max_features='sqrt'),
         dict(n_estimators=320,min_samples_leaf=1,max_features='sqrt'),
         dict(n_estimators=320,min_samples_leaf=4,max_features='sqrt'),
         dict(n_estimators=320,min_samples_leaf=2,max_features=0.15)]
WEIGHTS=[(1/3,1/3,1/3)]+[(i/4,j/4,(4-i-j)/4) for i in range(5) for j in range(5-i)]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,default=str));tmp.replace(path)
def fingerprint(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def validate_cache(path,meta,run_id,expected):
    info=json.loads(meta.read_text())
    if info['run_id']!=run_id:raise ValueError('Cached prediction protocol mismatch')
    if sha(path)!=info['prediction_sha256']:raise ValueError('Cached prediction checksum mismatch')
    p=pd.read_parquet(path)
    keys=['target','seed','fold','method','compound_index']
    if p.duplicated(keys).any() or not p.score.between(0,1).all():raise ValueError('Invalid cached predictions')
    if len(p)!=len(expected)*len(METHODS) or set(p.method)!=set(METHODS):raise ValueError('Incomplete cached methods')
    for _,g in p.groupby('method'):
        pd.testing.assert_frame_equal(g[expected.columns].sort_values('compound_index').reset_index(drop=True),expected.sort_values('compound_index').reset_index(drop=True),check_dtype=False)
    return p
def main():
    global BASE,INPUT,OUT
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--preflight-only',action='store_true');ap.add_argument('--workers',type=int,default=1);a=ap.parse_args()
    BASE=INPUT=a.inputs/'tox30';F=a.inputs/'baseline_features';OUT=a.output;OUT.mkdir(parents=True,exist_ok=True)
    assert sklearn.__version__=='1.6.1'
    old={'input_sha256':{str(p.relative_to(INPUT)):sha(p) for p in [INPUT/'inputs/labels.parquet',INPUT/'inputs/features.npz',INPUT/'results/nyan_features.npz']},'fold_sha256':sha(BASE/'results/fold_assignments.csv')}
    labels=pd.read_parquet(INPUT/'inputs/labels.parquet');outer=pd.read_csv(BASE/'results/fold_assignments.csv')
    f=np.load(INPUT/'inputs/features.npz');ny=np.load(INPUT/'results/nyan_features.npz')
    assert np.array_equal(f['smiles'],ny['smiles']) and ny['valid'].all()
    features={'CGP':f['CGP'],'RDKit':f['RDKit'],'NYAN':ny['NYAN_mean']}
    for key in ['CGP','NYAN']:features[key+'_RDKit']=np.c_[f['RDKit'],features[key]]
    for rep,key in [('ChemBERTa','chemberta'),('MoLFormer','molformer_xl')]:
        features[rep]=np.load(F/('tox30_'+key+'.npy'))
        assert features[rep].shape[0]==len(f['smiles']) and np.isfinite(features[rep]).all()
        features[rep+'_RDKit']=np.c_[f['RDKit'],features[rep]]
    assert json.loads((F/'smiles_sets.json').read_text())['tox30']==f['smiles'].tolist()
    protocol=dict(configs=CONFIGS,class_weight='balanced',classifier_seed='outer seed',n_jobs=1,representations=REPS,methods=METHODS,inner_folds=2,outer_folds=5,outer_seeds=[41,137],inner_seed='outer_seed * 100 + outer_fold; first class-feasible split in at most 20 consecutive seeds',selection_metric='pooled inner OOF AUROC',tie_break='first listed candidate/weight',weights=WEIGHTS,imputation='training-fold median; keep_empty_features=True',outer_aggregation='pooled OOF per endpoint and seed, then average seeds and endpoints',input_sha256=old['input_sha256'],outer_fold_sha256=old['fold_sha256'],code_sha256={p:sha(ROOT/p) for p in ['run_tox30.py','scaffold_split.py']},versions=dict(sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__),planned_fits=24300,no_encoder_training=True)
    protocol['added_features_sha256']={k:sha(F/('tox30_'+k+'.npy')) for k in ['chemberta','molformer_xl']};run_id=fingerprint(protocol);protocol['run_id']=run_id
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())['run_id']==run_id,'Output directory belongs to a different protocol'
    else:
        assert not list(OUT.glob('pred_*.parquet')),'Unidentified cached predictions'
        save(pp,protocol)
    jobs=[];splitrecords=[];checks=[]
    for ti,target in enumerate(sorted(labels.target.unique())):
        d=labels[labels.target.eq(target)].sort_values('compound_index').reset_index(drop=True)
        y=d.y.to_numpy(int);groups=d.scaffold.to_numpy()
        for seed in [41,137]:
            assignment=outer[outer.target.eq(target)&outer.seed.eq(seed)].set_index('compound_index').fold.reindex(d.compound_index).to_numpy()
            assert np.isfinite(assignment).all()
            for fold in range(5):
                tr=np.flatnonzero(assignment!=fold);te=np.flatnonzero(assignment==fold)
                assert not set(groups[tr])&set(groups[te]) and len(np.unique(y[tr]))==len(np.unique(y[te]))==2
                for attempt in range(20):
                    try:inner=make_splits(y[tr],groups[tr],2,seed*100+fold+attempt);break
                    except ValueError:continue
                else:raise RuntimeError(f'No feasible inner split: {target}, {seed}, {fold}')
                inf=np.full(len(tr),-1)
                for j,(itr,ite) in enumerate(inner):
                    assert not set(groups[tr[itr]])&set(groups[tr[ite]])
                    inf[ite]=j
                    checks.append(dict(target=target,seed=seed,outer_fold=fold,inner_fold=j,attempt=attempt,n_train=len(itr),n_val=len(ite),positive_train=int(y[tr[itr]].sum()),positive_val=int(y[tr[ite]].sum()),negative_train=int(len(itr)-y[tr[itr]].sum()),negative_val=int(len(ite)-y[tr[ite]].sum()),scaffold_overlap=0))
                assert (inf>=0).all()
                splitrecords.append(pd.DataFrame(dict(target=target,seed=seed,outer_fold=fold,compound_index=d.compound_index.iloc[tr].to_numpy(),inner_fold=inf)))
                jobs.append((f'{ti:02d}_{seed}_{fold}',d,seed,fold,tr,te,inner))
    assert len(jobs)==300 and len(checks)==600
    pd.concat(splitrecords,ignore_index=True).to_csv(OUT/'inner_assignments.csv.gz',index=False,compression='gzip')
    previous=pd.read_csv(BASE/'results/inner_assignments.csv.gz')
    current=pd.concat(splitrecords,ignore_index=True)
    pd.testing.assert_frame_equal(current,previous)
    pd.DataFrame(checks).to_csv(OUT/'inner_checks.csv',index=False)
    save(OUT/'preflight.json',dict(passed=True,run_id=run_id,outer_partitions=300,inner_partitions=600,planned_fits=24300,all_inner_train_val_have_both_classes=True,maximum_attempt=max(x['attempt'] for x in checks),outer_fold_sha256=old['fold_sha256']))
    print('PREFLIGHT_OK 300 outer partitions; 600 valid inner folds; 24300 fits',flush=True)
    if a.preflight_only:return
    lock=threading.Lock();completed=0
    def train(job):
        tag,d,seed,fold,tr,te,inner=job;y=d.y.to_numpy(int);ids=d.compound_index.to_numpy()
        path=OUT/f'pred_{tag}.parquet';meta=OUT/f'selection_{tag}.json'
        expected=d.iloc[te].copy();expected['seed']=seed;expected['fold']=fold
        if path.exists() or meta.exists():
            if not(path.exists() and meta.exists()):raise RuntimeError(f'Incomplete partition {tag}; inspect before resuming')
            return validate_cache(path,meta,run_id,expected)
        selected_inner={};selected_test={};selection={};audit=[]
        for rep in REPS:
            x=features[rep][ids];ip=np.full((len(tr),len(CONFIGS)),np.nan)
            for j,(itr,ite) in enumerate(inner):
                imputer=SimpleImputer(strategy='median',keep_empty_features=True)
                xa=imputer.fit_transform(x[tr[itr]]);xb=imputer.transform(x[tr[ite]])
                for ci,conf in enumerate(CONFIGS):
                    model=ExtraTreesClassifier(**conf,class_weight='balanced',random_state=seed,n_jobs=1)
                    model.fit(xa,y[tr[itr]]);ip[ite,ci]=model.predict_proba(xb)[:,1]
            assert np.isfinite(ip).all()
            scores=[roc_auc_score(y[tr],ip[:,ci]) for ci in range(len(CONFIGS))]
            win=int(np.argmax(scores));selection[rep]=dict(candidate=win,inner_AUROC=scores)
            selected_inner[rep]=ip[:,win]
            for ci,score in enumerate(scores):audit.append(dict(representation=rep,candidate=ci,AUROC=float(score)))
            imp=SimpleImputer(strategy='median',keep_empty_features=True)
            xa=imp.fit_transform(x[tr]);xb=imp.transform(x[te])
            model=ExtraTreesClassifier(**CONFIGS[win],class_weight='balanced',random_state=seed,n_jobs=1)
            model.fit(xa,y[tr]);selected_test[rep]=model.predict_proba(xb)[:,1]
        predictions=[]
        for method,score in selected_test.items():
            r=expected.copy();r['method']=method;r['score']=score;predictions.append(r)
        pred=pd.concat(predictions,ignore_index=True);assert pred.score.between(0,1).all()
        tmp=path.with_suffix('.parquet.tmp');pred.to_parquet(tmp,index=False);tmp.replace(path)
        save(meta,dict(run_id=run_id,prediction_sha256=sha(path),selection=selection,inner_scores=audit,completed_fits=81))
        validate_cache(path,meta,run_id,expected)
        print('FOLD_DONE',tag,flush=True)
        return pred
    save(OUT/'status.json',dict(stage='training',run_id=run_id,completed_partitions=0,total_partitions=300,planned_fits=24300))
    parts=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending=[pool.submit(train,job) for job in jobs]
        for future in as_completed(pending):
            parts.append(future.result());completed+=1
            save(OUT/'status.json',dict(stage='training',run_id=run_id,completed_partitions=completed,total_partitions=300,completed_fits=completed*81))
    p=pd.concat(parts,ignore_index=True)
    assert len(p)==len(labels)*2*len(METHODS) and not p.duplicated(['target','seed','method','compound_index']).any()
    p.to_parquet(OUT/'oof_predictions.parquet',index=False)
    rows=[]
    for (target,seed,method),g in p.groupby(['target','seed','method']):
        rows.append(dict(target=target,seed=seed,method=method,AUROC=roc_auc_score(g.y,g.score),AP=average_precision_score(g.y,g.score)))
    m=pd.DataFrame(rows);m.to_csv(OUT/'metrics_by_seed.csv',index=False)
    e=m.groupby(['target','method'],as_index=False)[['AUROC','AP']].mean();e.to_csv(OUT/'metrics_by_endpoint.csv',index=False)
    e.groupby('method')[['AUROC','AP']].mean().to_csv(OUT/'metrics_macro.csv')
    save(OUT/'status.json',dict(stage='complete',completed_partitions=300,completed_fits=24300,n_predictions=len(p),run_id=run_id,finished=time.time()))
    print('COMPLETE',len(p),flush=True)
if __name__=='__main__':
    main()
