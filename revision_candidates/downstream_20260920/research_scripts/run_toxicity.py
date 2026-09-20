from pathlib import Path
import sys,json,ast,hashlib
from typing import *
import numpy as np,pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.model_selection import StratifiedKFold
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'scripts'))
import benchmark_toxric_molecular_representations as bench
import eval_sr5_scaffold_split_representations as scaffold
R=ROOT/'reproducibility_audits/downstream_release_20260920';OUT=R/'figure5';OUT.mkdir(exist_ok=True)
RAW=ROOT/'output/cgp_align/paper/raw_results';TD=RAW/'tox_np_mechanism/raw_download_probe/toxric/toxric_30_datasets'
sets=json.loads((R/'smiles_sets.json').read_text());vectors={}
for key in ['tox_main','tox_atad5']:
 z=np.load(R/f'embeddings/seed41/{key}.npy')
 for s,v in zip(sets[key],z):vectors[bench.canonical_smiles(s)]=v
# Read-only extraction of non-visual legacy numerical functions.
p=ROOT/'scripts/run_cgp_expert_alert_prioritization_extratrees_repeats.py';tree=ast.parse(p.read_text(encoding='utf-8-sig'))
for node in tree.body:
 if isinstance(node,ast.FunctionDef) and node.name in ['safe_auroc','oof_extratrees_scores']:exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),globals())
folds=[];scfolds=[];cohorts=[]
for task,df in bench.load_tasks(TD,','.join(scaffold.SR5)):
 x=np.vstack([vectors[s] for s in df.smiles]);y=df.label.to_numpy(int);sp=bench.cv_splits(x,y,5,41)
 rows,summary=bench.evaluate_one(x,y,sp,task,'CGP_sep_k32','extratrees',41,160,2000);folds+=rows
 groups=df.smiles.map(scaffold.scaffold).to_numpy();scfolds+=scaffold.eval_one(x,y,groups,task,'CGP_sep_k32',5,41,160)
 for fold,(tr,te) in enumerate(sp):
  cohorts.append({'task':task,'fold':fold,'smiles_sha256':hashlib.sha256('\n'.join(df.smiles).encode()).hexdigest(),'labels_sha256':hashlib.sha256(y.tobytes()).hexdigest(),'test_indices_sha256':hashlib.sha256(te.tobytes()).hexdigest(),'n':len(y)})
 pd.DataFrame(folds).to_csv(OUT/'random_cgp_folds.csv',index=False);pd.DataFrame(scfolds).to_csv(OUT/'scaffold_cgp_folds.csv',index=False);print('benchmark',task,summary['auprc_mean'],flush=True)
(OUT/'cohort_checks.json').write_text(json.dumps(cohorts,indent=2))
cache=RAW/'expert_alert_prioritization_with_server_alerts_extratrees_repeated_seed';cnt=pd.read_csv(cache/'compound_expert_alert_counts.csv');idx={bench.canonical_smiles(s):i for i,s in enumerate(cnt.smiles)};alerts=sparse.load_npz(cache/'compound_expert_alert_matrix.npz').toarray().astype(np.float32)
rows=[]
for task,df in bench.load_tasks(TD,','.join(scaffold.SR5)):
 df=df[df.smiles.isin(idx)].reset_index(drop=True);y=df.label.to_numpy(int);z=np.vstack([vectors[s] for s in df.smiles]);a=alerts[[idx[s] for s in df.smiles]]
 for seed in [13,41,97,123,2026]:
  for name,x in [('structure_alert_extratrees',a),('cgp_latent_extratrees',z),('cgp_plus_alert_extratrees',np.hstack([z,a]))]:
   score,cv=oof_extratrees_scores(x,y,5,seed,240,4);order=np.argsort(-score);prevalence=y.mean();rows.append(dict(task=task,seed=seed,method=name,n=len(y),positive=int(y.sum()),auprc=average_precision_score(y,score),auroc=roc_auc_score(y,score),lift_at_top50=float(y[order[:50]].mean()/prevalence)))
   np.savez_compressed(OUT/f'{task.split("_")[-1]}_{seed}_{name}_oof.npz',smiles=df.smiles.to_numpy(str),labels=y,scores=score)
  pd.DataFrame(rows).to_csv(OUT/'alerts_metrics.csv',index=False);print('alerts',task,seed,flush=True)
(OUT/'COMPLETE').write_text('complete')
