import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import sys,json,hashlib,argparse
from pathlib import Path
import numpy as np,pandas as pd
ROOT=F=BASE=INP=EMB=None
CODE=Path(__file__).resolve().parents[2];sys.path[:0]=[str(CODE/'scripts'),str(CODE)]
import benchmark_toxric_molecular_representations as bench
from concurrent.futures import ThreadPoolExecutor
def toxicity():
 out=ROOT/'tox5';out.mkdir(exist_ok=True);rows=[];pred=[];splits=[]
 def run(item):
  task,df=item;y=df.label.to_numpy(int);key='atad5' if task.endswith('ATAD5') else 'main'
  sidx={s:i for i,s in enumerate(json.loads((INP/f'toxicity/{key}/smiles.json').read_text()))}
  original=np.load(INP/f'toxicity/{key}/RDKit2D.npy',mmap_mode='r')
  rdkit=np.asarray(original[[sidx[s] for s in df.smiles]])
  saved=pd.read_csv(F/(task+'.csv'));assert saved.smiles.tolist()==df.smiles.tolist();assert np.array_equal(saved.label,y)
  rr=[];pp=[];ss=[]
  for rep,x in [('RDKit2D',rdkit),('ChemBERTa',np.load(F/(task+'_chemberta.npy'))),('MoLFormer',np.load(F/(task+'_molformer_xl.npy')))]:
   for fold,(tr,te) in enumerate(bench.cv_splits(x,y,5,41)):
    model=bench.make_model('extratrees',41+fold,160,2000,x.shape[1]);model.fit(x[tr],y[tr]);score=bench.predict_scores(model,x[te])
    rr.append(dict(task=task,representation=rep,model='extratrees',fold=fold,n_train=len(tr),n_test=len(te),positive_train=int(y[tr].sum()),positive_test=int(y[te].sum()),auroc=bench.roc_auc_score(y[te],score),auprc=bench.average_precision_score(y[te],score)))
    pp.append(pd.DataFrame(dict(task=task,method=rep,fold=fold,row=te,y=y[te],score=score)))
    if rep=='RDKit2D':ss.append(pd.DataFrame(dict(task=task,fold=fold,row=te,smiles=df.smiles.iloc[te],y=y[te])))
  # Original fixed-parameter RDKit values are a positive control for curation/splits.
  ref=pd.read_csv(INP/f'reference/toxicity_{key}_random_folds.csv');ref=ref[(ref.task==task)&(ref.representation=='RDKit2D')&(ref.model=='extratrees')].sort_values('fold')
  current=pd.DataFrame(rr);got=current[current.representation=='RDKit2D'].sort_values('fold')
  assert len(ref)==5,(task,len(ref))
  for col in ['n_train','n_test','positive_train','positive_test','auroc','auprc']:assert np.allclose(ref[col],got[col],atol=1e-10), (task,col,ref[col].tolist(),got[col].tolist())
  print('TOX5_VERIFIED',task,flush=True);return rr,pp,ss
 tasks=bench.load_tasks(INP/'toxicity/tasks','all')
 with ThreadPoolExecutor(max_workers=4) as pool:
  for rr,pp,ss in pool.map(run,tasks):rows+=rr;pred+=pp;splits+=ss
 r=pd.DataFrame(rows);r.to_csv(out/'fold_metrics.csv',index=False);r.groupby(['task','representation'])[['auroc','auprc']].mean().to_csv(out/'endpoint_metrics.csv')
 pd.concat(pred,ignore_index=True).to_parquet(out/'oof_predictions.parquet',index=False);pd.concat(splits,ignore_index=True).to_csv(out/'fold_assignments.csv',index=False)
 (out/'checks.json').write_text(json.dumps({'RDKit_original_metrics_reproduced':True,'n_tasks':len(tasks),'all_samples_included':True,'splits':5,'classifier_seed':'41+fold','n_estimators':160},indent=2))
def prism():
 import evaluate_prism_high_confidence_hits as hits
 import evaluate_prism_continuous_metrics as cont
 from prism_nyan_utils import load_nyan_similarity
 out=ROOT/'prism';out.mkdir(exist_ok=True);inp=INP/'prism'
 z=np.load(inp/'prism_functional_retrieval_matrices.npz',allow_pickle=True);meta=pd.read_csv(inp/'prism_cgp_overlap_compounds.csv')
 assert np.array_equal(z['inchikey'],meta.inchikey)
 def cosine(x):
  x=np.asarray(x,np.float32);x=x/np.linalg.norm(x,axis=1,keepdims=True);s=x@x.T;np.fill_diagonal(s,-np.inf);return s
 scores={'CGP-Align':cosine(np.load(EMB/'seed41/prism.npy')),'RDKit-Morgan':cosine(np.load(inp/'morgan_rdkit_features.npy')),'NYAN':load_nyan_similarity(inp,meta.canonical_smiles.tolist()),'ChemBERTa':cosine(np.load(F/'prism_chemberta.npy')),'MoLFormer':cosine(np.load(F/'prism_molformer_xl.npy'))}
 assert json.loads((F/'smiles_sets.json').read_text())['prism']==meta.canonical_smiles.tolist()
 t=z['morgan_tanimoto'];response=z['response_similarity'];y=z['prism_response'][:,list(z['cell_lines']).index('ACH-000364')]
 summaries=[];continuous=[];u2os=[]
 for cutoff in [.30,.25,.20,.15]:
  mask=(t<cutoff)&np.isfinite(t);np.fill_diagonal(mask,False)
  random,_,_=hits.collect_hits('Random',np.zeros_like(t),response,mask,.30,50,np.random.default_rng(13+10007))
  for method,score in scores.items():
   assert np.isfinite(score[mask]).all()
   result,_,_=hits.collect_hits(method,score,response,mask,.30,50)
   result.update(cutoff=cutoff,enrichment=result['hit_yield_per_1000_ranked']/random['hit_yield_per_1000_ranked']);summaries.append(result)
   q=cont.evaluate_score_matrix(method,score,response,mask,[50],10);q['cutoff']=cutoff;continuous.append(q)
   if cutoff==.20:
    for i in range(len(y)):
     ids=np.flatnonzero(mask[i]&np.isfinite(y));assert len(ids)>=50
     selected=ids[np.argsort(-score[i,ids],kind='stable')[:50]]
     u2os.append(dict(method=method,query_idx=i,active=bool(y[i]<=-1),top_active_fraction=float((y[selected]<=-1).mean()),mean_absdiff=float(abs(y[selected]-y[i]).mean()),n_candidates=len(ids)))
   print('PRISM',cutoff,method,result['enrichment'],flush=True)
 summary=pd.DataFrame(summaries);summary.to_csv(out/'enrichment.csv',index=False)
 q=pd.concat(continuous,ignore_index=True);q.to_csv(out/'continuous_queries.csv',index=False);q.groupby(['cutoff','method'])[['retention50','top50_corr']].mean().to_csv(out/'continuous_summary.csv')
 u=pd.DataFrame(u2os);u.to_csv(out/'u2os_queries.csv',index=False);u.groupby(['active','method'])[['top_active_fraction','mean_absdiff']].mean().to_csv(out/'u2os_summary.csv')
 c=summary[(summary.method=='CGP-Align')&(summary.cutoff==.20)].iloc[0];assert round(c.enrichment,2)==5.57,c.enrichment
 v=u[(u.method=='CGP-Align')&u.active];assert len(v)==187 and round(v.mean_absdiff.mean(),2)==1.80
 (out/'checks.json').write_text(json.dumps({'all_1856_queries_and_common_candidates':True,'active_queries':187,'CGP_original_enrichment_reproduced':float(c.enrichment),'CGP_original_active_absdiff_reproduced':float(v.mean_absdiff.mean())},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--features',type=Path,required=True);ap.add_argument('--embeddings',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--task',choices=['toxicity','prism','all'],default='all');a=ap.parse_args()
 INP=a.inputs;F=a.features;EMB=a.embeddings;ROOT=a.output;ROOT.mkdir(parents=True,exist_ok=True)
 if a.task in ['toxicity','all']:toxicity()
 if a.task in ['prism','all']:prism()
 print('COMPLETE',flush=True)
