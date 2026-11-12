"""Fixed C=1, training-only PCA and complete cross-modal outer products."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import json,hashlib,warnings,argparse,time
import numpy as np,pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score,roc_auc_score
from concurrent.futures import ProcessPoolExecutor,as_completed
R=Path(__file__).resolve().parent;F=R.parent/'features';S=R/'matched_splits';O=R/'results'
METHODS=['RDKit','RDKit-Morgan','Morgan2048','AtomPair2048','Avalon1024','NYAN','ChemBERTa','MoLFormer','NYAN+RDKit','ChemBERTa+RDKit','MoLFormer+RDKit','CGP-Align_ESM2','CGP-Align+RDKit_ESM2','CGP-Align_aligned','CGP-Align+RDKit_aligned']
DIM=int(os.environ.get('RELATION_DIM',32))
assert DIM in [32,64]
O=R/('results'+str(DIM))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(y,p):return {'auprc':float(average_precision_score(y,p)),'auroc':float(roc_auc_score(y,p))}
def query_ap(pairs,y,score):
 vals=[]
 for ci in np.unique(pairs[:,0]):
  ix=pairs[:,0]==ci
  if 0<y[ix].sum()<ix.sum():vals.append(average_precision_score(y[ix],score[ix]))
 return float(np.mean(vals))
def one_split(file, features=None, output=None, dimension=None, methods=None):
 global F,O,DIM,METHODS
 if features is not None:F=Path(features)
 if output is not None:O=Path(output)
 if dimension is not None:DIM=dimension
 if methods is not None:METHODS=methods
 graph,seed,source=file.stem.split('_');seed=int(seed);z=np.load(file);projection={};proof={};records=[]
 def raw(key):
  if key=='CGP':return np.load(F/f'CGP_{seed}.npy')
  if key=='CGP_gene':return np.load(F/f'CGP_gene_{seed}.npy')
  if key.endswith('+RDKit'):return np.c_[raw(key[:-6]),raw('RDKit')]
  return np.load(F/(key+'.npy'))
 def projected(key,axis):
  if key in projection:return projection[key]
  x=raw(key);x=np.asarray(x,np.float64);x[~np.isfinite(x)]=np.nan
  ids=np.unique(z['train_pairs'][:,axis]);imp=SimpleImputer(strategy='median',keep_empty_features=True);tr=imp.fit_transform(x[ids]);sc=StandardScaler();tr=sc.fit_transform(tr)
  pca=PCA(n_components=DIM,whiten=True,svd_solver='randomized',iterated_power=7,random_state=41);pca.fit(tr)
  assert np.min(pca.explained_variance_)>1e-10,(file.name,key)
  a=pca.transform(sc.transform(imp.transform(x))).astype(np.float32);assert np.isfinite(a).all()
  projection[key]=a;proof[key]={'raw_dim':x.shape[1],'training_entities':len(ids),'explained_variance_ratio':float(pca.explained_variance_ratio_.sum())};return a
 # Both endpoint marginals must be exactly matched in all three partitions.
 for split in ['train','val','test']:
  p=z[split+'_pairs'];y=z[split+'_y'];assert y.sum()*2==len(y)
  for axis in [0,1]:assert np.array_equal(np.sort(p[y==1,axis]),np.sort(p[y==0,axis]))
 tr=z['train_pairs'];yt=z['train_y'];te=z['test_pairs'];y=z['test_y']
 counts=np.bincount(tr[yt==1,1],minlength=len(raw('ESM2')));deg=counts[te[:,1]];dm=metrics(y,deg);assert abs(dm['auprc']-.5)<1e-12 and abs(dm['auroc']-.5)<1e-12
 for method in METHODS:
  dest=O/(file.stem+'_'+method+'.json')
  if dest.exists():records.append(json.loads(dest.read_text()));continue
  t0=time.time()
  if method.startswith('CGP-Align'):
   ck='CGP+RDKit' if '+RDKit' in method else 'CGP';gk='ESM2' if method.endswith('_ESM2') else 'CGP_gene'
  else:ck=method;gk='ESM2'
  c,g=projected(ck,0),projected(gk,1)
  def features(pairs):
   a=c[pairs[:,0]];b=g[pairs[:,1]]
   return np.c_[a,b,np.einsum('bi,bj->bij',a,b).reshape(len(a),-1)]
  x=features(tr);sc=StandardScaler();x=sc.fit_transform(x).astype(np.float64)
  model=LogisticRegression(C=1.,max_iter=3000,solver='lbfgs',class_weight='balanced',tol=1e-4)
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always',ConvergenceWarning);model.fit(x,yt)
  cw=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
  score=model.predict_proba(sc.transform(features(te)).astype(np.float64))[:,1]
  val=model.predict_proba(sc.transform(features(z['val_pairs'])).astype(np.float64))[:,1]
  rec={'graph':graph,'seed':seed,'source':source,'method':method,'gene_feature':gk,'compound_feature':ck,'readout':f'{DIM}+{DIM}+{DIM}x{DIM} full outer product','C':1.,'features':x.shape[1],'n_iter':int(model.n_iter_[0]),'warnings':cw,'split_sha256':sha(file),'seconds':time.time()-t0,**metrics(y,score),'macro_query_ap':query_ap(te,y,score),'validation':metrics(z['val_y'],val),'projection':{k:proof[k] for k in [ck,gk]}}
  np.savez_compressed(dest.with_suffix('.npz'),y=y,score=score)
  dest.write_text(json.dumps(rec,indent=2));records.append(rec);print('DONE',file.stem,method,round(rec['auprc'],4),flush=True)
  if cw:raise RuntimeError(('Nonconvergence requires review',dest.name))
 (O/(file.stem+'_COMPLETE')).write_text('complete');return records
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=['pilot','full'],default='full');ap.add_argument('--features',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--dimension',type=int,choices=[32,64],default=32);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS);ap.add_argument('--split',help='Optional split filename stem');a=ap.parse_args();F=a.features;O=a.output;DIM=a.dimension;METHODS=a.methods;O.mkdir(parents=True,exist_ok=True)
 protocol={'C':1,'PCA_components_each_endpoint':DIM,'PCA_fit':'training-only unique entities; training-only imputation/scaling; whitened randomized PCA, fixed seed 41','pair_features':'all cross-dimensional products plus main effects; DIM*DIM+2*DIM dimensions; not coordinate-wise differences between independently projected modalities','methods':METHODS,'negative_ratio':1,'controls':'exact positive/negative compound and gene marginal counts; exclude union of known positives; all three splits pair-disjoint','control_randomization':'max-flow feasible matching followed by 30N degree-preserving swap attempts; not claimed uniform over feasible matchings','metrics':'AP and AUROC; per-query AP diagnostic; separate broad/strict and source-wise results; no test-based tuning','purpose':'fixed-budget matched-control relation discrimination; not original full-universe AP or cold-entity evaluation','secondary':'CGP-Align gene-feature controls','code_sha256':sha(__file__),'stage':a.stage}
 pp=O/'protocol.json'
 if pp.exists():
  existing=json.loads(pp.read_text());assert existing['code_sha256']==protocol['code_sha256']
 else:pp.write_text(json.dumps(protocol,indent=2))
 files=sorted(S.glob('*.npz'));files=[f for f in files if a.stage=='full' or '_31_' in f.name];rows=[]
 if a.split:files=[f for f in files if f.stem==a.split]
 assert files,'No matching splits'
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  futures={pool.submit(one_split,f,F,O,DIM,METHODS):f.name for f in files}
  for fut in as_completed(futures):rows.extend(fut.result());print('SPLIT_COMPLETE',futures[fut],flush=True)
 df=pd.DataFrame([{k:v for k,v in r.items() if k not in ['validation','projection','warnings']} for r in rows]);df.to_csv(O/(a.stage+'_metrics.csv'),index=False);df.groupby(['graph','method'])[['auprc','auroc','macro_query_ap']].mean().to_csv(O/(a.stage+'_summary.csv'))
 (O/(a.stage+'_COMPLETE')).write_text('complete');print('COMPLETE',a.stage,len(df),flush=True)
