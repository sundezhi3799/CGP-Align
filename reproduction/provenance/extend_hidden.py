"""Extend frozen-feature phenotype-neighbour recovery using the original cohort."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import numpy as np,pandas as pd,torch,json,hashlib
R=Path('/data3/sdz/projects/CGP-Align/revisions_R1_4_20261110')
O=R/'layout_refresh/hidden';O.mkdir(parents=True,exist_ok=True)
M=Path('/data3/sdz/cgp_align_matched_20260919/seed41/data')
old=R/'publication/figure3/source_data/figure3h_hidden_phenotype_neighbour_recovery.csv'
audit=json.loads((M/'audit.json').read_text())
rows=np.asarray(json.loads((M/'morgan/splits_intrinsic_entity.json').read_text())['compound']['test'])
ids=np.asarray(audit['test_ids']).astype(str);assert len(ids)==len(rows)==11578
table=pd.read_parquet(M/'morgan/compounds.parquet')
assert table.canonical_smiles.tolist()==json.loads((R/'features/fig3_smiles.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(M/'profile_features.npy')==audit['entity_mean_sha256']
def norm(a):
 a=np.asarray(a,dtype='float32');return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-8)
raw=norm(np.load(M/'profile_features.npy',mmap_mode='r')[rows])
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
n=len(rows)
def rankings(x,k):
 x=torch.as_tensor(x,device='cuda');all_top=[]
 for start in range(0,n,256):
  end=min(n,start+256);s=(x[start:end]@x.T).cpu().numpy();s[np.arange(end-start),np.arange(start,end)]=-np.inf
  all_top.append(np.argsort(-s,axis=1,kind='stable')[:,:k])
 return np.concatenate(all_top)
pos=rankings(raw,10);np.save(O/'positive_indices.npy',pos)
features={'Structure feature baseline':M/'morgan/compound_structure_features.npy','ChemBERTa':M/'chemberta/compound_structure_features.npy','MoLFormer':M/'molformer_xl/compound_structure_features.npy','NYAN':R/'nyan/compound_structure_features.npy'}
records=[];queries=[];provenance={}
for method,path in features.items():
 x=norm(np.load(path,mmap_mode='r')[rows]);xt=torch.as_tensor(x,device='cuda');hits=[];best=[];tops=[]
 for start in range(0,n,256):
  end=min(n,start+256);s=(xt[start:end]@xt.T).cpu().numpy();s[np.arange(end-start),np.arange(start,end)]=-np.inf
  order=np.argsort(-s,axis=1,kind='stable');inverse=np.empty_like(order);np.put_along_axis(inverse,order,np.arange(n)[None,:],axis=1)
  ranks=np.take_along_axis(inverse,pos[start:end],axis=1)+1
  hits.extend((ranks<=50).sum(axis=1).tolist());best.extend(ranks.min(axis=1).tolist());tops.append(order[:,:50])
 hit=np.asarray(hits)
 rec=dict(scope='compound_to_compound',method=method,k=50,n_query=n,n_gallery=n-1,hit_at_k=float((hit>0).mean()),recall_at_k=float((hit/10).mean()),precision_enrichment=float((hit/50).mean()/(10/(n-1))),mean_best_positive_rank=float(np.mean(best)))
 print(rec,flush=True);records.append(rec)
 queries.extend(dict(compound_id=i,method=method,hits_at_50=h,best_positive_rank=b) for i,h,b in zip(ids,hits,best))
 np.save(O/(method.replace(' ','_')+'_top50.npy'),np.concatenate(tops))
 provenance[method]={'input':str(path),'sha256':sha(path),'dimension':x.shape[1]}
original=pd.read_csv(old);reference=original[original.method=='Structure feature baseline'].iloc[0]
for key in ['hit_at_k','recall_at_k','precision_enrichment','mean_best_positive_rank']:assert abs(records[0][key]-reference[key])<1e-10,(key,records[0][key],reference[key])
combined=pd.concat([original,pd.DataFrame(records[1:])],ignore_index=True)
combined.to_csv(O/'figure3h_hidden_phenotype_neighbour_recovery.csv',index=False)
pd.DataFrame(queries).to_csv(O/'hidden_query_metrics_added.csv',index=False)
(O/'provenance.json').write_text(json.dumps({'n':n,'positive_topn':10,'k':50,'tie_policy':'ascending gallery index','original_RDKit_Morgan_reproduced':True,'profile_sha256':audit['entity_mean_sha256'],'positive_indices_sha256':sha(O/'positive_indices.npy'),'inputs':provenance,'MoLFormer':'Historical frozen features as used in Fig. 3f,g, without downstream deterministic re-encoding.'},indent=2))
print('COMPLETE',flush=True)
