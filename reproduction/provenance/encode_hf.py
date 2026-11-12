import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['OMP_NUM_THREADS']='4'
import sys,json,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from transformers import AutoModel,AutoTokenizer
ROOT=Path('/data3/sdz/projects/CGP-Align/revisions_R1_4_20261110')
OUT=ROOT/'features';OUT.mkdir(exist_ok=True)
BASE=Path('/data3/sdz/cgp_align_completion_release_20260921/inputs')
MATCH=Path('/data3/sdz/cgp_align_matched_20260919/seed41/data')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
sets={}
sets['tox30']=np.load('/data3/sdz/projects/TCM-Safety/toxric30_nyan_20261013/inputs/features.npz')['smiles'].astype(str).tolist()
sets['prism']=pd.read_csv(BASE/'prism/prism_cgp_overlap_compounds.csv').canonical_smiles.astype(str).tolist()
from rdkit import Chem,RDLogger
RDLogger.DisableLog('rdApp.*')
for p in sorted((BASE/'toxicity/tasks').glob('*.csv')):
 d=pd.read_csv(p);d['smiles']=d['Canonical SMILES'].map(lambda s:Chem.MolToSmiles(Chem.MolFromSmiles(str(s)),canonical=True) if Chem.MolFromSmiles(str(s)) else '')
 d['label']=pd.to_numeric(d['Toxicity Value'],errors='coerce');d=d[(d.smiles!='')&d.label.isin([0,1])].drop_duplicates(['smiles','label'])
 bad=d.groupby('smiles').label.nunique();d=d[~d.smiles.isin(bad[bad>1].index)].reset_index(drop=True)
 key=p.stem;sets[key]=d.smiles.tolist();d[['smiles','label']].to_csv(OUT/(key+'.csv'),index=False)
(OUT/'smiles_sets.json').write_text(json.dumps(sets))
allsm=sorted(set(s for seq in sets.values() for s in seq));lookup={s:i for i,s in enumerate(allsm)}
for method,modelid in [('chemberta','DeepChem/ChemBERTa-77M-MTR'),('molformer_xl','ibm/MoLFormer-XL-both-10pct')]:
 dest=OUT/(method+'.npz')
 if dest.exists():print('EXISTS',dest,flush=True);continue
 torch.manual_seed(41)
 tok=AutoTokenizer.from_pretrained(modelid,trust_remote_code=True,local_files_only=True)
 extra={'deterministic_eval':True} if method=='molformer_xl' else {}
 model=AutoModel.from_pretrained(modelid,trust_remote_code=True,local_files_only=True,**extra).cuda().eval()
 def encode(sm):
  parts=[]
  with torch.no_grad():
   for k in range(0,len(sm),128):
    t=tok(sm[k:k+128],padding=True,truncation=True,max_length=256,return_tensors='pt');t={n:v.cuda() for n,v in t.items()}
    h=model(**t).last_hidden_state;m=t['attention_mask'].unsqueeze(-1).to(h.dtype)
    parts.append(((h*m).sum(1)/m.sum(1).clamp_min(1)).cpu().numpy().astype('float32'))
    if k%5120==0:print(method,k,len(sm),flush=True)
  return np.concatenate(parts)
 table=pd.read_parquet(MATCH/method/'compounds.parquet');reference=np.load(MATCH/method/'compound_structure_features.npy',mmap_mode='r')
 probe=encode(table.canonical_smiles.iloc[:32].tolist());expected=np.asarray(reference[:32])
 check={'rms':float(np.sqrt(np.mean((probe-expected)**2))),'max_abs':float(abs(probe-expected).max()),'cosine_mean':float(np.mean(np.sum(probe*expected,1)/(np.linalg.norm(probe,axis=1)*np.linalg.norm(expected,axis=1))))}
 print('REPRODUCTION',method,check,flush=True)
 # Same canonical SMILES, pretrained weights and pooling as historical features.
 if method=='chemberta' and check['cosine_mean']<.999:raise ValueError(('Historical feature reproduction failed',method,check))
 repeat=encode(table.canonical_smiles.iloc[:32].tolist())
 assert np.allclose(probe,repeat,atol=1e-6), 'Encoder is not repeatable'
 check['repeat_max_abs']=float(abs(probe-repeat).max())
 check['historical_note']='Historical MoLFormer redrew random attention features on each forward pass; added downstream evaluations use deterministic_eval=True with the same pretrained checkpoint.' if method=='molformer_xl' else 'Historical features reproduced.'
 x=encode(allsm);assert np.isfinite(x).all() and (np.linalg.norm(x,axis=1)>0).all()
 np.savez_compressed(dest,smiles=np.asarray(allsm),features=x)
 for key,sm in sets.items():np.save(OUT/(key+'_'+method+'.npy'),x[[lookup[s] for s in sm]])
 hub=Path.home()/'.cache/huggingface/hub'/('models--'+modelid.replace('/','--'))
 snapshots=list((hub/'snapshots').iterdir())
 provenance={'model':modelid,'pooling':'attention-mask mean including special tokens','max_length':256,'canonical_only':True,'frozen':True,'shape':list(x.shape),'historical_reproduction':check,'torch':torch.__version__,'input_sets_sha256':sha(OUT/'smiles_sets.json'),'output_sha256':sha(dest),'snapshot_files':{str(p.relative_to(hub)):sha(p) for snap in snapshots for p in snap.rglob('*') if p.is_file()}}
 (OUT/(method+'_provenance.json')).write_text(json.dumps(provenance,indent=2))
 del model;torch.cuda.empty_cache()
print('COMPLETE',flush=True)
