from pathlib import Path
import json,numpy as np,pandas as pd
from scipy.sparse import csr_matrix
OUT=Path('/data3/sdz/cgp_align_downstream_20260920/s2');OUT.mkdir(exist_ok=True)
composition=[];splits=[];repdist=[];r2rows=[];overlap=[];audits=[]
def label_r2(x,labels):
 codes,unique=pd.factorize(labels,sort=True);assert (codes>=0).all()
 counts=np.bincount(codes);a=csr_matrix((np.ones(len(codes)),(codes,np.arange(len(codes)))),shape=(len(unique),len(codes)))
 r=[]
 for j in range(0,x.shape[1],64):
  v=np.asarray(x[:,j:j+64],dtype=np.float64);mu=v.mean(0);total=np.square(v-mu).sum(0);means=(a@v)/counts[:,None]
  r.extend((np.square(means-mu)*counts[:,None]).sum(0)/np.maximum(total,1e-12))
 return dict(groups=len(unique),mean_r2=float(np.mean(r)),median_r2=float(np.median(r)),max_r2=float(np.max(r)),p90_r2=float(np.quantile(r,.9)))
for seed in (31,37,41):
 for kind in ('compound','gene'):
  d=Path('/data3/sdz/cgp_align_strict_20260918')/f'seed{seed}/data/{kind}';e=pd.read_parquet(d/f'{kind}_mocop_entities.parquet');p=pd.read_parquet(d/f'{kind}_mocop_replicates.parquet');x=np.load(d/f'{kind}_mocop_replicate_features.npy',mmap_mode='r');folds=json.loads((d/f'splits_{kind}_mocop.json').read_text())['cold_'+kind]
  assert np.array_equal(p.feature_index,np.arange(len(p)))
  modalities=np.repeat('Compound',len(e)) if kind=='compound' else e.perturbation_modality.str.upper().to_numpy()
  counts=p.entity_index.value_counts().reindex(np.arange(len(e)),fill_value=0).to_numpy()
  for m in sorted(set(modalities)):
   ix=np.flatnonzero(modalities==m)
   composition.append(dict(seed=seed,modality=m,entity_count=len(ix),profile_count=int(counts[ix].sum())))
   if seed==31:
    for i in ix:repdist.append(dict(modality=m,entity_index=int(i),replicate_count=int(counts[i])))
   for fold,values in folds.items():
    rows=np.intersect1d(ix,values);splits.append(dict(seed=seed,modality=m,split=fold,entity_count=len(rows),profile_count=int(counts[rows].sum())))
  for first,second in [('train','val'),('train','test'),('val','test')]:
   n=len(set(folds[first])&set(folds[second]));assert n==0
   rec=dict(seed=seed,dataset=kind,first=first,second=second,entity_overlap=n)
   if kind=='gene':
    symbols=e.gene_symbol.fillna('').str.upper();n=len((set(symbols.iloc[folds[first]])&set(symbols.iloc[folds[second]]))-{''});assert n==0;rec['gene_symbol_overlap']=n
   overlap.append(rec)
  for factor,col in [('entity','entity_id'),('plate','Metadata_Plate'),('well','Metadata_Well')]:
   r2rows.append(dict(seed=seed,dataset=kind.title(),factor=factor,**label_r2(x,p[col].fillna('').astype(str))))
   print(seed,kind,factor,r2rows[-1]['mean_r2'],flush=True)
  a=json.loads((d/'profile_correction_audit.json').read_text());audits.append(a)
for name,rows in [('composition',composition),('splits',splits),('replicate_distribution',repdist),('r2',r2rows),('overlap',overlap)]:pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
(OUT/'preprocessing_audits.json').write_text(json.dumps(audits,indent=2));(OUT/'COMPLETE').write_text('complete')
