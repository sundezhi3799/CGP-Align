import sys,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path('/data3/sdz/cgp_align_final_arch_code_20260920');sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import eval_raw3180_gene_union_latent_link_classifier as mod
OUT=Path('/data3/sdz/cgp_align_downstream_20260920');EMB=OUT/'embeddings'
def cached(checkpoint,device,batch,modalities):
 seed=int(str(checkpoint));d=EMB/f'seed{seed}'
 while not (d/'provenance.json').exists():time.sleep(10)
 c=pd.read_parquet(d/'compound_entities.parquet');g=pd.read_parquet(d/'gene_entities.parquet');zc=np.load(d/'compound.npy');zg=np.load(d/'gene.npy');symbols=g.gene_symbol.fillna('').str.upper();names=sorted(set(symbols)-{''});groups={s:[] for s in names}
 for i,s in enumerate(symbols):
  if s:groups[s].append(i)
 z=mod.l2_normalize(np.vstack([zg[groups[s]].mean(0) for s in names]));meta=json.loads((d/'provenance.json').read_text());return {'epoch':meta['epoch']},c,np.arange(len(c)),zc,names,z
mod.load_model_gene_union_embeddings=cached
for graph,folder in [('broad','cgp_cpg_full_motive_edges_graphclean'),('strict','cgp_cpg_full_strict_target_edges_graphclean')]:
 sys.argv=['classifier','--rows','final_seed31:31,final_seed37:37,final_seed41:41','--target_edges',f'/home/sdz/projects/cgp_align_cpg_full/data/{folder}/compound_target_edges_typed.parquet','--output_dir',str(OUT/'figure4'/graph),'--source','all','--split_modes','','--holdout_sources','dgidb,drugrep,hetionet,openbiolink,pharmebinet','--negative_ratio','5','--max_train_pos','10000','--max_eval_pos','3000','--seed','13','--device','cpu']
 mod.main()
(OUT/'figure4/CLASSIFIER_COMPLETE').write_text('complete')
