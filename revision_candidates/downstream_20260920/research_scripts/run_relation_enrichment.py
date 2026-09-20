import sys,json,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/data3/sdz/cgp_align_final_arch_code_20260920');sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import eval_cgp_align_target_enrichment_gene_union as mod
OUT=Path('/data3/sdz/cgp_align_downstream_20260920/figure4');OUT.mkdir(exist_ok=True)
results=[]
for seed in (31,37,41):
 d=OUT.parent/'embeddings'/f'seed{seed}'
 while not (d/'provenance.json').exists():time.sleep(10)
 c=pd.read_parquet(d/'compound_entities.parquet');g=pd.read_parquet(d/'gene_entities.parquet');zc=np.load(d/'compound.npy');zg=np.load(d/'gene.npy');rows=np.arange(len(c));symbols,lookup,groups,mods=mod.gene_symbol_groups(g,np.arange(len(g)))
 scores=np.lib.format.open_memmap(OUT/f'seed{seed}_union_scores.npy',mode='w+',dtype=np.float32,shape=(len(c),len(symbols)))
 for start in range(0,len(c),2048):scores[start:start+2048]=mod.collapse_entity_scores_to_symbols(zc[start:start+2048]@zg.T,groups)
 scores.flush()
 for graph in ('broad','strict'):
  for source in ('dgidb','drugrep','hetionet','openbiolink','pharmebinet'):
   p=Path('/home/sdz/projects/cgp_align_cpg_full/output/cgp_align/target_source_subsets_20260608')/(graph+'_'+source+'.parquet')
   targets=mod.load_targets(p,False,False);cp,gp,meta=mod.build_symbol_positive_maps(targets,c,rows,lookup)
   m={'C2G':mod.multi_positive_rank_metrics(scores,cp,[10,50,100]),'G2C':mod.multi_positive_rank_metrics(scores.T,gp,[10,50,100])}
   rec=mod.flatten(f'final_seed{seed}','gene_union',m,meta,json.loads((d/'provenance.json').read_text())['epoch']);rec.update(seed=seed,graph=graph,source=source);results.append(rec);pd.DataFrame(results).to_csv(OUT/'enrichment_runs.csv',index=False);print(seed,graph,source,flush=True)
 del scores
(OUT/'ENRICHMENT_COMPLETE').write_text('complete')
