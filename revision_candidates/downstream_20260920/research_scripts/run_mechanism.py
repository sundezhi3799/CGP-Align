from pathlib import Path
import ast,json,math
from typing import *
import numpy as np,pandas as pd
from scipy.stats import hypergeom
ROOT=Path.cwd();R=ROOT/'reproducibility_audits/downstream_release_20260920';PRISM_DIR=R/'figure6';OUT=PRISM_DIR/'mechanism';OUT.mkdir(exist_ok=True)
p=ROOT/'scripts/run_prism_gene_mechanism_interpretation.py';tree=ast.parse(p.read_text(encoding='utf-8-sig'))
keep=[]
for node in tree.body:
 if isinstance(node,(ast.Assign,ast.AnnAssign)):
  targets=[t.id for t in node.targets if isinstance(t,ast.Name)] if isinstance(node,ast.Assign) else [node.target.id]
  if any(n.startswith('CASE_') or n in ['GENE_TOPK','METHOD_ORDER','MECHANISM_RULES'] for n in targets):keep.append(node)
 if isinstance(node,ast.FunctionDef) and node.name not in ['main','make_plot','write_report']:keep.append(node)
exec(compile(ast.Module(body=keep,type_ignores=[]),str(p),'exec'),globals())
z=np.load(R/'embeddings/seed41/gene.npy');e=pd.read_csv(R/'embeddings/seed41/gene_metadata.csv');sym=e.gene_symbol.fillna('').str.upper();idx={s:np.flatnonzero(sym.to_numpy()==s) for s in set(sym)-{''}}
old=np.load(ROOT/'output/cgp_align/paper/manuscript_cgp_align_profile/drug_repositioning_v1/repositioning_cgp_embeddings.npz',allow_pickle=True);oldnames=old['gene_symbols'].astype(str);names=np.array([s for s in oldnames if s in idx]);assert len(set(names))==len(names)
v=l2_normalize(np.vstack([z[idx[s]].mean(0) for s in names]));c=l2_normalize(np.load(R/'embeddings/seed41/prism.npy'));score=c@v.T;top=topk_indices(score,100);meta=pd.read_csv(PRISM_DIR/'prism_cgp_overlap_compounds.csv');pairs=build_pair_sets();metrics,categories,cases=annotate_pair_rows(pairs,meta,top,names,score);summary,catsummary=summarize(metrics,categories)
for name,df in [('pair_gene_mechanism_metrics',metrics),('pair_mechanism_category_hits',categories),('pair_gene_mechanism_summary',summary),('pair_mechanism_category_summary',catsummary),('representative_cgp_gene_mechanism_cases',cases),('mechanism_category_rules',category_descriptions())]:df.to_csv(OUT/(name+'.csv'),index=False)
(OUT/'provenance.json').write_text(json.dumps({'old_gene_universe':len(oldnames),'current_gene_universe':len(names),'missing_genes':sorted(set(oldnames)-set(names)),'aggregation':'encode each ORF/CRISPR entity, mean by gene symbol, L2 normalize','topk':50},indent=2));print('mechanism complete',len(names),len(pairs))
