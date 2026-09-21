"""Portable orchestration of the numerical implementations used for Figures 4–6."""
from pathlib import Path
import json,sys,hashlib,shutil,ast,math
from types import SimpleNamespace
from typing import *
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]

def embeddings(a):
 import torch
 import train_cgp_align_replicate as cgp
 from eval_raw3180_branch_retrieval_from_checkpoints import load_model_and_data
 torch.set_num_threads(4);device=torch.device(a.device)
 sets=json.loads((a.inputs/'smiles_sets.json').read_text())
 for seed in a.seeds:
  out=a.embeddings/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
  ck=a.artifacts/f'models/seed{seed}/joint/best_model.pt'
  overrides=SimpleNamespace(compound_data_dir=a.artifacts/f'data/seed{seed}/compound',gene_data_dir=a.artifacts/f'data/seed{seed}/gene',gene_protein_embedding_dir=a.artifacts/'protein')
  model,data,payload,_=load_model_and_data(ck,device,512,overrides)
  rows=np.unique(np.concatenate([data['compound'][k+'_entities'] for k in ('train','val','test')]))
  np.save(out/'compound.npy',cgp.encode_compounds(model,data['compound']['graph_store'],rows,device,512))
  data['compound']['entities'].iloc[rows].to_parquet(out/'compound_entities.parquet',index=False)
  g=data['gene'];rows=np.unique(np.concatenate([g[k+'_entities'] for k in ('train','val','test')]))
  np.save(out/'gene.npy',cgp.encode_gene_entities(model,g['entities'],g['protein_embeddings'],g['modality_ids'],rows,device,1024))
  g['entities'].iloc[rows].to_parquet(out/'gene_entities.parquet',index=False)
  g['entities'].iloc[rows].to_csv(out/'gene_metadata.csv',index=False)
  if seed==41:
   for label,smiles in sets.items():np.save(out/(label+'.npy'),cgp.encode_compounds(model,cgp.GraphStore(smiles),np.arange(len(smiles)),device,512))
  (out/'provenance.json').write_text(json.dumps({'seed':seed,'epoch':payload['epoch'],'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest()},indent=2))
  print('Encoded',seed,flush=True)
  del model,data;torch.cuda.empty_cache()

def relations(a):
 import eval_cgp_align_target_enrichment_gene_union as mod
 import eval_raw3180_gene_union_latent_link_classifier as classifier
 out=a.output/'figure4';out.mkdir(exist_ok=True);records=[]
 for seed in a.seeds:
  d=a.embeddings/f'seed{seed}';c=pd.read_parquet(d/'compound_entities.parquet');g=pd.read_parquet(d/'gene_entities.parquet');zc=np.load(d/'compound.npy');zg=np.load(d/'gene.npy')
  symbols,lookup,groups,_=mod.gene_symbol_groups(g,np.arange(len(g)))
  path=out/f'seed{seed}_union_scores.npy'
  scores=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(c),len(symbols)))
  for start in range(0,len(c),2048):scores[start:start+2048]=mod.collapse_entity_scores_to_symbols(zc[start:start+2048]@zg.T,groups)
  scores.flush()
  for graph in ['broad','strict']:
   for source in ['dgidb','drugrep','hetionet','openbiolink','pharmebinet']:
    targets=mod.load_targets(a.inputs/f'relations/subsets/{graph}_{source}.parquet',False,False);cp,gp,meta=mod.build_symbol_positive_maps(targets,c,np.arange(len(c)),lookup)
    m={'C2G':mod.multi_positive_rank_metrics(scores,cp,[10,50,100]),'G2C':mod.multi_positive_rank_metrics(scores.T,gp,[10,50,100])}
    rec=mod.flatten(f'final_seed{seed}','gene_union',m,meta,json.loads((d/'provenance.json').read_text())['epoch']);rec.update(seed=seed,graph=graph,source=source);records.append(rec)
    pd.DataFrame(records).to_csv(out/'enrichment_runs.csv',index=False);print('Enrichment',seed,graph,source,flush=True)
  del scores
 def cached(checkpoint,device,batch,modalities):
  d=a.embeddings/f'seed{int(str(checkpoint))}';c=pd.read_parquet(d/'compound_entities.parquet');g=pd.read_parquet(d/'gene_entities.parquet');zc=np.load(d/'compound.npy');zg=np.load(d/'gene.npy');symbols=g.gene_symbol.fillna('').str.upper();names=sorted(set(symbols)-{''});groups={s:[] for s in names}
  for i,s in enumerate(symbols):
   if s:groups[s].append(i)
  z=classifier.l2_normalize(np.vstack([zg[groups[s]].mean(0) for s in names]));return {'epoch':json.loads((d/'provenance.json').read_text())['epoch']},c,np.arange(len(c)),zc,names,z
 classifier.load_model_gene_union_embeddings=cached
 for graph in ['broad','strict']:
  sys.argv=['classifier','--rows',','.join(f'final_seed{s}:{s}' for s in a.seeds),'--target_edges',str(a.inputs/f'relations/{graph}_edges.parquet'),'--output_dir',str(out/graph),'--source','all','--split_modes','','--holdout_sources','dgidb,drugrep,hetionet,openbiolink,pharmebinet','--negative_ratio','5','--max_train_pos','10000','--max_eval_pos','3000','--seed','13','--device','cpu'];classifier.main()
 if 41 in a.seeds:
  from relation_pca import run
  run(a)

def toxicity(a):
 import benchmark_toxric_molecular_representations as bench
 import eval_sr5_scaffold_split_representations as scaffold
 from scipy import sparse
 from sklearn.ensemble import ExtraTreesClassifier
 from sklearn.metrics import average_precision_score,roc_auc_score
 from sklearn.model_selection import StratifiedKFold
 out=a.output/'figure5';out.mkdir(exist_ok=True);inp=a.inputs/'toxicity'
 sets=json.loads((a.inputs/'smiles_sets.json').read_text());vectors={}
 for key in ['tox_main','tox_atad5']:
  z=np.load(a.embeddings/f'seed41/{key}.npy')
  for s,v in zip(sets[key],z):vectors[bench.canonical_smiles(s)]=v
 indexes={k:scaffold.smiles_index(inp/k/'smiles.json') for k in ['main','atad5']}
 nyan_idx,nyan_x=scaffold.load_nyan(inp/'scaffold_nyan.npz')
 folds=[];scfolds=[]
 for task,df in bench.load_tasks(inp/'tasks',','.join(scaffold.SR5)):
  y=df.label.to_numpy(int);key='atad5' if task.endswith('SR-ATAD5') else 'main'
  for rep in scaffold.REPS:
   if rep=='CGP_sep_k32':x=np.vstack([vectors[s] for s in df.smiles])
   elif rep=='NYAN_latent':x=np.array(nyan_x[[nyan_idx[s] for s in df.smiles]],dtype=np.float32)
   else:
    matrix=np.load(inp/key/(rep+'.npy'),mmap_mode='r');x=np.array(matrix[[indexes[key][s] for s in df.smiles]])
   sp=bench.cv_splits(x,y,5,41);rr,_=bench.evaluate_one(x,y,sp,task,rep,'extratrees',41,160,2000);folds+=rr
   sx=np.array(x,dtype=np.float32)
   scfolds+=scaffold.eval_one(sx,y,df.smiles.map(scaffold.scaffold).to_numpy(),task,rep,5,41,160)
   pd.DataFrame(folds).to_csv(out/'random_all_folds.csv',index=False);pd.DataFrame(scfolds).to_csv(out/'scaffold_all_folds.csv',index=False);print('Toxicity',task,rep,flush=True)
 alerts(a)

def alerts(a):
 import benchmark_toxric_molecular_representations as bench
 import eval_sr5_scaffold_split_representations as scaffold
 from scipy import sparse
 from sklearn.ensemble import ExtraTreesClassifier
 from sklearn.metrics import average_precision_score,roc_auc_score
 from sklearn.model_selection import StratifiedKFold
 out=a.output/'figure5';out.mkdir(exist_ok=True);inp=a.inputs/'toxicity'
 sets=json.loads((a.inputs/'smiles_sets.json').read_text());vectors={}
 for key in ['tox_main','tox_atad5']:
  for s,v in zip(sets[key],np.load(a.embeddings/f'seed41/{key}.npy')):vectors[bench.canonical_smiles(s)]=v
 tie_order=np.load(inp/'alerts/original_alert_tie_order.npz',allow_pickle=False)
 # Load exactly the two numerical readout helpers from the archived analysis.
 tree=ast.parse((ROOT/'scripts/run_cgp_expert_alert_prioritization_extratrees_repeats.py').read_text(encoding='utf-8-sig'))
 ns=dict(np=np,pd=pd,ExtraTreesClassifier=ExtraTreesClassifier,StratifiedKFold=StratifiedKFold,roc_auc_score=roc_auc_score,average_precision_score=average_precision_score,Any=Any,Tuple=Tuple,Dict=Dict,List=List)
 for node in tree.body:
  if isinstance(node,ast.FunctionDef) and node.name in ['safe_auroc','oof_extratrees_scores']:exec(compile(ast.Module(body=[node],type_ignores=[]),'<archived readout>','exec'),ns)
 cnt=pd.read_csv(inp/'alerts/compound_expert_alert_counts.csv');idx={bench.canonical_smiles(s):i for i,s in enumerate(cnt.smiles)};alerts=sparse.load_npz(inp/'alerts/compound_expert_alert_matrix.npz').toarray().astype(np.float32);records=[]
 for task,df in bench.load_tasks(inp/'tasks',','.join(scaffold.SR5)):
  df=df[df.smiles.isin(idx)].reset_index(drop=True);y=df.label.to_numpy(int);z=np.vstack([vectors[s] for s in df.smiles]);features=alerts[[idx[s] for s in df.smiles]]
  for seed in [13,41,97,123,2026]:
   for method,x in [('structure_alert_extratrees',features),('cgp_latent_extratrees',z),('cgp_plus_alert_extratrees',np.hstack([z,features]))]:
    score,_=ns['oof_extratrees_scores'](x,y,5,seed,240,4)
    key=f'{task.split("_")[-1]}_{seed}_{method}'
    if not np.array_equal(df.smiles.to_numpy(str),tie_order[key+'_smiles']):raise ValueError('Alert tie-order cohort differs: '+key)
    # Preserve the original NumPy execution's ordering ONLY within equal-score
    # groups. The primary key is the newly fitted prediction, never a label.
    order=np.lexsort((tie_order[key+'_priority'],-score))
    records.append(dict(task=task,seed=seed,method=method,n=len(y),positive=int(y.sum()),auprc=average_precision_score(y,score),auroc=roc_auc_score(y,score),lift_at_top50=float(y[order[:50]].mean()/y.mean())))
    np.savez_compressed(out/(key+'_oof.npz'),smiles=df.smiles.to_numpy(str),labels=y,scores=score,ranking=order)
   pd.DataFrame(records).to_csv(out/'alerts_metrics.csv',index=False);print('Alerts',task,seed,flush=True)

def prism(a):
 import evaluate_prism_high_confidence_hits as hits
 import evaluate_prism_u2os_single_cell as u2os
 import evaluate_prism_continuous_metrics as cont
 out=a.output/'figure6';out.mkdir(exist_ok=True);inp=a.inputs/'prism'
 z=np.load(inp/'prism_functional_retrieval_matrices.npz',allow_pickle=True);payload={k:z[k] for k in z.files};meta=pd.read_csv(inp/'prism_cgp_overlap_compounds.csv');assert list(meta.inchikey)==list(payload['inchikey'])
 v=np.load(a.embeddings/'seed41/prism.npy');v=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8);sim=v@v.T;np.fill_diagonal(sim,-np.inf);payload['cgp_similarity']=sim
 # Compact descriptor subset retains exact values and row order; remap the index only.
 payload['cgp_rows']=np.arange(len(meta));np.savez_compressed(out/'prism_functional_retrieval_matrices.npz',**payload)
 for name in ['prism_cgp_overlap_compounds.csv','prism_official_nyan_latents_cache.npz']:shutil.copy2(inp/name,out/name)
 features=np.load(inp/'morgan_rdkit_features.npy')
 for module in [hits,u2os,cont]:
  module.CGPBundle=lambda *args:SimpleNamespace(structure_features=features)
  sys.argv=['evaluate','--input_dir',str(out)];module.main();print(module.__name__,'complete',flush=True)
 # Preserve historical numerical method keys in raw output; display labels identify
 # the actual Morgan-2048 + 11-descriptor baseline in plots and documentation.
 mechanism(a,out,meta)
 from evaluate_prism_response_similarity import evaluate as response_similarity
 response_similarity(out,inp/'morgan_rdkit_features.npy',out/'u2os_similarity')

def mechanism(a,out,meta):
 from scipy.stats import hypergeom
 tree=ast.parse((ROOT/'scripts/run_prism_gene_mechanism_interpretation.py').read_text(encoding='utf-8-sig'));keep=[]
 for node in tree.body:
  if isinstance(node,(ast.Assign,ast.AnnAssign)):
   names=[t.id for t in node.targets if isinstance(t,ast.Name)] if isinstance(node,ast.Assign) else [getattr(node.target,'id','')]
   if any(n.startswith('CASE_') or n in ['GENE_TOPK','METHOD_ORDER','MECHANISM_RULES'] for n in names):keep.append(node)
  if isinstance(node,ast.FunctionDef) and node.name not in ['main','make_plot','write_report']:keep.append(node)
 dest=out/'mechanism';dest.mkdir(exist_ok=True)
 ns=dict(np=np,pd=pd,math=math,hypergeom=hypergeom,Path=Path,PRISM_DIR=out,OUT=dest,Any=Any,Dict=Dict,List=List,Tuple=Tuple,Set=Set,Optional=Optional,Iterable=Iterable)
 exec(compile(ast.Module(body=keep,type_ignores=[]),'<archived mechanism>','exec'),ns)
 z=np.load(a.embeddings/'seed41/gene.npy');e=pd.read_csv(a.embeddings/'seed41/gene_metadata.csv');symbols=e.gene_symbol.fillna('').str.upper();idx={s:np.flatnonzero(symbols.to_numpy()==s) for s in set(symbols)-{''}};names=np.array([s for s in json.loads((a.inputs/'prism/gene_universe.json').read_text()) if s in idx]);v=ns['l2_normalize'](np.vstack([z[idx[s]].mean(0) for s in names]));c=ns['l2_normalize'](np.load(a.embeddings/'seed41/prism.npy'));scores=c@v.T;top=ns['topk_indices'](scores,100);pairs=ns['build_pair_sets']();metrics,categories,cases=ns['annotate_pair_rows'](pairs,meta,top,names,scores);summary,catsummary=ns['summarize'](metrics,categories)
 for name,df in [('pair_gene_mechanism_metrics',metrics),('pair_mechanism_category_hits',categories),('pair_gene_mechanism_summary',summary),('pair_mechanism_category_summary',catsummary),('representative_cgp_gene_mechanism_cases',cases),('mechanism_category_rules',ns['category_descriptions']())]:df.to_csv(dest/(name+'.csv'),index=False)
 # Figure 6e uses individually reviewed protein-function assignments.
 annotations=pd.read_csv(ROOT/'assets/figure6/shared_gene_annotations.csv')
 q=int(np.flatnonzero(meta.inchikey.to_numpy()=='QULDDKSCVCJTPV-UHFFFAOYSA-N')[0])
 r=int(np.flatnonzero(meta.inchikey.to_numpy()=='JRNJNYBQQYBCLE-UHFFFAOYSA-N')[0])
 q_genes=names[top[q,:50]];r_genes=names[top[r,:50]]
 shared=set(q_genes)&set(r_genes)
 if shared!=set(annotations.gene):raise ValueError('BIIB021/R547 shared-gene set differs from the annotated Figure 6e inputs.')
 annotations['rank_BIIB021']=annotations.gene.map({g:i+1 for i,g in enumerate(q_genes)})
 annotations['rank_R547']=annotations.gene.map({g:i+1 for i,g in enumerate(r_genes)})
 annotations.to_csv(dest/'figure6e_shared_gene_annotations.csv',index=False)
 annotations.groupby('display_function',sort=False).size().rename('n').reset_index().to_csv(dest/'figure6e_function_counts.csv',index=False)
