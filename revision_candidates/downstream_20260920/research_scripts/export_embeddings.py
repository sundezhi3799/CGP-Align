import json,sys,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path('/data3/sdz/cgp_align_final_arch_code_20260920');sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import train_cgp_align_replicate as cgp
from eval_cgp_align_crossmodal_bridge import namespace_from_config
OUT=Path('/data3/sdz/cgp_align_downstream_20260920/embeddings');OUT.mkdir(exist_ok=True)
sets=json.loads((ROOT/'smiles_sets.json').read_text());device=torch.device('cuda:0');torch.set_num_threads(4)
for seed in (31,37,41):
 out=OUT/f'seed{seed}';out.mkdir(exist_ok=True)
 path=Path('/data3/sdz/cgp_align_final_architecture_20260920')/f'seed{seed}/joint/checkpoints/final_arch_seed{seed}_joint/best_model.pt'
 payload=torch.load(path,map_location='cpu',weights_only=False);args=namespace_from_config(payload['config']);args.smoke_test=False
 for k in ('init_compound_checkpoint','init_gene_checkpoint','init_orf_gene_checkpoint','init_crispr_gene_checkpoint'):setattr(args,k,None)
 data=cgp.build_data(args);model,_=cgp.build_model(args,data,device);model.load_state_dict(payload['model_state_dict'],strict=True);model.to(device).eval()
 rows=np.unique(np.concatenate([data['compound'][k+'_entities'] for k in ('train','val','test')]))
 z=cgp.encode_compounds(model,data['compound']['graph_store'],rows,device,512);np.save(out/'compound.npy',z);data['compound']['entities'].iloc[rows].to_parquet(out/'compound_entities.parquet',index=False)
 grows=np.unique(np.concatenate([data['gene'][k+'_entities'] for k in ('train','val','test')]))
 g=data['gene'];z=cgp.encode_gene_entities(model,g['entities'],g['protein_embeddings'],g['modality_ids'],grows,device,1024)
 np.save(out/'gene.npy',z);g['entities'].iloc[grows].to_parquet(out/'gene_entities.parquet',index=False)
 for label,smiles in sets.items():
  if seed!=41:continue
  graph=cgp.GraphStore(smiles);v=cgp.encode_compounds(model,graph,np.arange(len(smiles)),device,512);np.save(out/(label+'.npy'),v);print(seed,label,v.shape,flush=True)
 meta={'seed':seed,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'epoch':payload['epoch'],'compound_rows':len(rows),'gene_rows':len(grows),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}}
 (out/'provenance.json').write_text(json.dumps(meta,indent=2));print('complete',seed,flush=True)
 del model,data;torch.cuda.empty_cache()
