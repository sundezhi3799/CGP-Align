from pathlib import Path
import json,sys,shutil,hashlib
from types import SimpleNamespace
import numpy as np,pandas as pd
ROOT=Path.cwd();sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import evaluate_prism_high_confidence_hits as hits
import evaluate_prism_u2os_single_cell as u2os
import evaluate_prism_continuous_metrics as cont
R=ROOT/'reproducibility_audits/downstream_release_20260920';OUT=R/'figure6';OUT.mkdir(exist_ok=True)
old=ROOT/'output/cgp_align/paper/manuscript_cgp_align_profile/prism_response_profile_retrieval_v1'
z=np.load(old/'prism_functional_retrieval_matrices.npz',allow_pickle=True);payload={k:z[k] for k in z.files};meta=pd.read_csv(old/'prism_cgp_overlap_compounds.csv');assert list(meta.inchikey)==list(payload['inchikey']);assert np.array_equal(meta._cgp_row,payload['cgp_rows'])
v=np.load(R/'embeddings/seed41/prism.npy');v=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8);sim=v@v.T;np.fill_diagonal(sim,-np.inf);payload['cgp_similarity']=sim
np.savez_compressed(OUT/'prism_functional_retrieval_matrices.npz',**payload)
for name in ['prism_cgp_overlap_compounds.csv','prism_official_nyan_latents_cache.npz']:shutil.copy2(old/name,OUT/name)
features=np.load(ROOT/'data/cgp_cpg_full_motive_edges/compound_structure_features.npy',mmap_mode='r')
for module in [hits,u2os,cont]:
 module.CGPBundle=lambda *args:SimpleNamespace(structure_features=features)
 sys.argv=['evaluate','--input_dir',str(OUT)];module.main();print(module.__name__,'complete',flush=True)
(OUT/'response_provenance.json').write_text(json.dumps({'historical_response_matrix_sha256':hashlib.sha256((old/'prism_functional_retrieval_matrices.npz').read_bytes()).hexdigest(),'new_embeddings_sha256':hashlib.sha256((R/'embeddings/seed41/prism.npy').read_bytes()).hexdigest(),'n':len(meta),'cell_lines':len(payload['cell_lines']),'B_response':'pan-cancer correlation across 578 cell lines','C_response':'U2OS ACH-000364 activity'},indent=2));(OUT/'RETRIEVAL_COMPLETE').write_text('complete')
