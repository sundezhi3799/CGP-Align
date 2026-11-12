import os
os.environ.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4')
import json
from pathlib import Path
import torch,numpy as np
from transformers import AutoModel,AutoTokenizer
R=Path('/data3/sdz/projects/CGP-Align/revisions_R1_4_20261110/features')
sets=json.loads((R/'smiles_sets_corrected.json').read_text())
for method,mid in [('chemberta','DeepChem/ChemBERTa-77M-MTR'),('molformer_xl','ibm/MoLFormer-XL-both-10pct')]:
 z=np.load(R/(method+'.npz'));lookup=dict(zip(z['smiles'],z['features']))
 missing=sorted(set(s for seq in sets.values() for s in seq)-set(lookup));torch.manual_seed(41)
 tok=AutoTokenizer.from_pretrained(mid,local_files_only=True,trust_remote_code=True)
 extra={'deterministic_eval':True} if method=='molformer_xl' else {}
 model=AutoModel.from_pretrained(mid,local_files_only=True,trust_remote_code=True,**extra).cuda().eval()
 with torch.no_grad():
  t=tok(missing,padding=True,truncation=True,max_length=256,return_tensors='pt');t={k:v.cuda() for k,v in t.items()}
  h=model(**t).last_hidden_state;m=t['attention_mask'].unsqueeze(-1).to(h.dtype);x=((h*m).sum(1)/m.sum(1).clamp_min(1)).cpu().numpy().astype('float32')
 lookup.update(zip(missing,x));np.savez_compressed(R/(method+'_additional_canonical.npz'),smiles=np.asarray(missing),features=x)
 for key,smiles in sets.items():
  if key.startswith('Endocrine'):np.save(R/(key+'_'+method+'.npy'),np.vstack([lookup[s] for s in smiles]))
 print(method,'additional',len(missing),flush=True);del model;torch.cuda.empty_cache()
(R/'smiles_sets.json').write_text(json.dumps(sets))
(R/'canonicalization_correction.json').write_text(json.dumps({'five_endpoint_curation':'Original benchmark loader under RDKit 2025.09.6; input order and samples preserved. Initial RDKit 2023.09.6 exports excluded.','tox30_and_prism_unchanged':True},indent=2))
