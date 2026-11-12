import os,sys,json,time,hashlib,subprocess,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np,pandas as pd
ROOT=Path('/data3/sdz/projects/CGP-Align/revisions_R1_4_20261110')
CODE=Path('/data3/sdz/cgp_align_final_arch_code_20260920')
sys.path[:0]=[str(CODE/'tools'),str(CODE/'scripts'),str(CODE)]
from run_strict_seed import command_from_config
CONFIG_LOCK=threading.Lock()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
out=ROOT/'matched';out.mkdir(exist_ok=True)
def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2))
smiles=json.loads((ROOT/'features/fig3_smiles.json').read_text());x=np.load(ROOT/'nyan/compound_structure_features.npy')
def worker(seed):
 job=out/f'seed{seed}';job.mkdir(exist_ok=True);prepared=job/'data';prepared.mkdir(exist_ok=True)
 if (job/'complete.json').exists():return seed
 prior=Path('/data3/sdz/cgp_align_matched_20260919')/f'seed{seed}'
 for p in (prior/'data').iterdir():
  if p.name=='audit.json':continue
  dest=prepared/p.name
  if not dest.exists():dest.symlink_to(p.resolve())
 new=prepared/'nyan';new.mkdir(exist_ok=True)
 for p in (prior/'data/morgan').iterdir():
  if p.name=='compound_structure_features.npy':continue
  dest=new/p.name
  if not dest.exists():dest.symlink_to(p.resolve())
 table=pd.read_parquet(new/'compounds.parquet');assert table.canonical_smiles.tolist()==smiles
 feature=new/'compound_structure_features.npy'
 if not feature.exists():feature.symlink_to(ROOT/'nyan/compound_structure_features.npy')
 audit=json.loads((prior/'data/audit.json').read_text());audit['methods']['nyan']={'feature_sha256':sha(feature),'shape':list(x.shape),'source':'Official pretrained frozen NYAN posterior mean'}
 (prepared/'audit.json').write_text(json.dumps(audit,indent=2))
 cfg=json.loads((prior/'molformer_xl_recipe.json').read_text())
 cfg.update(structure_dim=64,data_dir=str(new),intrinsic_split_path=str(new/'splits_intrinsic_entity.json'),checkpoint_dir=str(job/'checkpoints'),log_dir=str(job/'logs'),run_name=f'R1_4_nyan_seed{seed}',device='cuda:0')
 assert cfg['epochs']==160 and cfg['batch_size']==512 and cfg['lambda_protein_profile']==0 and cfg['selection_metric']=='val_compound_to_profile_Recall@10'
 with CONFIG_LOCK:cmd=command_from_config('train_cgp_align_base_intrinsic',cfg)
 (job/'recipe.json').write_text(json.dumps(cfg,indent=2));(job/'command.json').write_text(json.dumps(cmd,indent=2))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTORCH_CUDA_ALLOC_CONF='')
 logpath=job/'train.log'
 if not (logpath.exists() and 'Saved checkpoint to ' in logpath.read_text()):
  with logpath.open('x') as log:subprocess.run(cmd,env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,check=True)
 checkpoint=job/'checkpoints'/cfg['run_name']/'best_model.pt'
 cmd=[sys.executable,str(ROOT/'eval_nyan_matched.py'),'--prepared',str(prepared),'--checkpoint',str(checkpoint),'--method','nyan','--output',str(job/'evaluation'),'--aggregation','mean_encoded','--replicate-data',f'/data3/sdz/cgp_align_strict_20260918/seed{seed}/data/compound']
 with (job/'evaluate.log').open('x') as log:subprocess.run(cmd,env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,check=True)
 (job/'complete.json').write_text(json.dumps({'seed':seed,'checkpoint_sha256':sha(checkpoint)}));print('SEED_COMPLETE',seed,flush=True)
 return seed
worker(37)
