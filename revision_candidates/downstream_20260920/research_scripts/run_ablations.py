import json,os,sys,time,subprocess
from pathlib import Path
ROOT=Path('/data3/sdz/cgp_align_final_arch_code_20260920')
sys.path.insert(0,str(ROOT/'tools'))
from run_strict_seed import command_from_config
OUT=Path('/data3/sdz/cgp_align_downstream_20260920/ablations')
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 state={'status':'running','started':time.time(),'jobs':[]}; children=[]
 def save():
  tmp=OUT/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(OUT/'status.json')
 for variant in ('no_source_indicator','no_pretraining'):
  for seed in (31,37,41):
   cfg=json.loads((Path('/data3/sdz/cgp_align_final_architecture_20260920')/f'seed{seed}/joint/configuration.json').read_text())
   d=OUT/variant/f'seed{seed}';d.mkdir(parents=True)
   cfg.update(run_name=f'{variant}_seed{seed}',checkpoint_dir=str(d/'checkpoints'),log_dir=str(d/'logs'),output_dir=str(d/'eval'),resume_checkpoint=None)
   if variant=='no_source_indicator':cfg['disable_profile_source_embedding']=True
   else:
    for key in ('init_compound_checkpoint','init_gene_checkpoint','init_orf_gene_checkpoint','init_crispr_gene_checkpoint'):cfg[key]=None
   cmd=command_from_config('train_cgp_align_replicate',cfg)
   (d/'configuration.json').write_text(json.dumps(cfg,indent=2))
   gpu=len(children);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='')
   with (d/'train.log').open('w') as log:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   j={'variant':variant,'seed':seed,'gpu':gpu,'pid':p.pid,'directory':str(d),'command':cmd,'status':'running'};state['jobs'].append(j);children.append((p,j));save()
 while any(p.poll() is None for p,j in children):
  for p,j in children:
   rc=p.poll()
   if rc is not None:j.update(exit_code=rc,status='complete' if rc==0 else 'failed')
  save();time.sleep(15)
 for p,j in children:j.update(exit_code=p.returncode,status='complete' if p.returncode==0 else 'failed')
 state.update(status='complete' if all(p.returncode==0 for p,j in children) else 'failed',finished=time.time());save()
if __name__=='__main__':main()
