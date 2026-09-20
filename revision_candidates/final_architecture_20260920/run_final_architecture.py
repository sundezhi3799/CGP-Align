"""Fresh branch pretraining then joint training matching the submission architecture."""
import json,os,time,subprocess,hashlib
from pathlib import Path
from run_strict_seed import command_from_config
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/data3/sdz/cgp_align_final_architecture_20260920')
STRICT=Path('/data3/sdz/cgp_align_strict_20260918')
OLD=Path('/data3/sdz/cgp_align_equal_three_term_20260920')
def write(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str));tmp.replace(path)
def main():
 audit=json.loads((ROOT/'equivalence_audit.json').read_text());assert audit['status']=='PASS'
 OUT.mkdir(exist_ok=False)
 state=dict(status='preflight',controller_pid=os.getpid(),started=time.time(),code_directory=str(ROOT),jobs=[])
 active=[]
 def save():write(OUT/'status.json',state)
 def launch(job,gpu):
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTORCH_CUDA_ALLOC_CONF='',PYTHONUNBUFFERED='1')
  with (Path(job['directory'])/'train.log').open('w') as log:
   child=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  job.update(pid=child.pid,gpu=gpu,status='running',started=time.time());active.append((child,job));save()
 def stage(queue,gpus):
  while queue or active:
   for child,job in list(active):
    rc=child.poll()
    if rc is not None:
     job.update(exit_code=rc,status='complete' if rc==0 else 'failed',finished=time.time());active.remove((child,job));save()
     if rc:raise RuntimeError(f"{job['name']} exited {rc}")
   used={j['gpu'] for _,j in active}
   for gpu in gpus:
    if queue and gpu not in used:launch(queue.pop(0),gpu)
   if queue or active:time.sleep(15)
 def prepare(seed,branch,cfg,module,directory):
  directory.mkdir(parents=True)
  cfg.update(run_name=f'final_arch_seed{seed}_{branch}',checkpoint_dir=str(directory/'checkpoints'),log_dir=str(directory/'logs'),output_dir=str(directory/'eval'),device='cuda:0',eval_only=False,smoke_test=False)
  command=command_from_config(module,cfg)
  write(directory/'configuration.json',cfg)
  job=dict(seed=seed,branch=branch,name=cfg['run_name'],directory=str(directory),command=command,status='queued')
  state['jobs'].append(job);save();return job
 try:
  save();queue=[]
  for seed in (31,37,41):
   for kind in ('compound','gene'):
    d=STRICT/f'seed{seed}/data/{kind}';c=json.loads((d/'profile_correction_audit.json').read_text())
    assert c['seed']==seed and c['fit_val_overlap']==c['fit_test_overlap']==0
    split=json.loads((d/f'splits_{kind}_mocop.json').read_text())['cold_'+kind]
    tr,va,te=[set(split[x]) for x in ('train','val','test')];assert not(tr&va or tr&te or va&te)
   # An explicit symlink permits later figure tools to locate this run's strict inputs.
   seedroot=OUT/f'seed{seed}';seedroot.mkdir();(seedroot/'data').symlink_to(STRICT/f'seed{seed}/data',target_is_directory=True)
   for branch in ('compound','orf','crispr'):
    cfg=json.loads((STRICT/f'seed{seed}/branches/{branch}/configuration.json').read_text())
    cfg.update(train_only=True,epochs=300)
    if branch!='compound':cfg.update(protein_hidden_dims='512',profile_hidden_dims='512')
    else:cfg['profile_hidden_dim']=512
    queue.append(prepare(seed,branch,cfg,'train_compound_mocop_replicate' if branch=='compound' else 'train_gene_mocop',seedroot/'branches'/branch))
  hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('scripts','tools','submission_model/cgp_align') for p in (ROOT/folder).glob('*.py')}
  write(OUT/'code_hashes.json',hashes);write(OUT/'equivalence_audit.json',audit)
  state['status']='branch_pretraining';save();stage(queue,list(range(6)))
  state['status']='preparing_joint';save();queue=[]
  for seed in (31,37,41):
   cfg=json.loads((OLD/f'seed{seed}/joint/configuration.json').read_text())
   cfg.update(gene_hidden_dims='512',profile_hidden_dims='512',gene_branch_loss_reduction='sum',train_only=False,resume_checkpoint=None,epochs=300)
   cfg.pop('init_summary',None)
   provenance={}
   for branch,key in [('compound','init_compound_checkpoint'),('orf','init_orf_gene_checkpoint'),('crispr','init_crispr_gene_checkpoint')]:
    p=OUT/f'seed{seed}/branches/{branch}/checkpoints/final_arch_seed{seed}_{branch}/best_model.pt';assert p.is_file()
    cfg[key]=str(p);provenance[branch]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
   directory=OUT/f'seed{seed}/joint';queue.append(prepare(seed,'joint',cfg,'train_cgp_align_replicate',directory));write(directory/'initializer_provenance.json',provenance)
  state['status']='joint_training_and_test_evaluation';save();stage(queue,[0,1,2])
  state.update(status='training_and_replicate_evaluation_complete',finished=time.time());save()
 except BaseException as e:
  for child,_ in active:
   if child.poll() is None:child.terminate()
  state.update(status='failed',error=repr(e));save();raise
if __name__=='__main__':main()
