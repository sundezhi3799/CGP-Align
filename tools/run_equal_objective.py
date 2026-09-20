"""Train the selected equal-three-term objective from completed strict branch initializers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from run_strict_seed import command_from_config
ROOT = Path(__file__).resolve().parents[1]

def write(p, obj):
    t=p.with_suffix('.tmp'); t.write_text(json.dumps(obj,indent=2,default=str)+'\n'); t.replace(p)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--strict-root',type=Path,required=True)
    p.add_argument('--matched-root',type=Path,required=True)
    p.add_argument('--encoded-baselines',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',default='0,1,2')
    a=p.parse_args()
    gpus=a.gpus.split(',')
    if len(gpus)!=3 or len(set(gpus))!=3: raise ValueError('Three distinct GPUs required')
    a.output.mkdir(parents=True,exist_ok=False)
    state=dict(status='validating',started=time.time(),controller_pid=os.getpid(),objective='compound + ORF + CRISPR',jobs=[])
    def save(): write(a.output/'status.json',state)
    save()
    children=[]
    try:
        for seed,gpu in zip((31,37,41),gpus):
            old=a.strict_root/f'seed{seed}'
            for kind in ('compound','gene'):
                audit=json.loads((old/f'data/{kind}/profile_correction_audit.json').read_text())
                assert audit['seed']==seed and audit['fit_val_overlap']==audit['fit_test_overlap']==0
            cfg=json.loads((old/('joint_ampfix01' if seed==31 else 'joint')/'configuration.json').read_text())
            inputs={}
            for key in ('init_compound_checkpoint','init_orf_gene_checkpoint','init_crispr_gene_checkpoint'):
                path=Path(cfg[key]); assert path.is_file() and old/'branches' in path.parents
                inputs[key]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            job=a.output/f'seed{seed}/joint'; job.mkdir(parents=True)
            cfg.update(run_name=f'cgp_align_equal_three_term_seed{seed}',gene_branch_loss_reduction='sum',
                       compound_loss_weight=1.0,gene_loss_weight=1.0,anchor_weight_stage2=0.0,anchor_weight_stage3=0.0,
                       cg_teacher_weight=0.0,profile_structure_weight=0.0,split_gene_distribution_alignment_weight=0.0,
                       resume_checkpoint=None,eval_only=False,train_only=False,device='cuda:0',
                       checkpoint_dir=str(job/'checkpoints'),log_dir=str(job/'logs'),output_dir=str(job/'eval'))
            cfg.pop('init_summary',None)
            command=command_from_config('train_cgp_align_replicate',cfg)
            write(job/'configuration.json',cfg)
            write(job/'initializer_provenance.json',inputs)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTORCH_CUDA_ALLOC_CONF='',PYTHONUNBUFFERED='1')
            log=(job/'train.log').open('w')
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            log.close(); children.append((child,seed,env,job))
            state['jobs'].append(dict(seed=seed,gpu=gpu,pid=child.pid,command=command));save()
        state['status']='joint_training';save()
        for child,seed,env,job in children:
            rc=child.wait()
            if rc: raise RuntimeError(f'seed{seed} training failed: {rc}')
        state['status']='matched_evaluation';save()
        for child,seed,env,job in children:
            encoded=a.output/f'encoded/seed{seed}';encoded.mkdir(parents=True)
            for method in ('molformer_xl','chemberta','morgan'):
                source=a.encoded_baselines/f'seed{seed}/{method}'
                assert (source/'metrics.json').is_file()
                (encoded/method).symlink_to(source.resolve(),target_is_directory=True)
            checkpoint=job/f'checkpoints/cgp_align_equal_three_term_seed{seed}/best_model.pt'
            command=[sys.executable,'-u',str(ROOT/'tools/eval_matched_benchmark.py'),'--prepared',str(a.matched_root/f'seed{seed}/data'),'--checkpoint',str(checkpoint),'--method','cgp','--aggregation','mean_encoded','--replicate-data',str(a.strict_root/f'seed{seed}/data/compound'),'--output',str(encoded/'cgp')]
            state['current_command']=command;save()
            with (job/'matched_eval.log').open('w') as log: subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        state['status']='exporting_figure_sources';save()
        command=[sys.executable,'-u',str(ROOT/'tools/export_strict_figure3.py'),'--strict-root',str(a.output),'--matched-root',str(a.matched_root),'--encoded-root',str(a.output/'encoded'),'--output',str(a.output/'figure3_sources'),'--joint-directory','joint','--metric-pattern','cgp_align_equal_three_term_seed{seed}_test_metrics.json']
        state['current_command']=command;save()
        with (a.output/'figure_export.log').open('w') as log: subprocess.run(command,cwd=ROOT,env=children[0][2],stdout=log,stderr=subprocess.STDOUT,check=True)
        state.update(status='sources_ready',finished=time.time());save()
    except BaseException as e:
        for child,*_ in children:
            if child.poll() is None: child.terminate()
        state.update(status='failed',error=repr(e));save();raise
if __name__=='__main__': main()
