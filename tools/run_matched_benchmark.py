"""Run three strict seed cohorts and nine historical-recipe molecular baselines."""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
from run_strict_seed import command_from_config
from prepare_matched_benchmark import METHODS, write
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--research-root', type=Path, required=True)
    p.add_argument('--strict-root', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--gpus', default='3,4,5')
    p.add_argument('--worker-seed', type=int, choices=(31,37,41))
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--prepared-run', action='store_true', help='Start training after preparation and independent smoke validation.')
    a = p.parse_args()
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTORCH_CUDA_ALLOC_CONF='', PYTHONUNBUFFERED='1')
    job = a.run_root.resolve()
    children = []
    if a.worker_seed:
        job = job / f'seed{a.worker_seed}'
    elif a.prepared_run:
        previous = json.loads((job / 'status.json').read_text())
        assert previous['status'] == 'prepared'
        with (job / 'training_controller.lock').open('x') as lock: lock.write(str(os.getpid()))
        write(job / 'preparation_status.json', previous)
        for seed in (31,37,41):
            assert json.loads((job / f'seed{seed}/data/audit.json').read_text())['seed'] == seed
    else:
        job.mkdir(parents=True, exist_ok=False)
    state = dict(status='starting', pid=os.getpid(), started=time.time(), code_directory=str(ROOT), commands=[])
    def save():
        write(job / 'status.tmp', state); (job / 'status.tmp').replace(job / 'status.json')
    def run(command, log):
        state['commands'].append(command); save()
        with log.open('w') as f: subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    try:
        save()
        if a.worker_seed:
            seed = a.worker_seed
            env['CUDA_VISIBLE_DEVICES'] = a.gpus
            prepared = job / 'data'
            primary = a.strict_root / f'seed{seed}' / ('joint_ampfix01' if seed == 31 else 'joint') / 'checkpoints' / f'strict_seed{seed}_joint/best_model.pt'
            state['status'] = 'evaluating_cgp'; save()
            def evaluate(method, checkpoint):
                run([sys.executable, '-u', str(ROOT / 'tools/eval_matched_benchmark.py'), '--prepared', str(prepared),
                     '--checkpoint', str(checkpoint), '--method', method, '--output', str(job / 'eval' / method)], job / (method + '_eval.log'))
            evaluate('cgp', primary)
            for method in METHODS:
                cfg = json.loads((ROOT / f'manifests/checkpoint_configs/baseline_{method}_historical_seed13.json').read_text())
                cfg.update(seed=seed, data_dir=str(prepared / method), intrinsic_split_path=str(prepared / method / 'splits_intrinsic_entity.json'),
                           protein_embedding_dir=str(a.research_root / 'protein_embeddings/cgp_cpg_full_esm'),
                           checkpoint_dir=str(job / 'checkpoints'), log_dir=str(job / 'logs'),
                           run_name=f'strict_matched_{method}_seed{seed}', device='cuda:0', smoke_test=False)
                assert cfg['lambda_protein_profile'] == 0 and cfg['epochs'] == 160
                command = command_from_config('train_cgp_align_base_intrinsic', cfg)
                write(job / (method + '_recipe.json'), cfg)
                state.update(status='baseline_training', method=method); save()
                run(command, job / (method + '_train.log'))
                state['status'] = 'baseline_evaluation'; save()
                evaluate(method, Path(cfg['checkpoint_dir']) / cfg['run_name'] / 'best_model.pt')
        else:
            gpus = a.gpus.split(',')
            assert len(gpus) == len(set(gpus)) == 3
            if not a.prepared_run:
                state['status'] = 'preparing'; save()
                for seed in (31,37,41):
                    seed_job = job / f'seed{seed}'; seed_job.mkdir()
                    command = [sys.executable, '-u', str(ROOT / 'tools/prepare_matched_benchmark.py'),
                               '--strict-root', str(a.strict_root), '--feature-root', str(a.research_root / 'output/cgp_align/raw3180_cellclip_style_baseline_data_20260604'),
                               '--output', str(seed_job / 'data'), '--seed', str(seed)]
                    state['commands'].append(command)
                    with (seed_job / 'prepare.log').open('w') as f:
                        children.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True))
                state['preparation_pids'] = [p.pid for p in children]; save()
            def wait_all():
                while True:
                    codes = [p.poll() for p in children]
                    if any(c not in (None,0) for c in codes): raise RuntimeError('Child failed: ' + str(codes))
                    if all(c == 0 for c in codes): return
                    time.sleep(10)
            wait_all()
            if a.prepare_only:
                state.update(status='prepared', finished=time.time()); save(); return
            children = []
            for seed, gpu in zip((31,37,41),gpus):
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--research-root', str(a.research_root),
                           '--strict-root', str(a.strict_root), '--run-root', str(a.run_root), '--worker-seed', str(seed), '--gpus', gpu]
                with (job / f'seed{seed}/worker.log').open('w') as f:
                    children.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True))
            state.update(status='workers_running', worker_pids=[p.pid for p in children]); save()
            wait_all()
        state.update(status='complete', finished=time.time()); save()
    except BaseException as error:
        for child in children:
            if child.poll() is None:
                if os.name == 'posix': os.killpg(child.pid, signal.SIGTERM)
                else: child.terminate()
        state.update(status='failed', error=repr(error)); save()
        raise


if __name__ == '__main__': main()
