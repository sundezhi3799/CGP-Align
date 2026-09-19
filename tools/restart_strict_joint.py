"""Restart a failed joint stage using verified completed strict seed-31 branches."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from run_strict_seed import command_from_config

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--attempt', default='joint_ampfix01')
    p.add_argument('--gpu', default='0')
    a = p.parse_args()
    if Path(a.attempt).name != a.attempt: raise ValueError('Attempt must be a directory name')
    job = a.run_root.resolve()
    status_file = job / 'status.json'
    old = json.loads(status_file.read_text())
    if old['status'] != 'failed': raise ValueError('Recovery requires a failed run')
    for kind in ('compound', 'gene'):
        audit = json.loads((job / 'data' / kind / 'profile_correction_audit.json').read_text())
        assert audit['seed'] == 31 and audit['fit_val_overlap'] == audit['fit_test_overlap'] == 0
    cfg = json.loads((job / 'joint/configuration.json').read_text())
    for branch, key in [('compound', 'init_compound_checkpoint'), ('orf', 'init_orf_gene_checkpoint'), ('crispr', 'init_crispr_gene_checkpoint')]:
        rows = []
        for line in (job / 'branches' / branch / 'train.log').read_text().splitlines():
            try: row = json.loads(line)
            except ValueError: continue
            if isinstance(row, dict): rows.append(row)
        assert max(r.get('epoch', 0) for r in rows) == 300
        assert Path(cfg[key]).is_file()
    target = job / a.attempt
    target.mkdir()  # Never overwrite a previous attempt.
    (target / 'previous_status.json').write_text(json.dumps(old, indent=2))
    cfg.update(checkpoint_dir=str(target / 'checkpoints'), log_dir=str(target / 'logs'),
               output_dir=str(target / 'eval'), resume_checkpoint=None)
    command = command_from_config('train_cgp_align_replicate', cfg)
    (target / 'configuration.json').write_text(json.dumps(cfg, indent=2))
    state = dict(status='joint_training', seed=31, controller_pid=os.getpid(),
                 started=old['started'], recovery_started=time.time(), joint_attempt=a.attempt,
                 jobs=old['jobs'], branch_exit_codes=[0, 0, 0], current_command=command)
    def save():
        temp = job / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(status_file)
    save()
    try:
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   PYTORCH_CUDA_ALLOC_CONF='', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES=a.gpu)
        with (target / 'train.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        state.update(status='complete', finished=time.time())
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally: save()

if __name__ == '__main__': main()
