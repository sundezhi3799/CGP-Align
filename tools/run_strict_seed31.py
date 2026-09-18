"""Foreground controller: strict preprocessing, three fresh branches, then joint training."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


def command_from_config(module_name, config):
    module = importlib.import_module(module_name)
    original = argparse.ArgumentParser.parse_args
    try:
        argparse.ArgumentParser.parse_args = lambda self, *a, **kw: self
        parser = module.parse_args()
    finally:
        argparse.ArgumentParser.parse_args = original
    argv = []
    recognized = {}
    for action in parser._actions:
        key = action.dest
        if key not in config or not action.option_strings or key == 'help': continue
        value = config[key]
        if isinstance(value, dict) and key in ('dynamic_entity_encoding', 'entity_features'):
            value = value['mode']
        if isinstance(value, (dict, list)):
            raise ValueError('Unexpected structured CLI value: ' + key)
        recognized[key] = value
        if isinstance(action, argparse._StoreTrueAction):
            if value: argv.append(action.option_strings[0])
        elif isinstance(action, argparse._StoreFalseAction):
            if not value: argv.append(action.option_strings[0])
        elif value is not None:
            argv.extend([action.option_strings[0], str(value)])
    parsed = vars(parser.parse_args(argv))
    for key, value in recognized.items():
        if isinstance(parsed[key], Path) and parsed[key] == Path(value): continue
        if str(parsed[key]) != str(value):
            raise ValueError('Config round-trip mismatch: ' + key)
    return [sys.executable, '-u', str(ROOT / 'scripts' / (module_name + '.py'))] + argv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--research-root', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--gpus', default='0,1,2', help='Three idle physical GPUs for compound, ORF, CRISPR.')
    a = p.parse_args()
    job = a.run_root.resolve()
    job.mkdir(parents=True, exist_ok=True)
    lock = job / 'controller.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    state = dict(status='preprocessing', controller_pid=os.getpid(), seed=31, started=time.time(), jobs=[])
    children = []
    def save():
        tmp = job / 'status.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(job / 'status.json')
    def run(command, log):
        state['current_command'] = command; save()
        with log.open('w') as f: subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTORCH_CUDA_ALLOC_CONF='', PYTHONUNBUFFERED='1')
    gpus = a.gpus.split(',')
    if len(gpus) != 3 or len(set(gpus)) != 3: raise ValueError('Expected three different GPUs')
    try:
        save()
        base = a.research_root.resolve() / 'output/cgp_align/profile_preprocess_version_matrix'
        dirs = {}
        for kind, family, raw in [('compound', 'compound_raw_common3180_v1_20260604_0155', 'raw_common3180_compound'),
                                  ('gene', 'gene_raw_common3180_v1_20260604_0155', 'raw_common3180_reagent')]:
            target = job / 'data' / kind
            run([sys.executable, '-u', str(ROOT / 'tools/prepare_strict_profiles.py'),
                 '--raw-dir', str(base / family / raw), '--split-dir', str(base / 'raw3180_seeded_split_3seed_20260605/seed31' / (kind + '_source_zscore_plate_well_center_3180')),
                 '--kind', kind, '--seed', '31', '--output-dir', str(target)], job / (kind + '_preprocessing.log'))
            audit = json.loads((target / 'profile_correction_audit.json').read_text())
            if audit['seed'] != 31 or audit['fit_val_overlap'] or audit['fit_test_overlap']:
                raise ValueError('Strict preprocessing audit failed')
            dirs[kind] = target
        protein = a.research_root.resolve() / 'protein_embeddings/protein_encoder_ablation/esm650_l33_mean_cls'
        checkpoints = {}
        for branch, gpu in zip(('compound', 'orf', 'crispr'), gpus):
            cfg = json.loads((ROOT / 'manifests/checkpoint_configs' / ('initializer_seed31_' + branch + '.json')).read_text())
            branch_root = job / 'branches' / branch
            branch_root.mkdir(parents=True)
            cfg.update(data_dir=str(dirs['compound' if branch == 'compound' else 'gene']),
                       run_name='strict_seed31_' + branch, checkpoint_dir=str(branch_root / 'checkpoints'),
                       log_dir=str(branch_root / 'logs'), output_dir=str(branch_root / 'eval'),
                       device='cuda:0', train_only=True, eval_only=False)
            if branch != 'compound': cfg['protein_embedding_dir'] = str(protein)
            # No old trained initializer is loaded; branch config is used only as a recipe.
            command = command_from_config('train_compound_mocop_replicate' if branch == 'compound' else 'train_gene_mocop', cfg)
            (branch_root / 'configuration.json').write_text(json.dumps(cfg, indent=2) + '\n')
            handle = (branch_root / 'train.log').open('w')
            child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                     cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=gpu))
            handle.close(); children.append(child)
            checkpoints[branch] = branch_root / 'checkpoints' / cfg['run_name'] / 'best_model.pt'
            state['jobs'].append(dict(branch=branch, gpu=gpu, pid=child.pid, command=command))
        state['status'] = 'branch_pretraining'; save()
        while True:
            codes = [child.poll() for child in children]
            if any(code not in (None, 0) for code in codes): raise RuntimeError('A branch failed: ' + str(codes))
            if all(code == 0 for code in codes): break
            state['branch_exit_codes'] = codes; save(); time.sleep(30)
        for path in checkpoints.values():
            if not path.is_file(): raise FileNotFoundError(path)
        cfg = json.loads((ROOT / 'manifests/checkpoint_configs/primary_seed31.json').read_text())
        joint = job / 'joint'; joint.mkdir()
        cfg.update(compound_data_dir=str(dirs['compound']), gene_data_dir=str(dirs['gene']),
                   gene_protein_embedding_dir=str(protein), run_name='strict_seed31_joint',
                   checkpoint_dir=str(joint / 'checkpoints'), log_dir=str(joint / 'logs'), output_dir=str(joint / 'eval'),
                   init_compound_checkpoint=str(checkpoints['compound']), init_orf_gene_checkpoint=str(checkpoints['orf']),
                   init_crispr_gene_checkpoint=str(checkpoints['crispr']), init_gene_checkpoint=None, resume_checkpoint=None,
                   device='cuda:0', train_only=False, eval_only=False)
        command = command_from_config('train_cgp_align_replicate', cfg)
        (joint / 'configuration.json').write_text(json.dumps(cfg, indent=2) + '\n')
        state['status'] = 'joint_training'; env['CUDA_VISIBLE_DEVICES'] = gpus[0]; save()
        run(command, joint / 'train.log')
        state['status'] = 'complete'; state['finished'] = time.time(); save()
    except BaseException as error:
        for child in children:
            if child.poll() is None: child.terminate()
        state['status'] = 'failed'; state['error'] = repr(error); save()
        raise


if __name__ == '__main__': main()
