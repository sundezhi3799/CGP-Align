"""Evaluate released checkpoints on their matching prepared test data in a staged runtime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'scripts'), str(ROOT)]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=[31, 37, 41], default=41)
    p.add_argument('--variant', choices=['full','no_source_indicator','no_pretraining'], default='full')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    marker = ROOT / 'RUNTIME_VERSION.json'
    if not marker.exists() or json.loads(marker.read_text())['version'] != 'final_architecture_20260920':
        p.error('First stage tools/prepare_final_architecture_runtime.py, then run this tool inside that runtime.')
    from run_strict_seed import command_from_config
    base = a.artifacts.resolve()
    model = base / f'models/seed{a.seed}/joint' if a.variant == 'full' else base / 'ablations' / a.variant / f'seed{a.seed}'
    config = json.loads((model / 'configuration.json').read_text())
    config.update(compound_data_dir=str(base / f'data/seed{a.seed}/compound'),
                  gene_data_dir=str(base / f'data/seed{a.seed}/gene'),
                  gene_protein_embedding_dir=str(base / 'protein'),
                  checkpoint_dir=str(model.parent), run_name=model.name,
                  output_dir=str(a.output.resolve()), log_dir=str(a.output.resolve() / 'logs'), device=a.device,
                  eval_only=True, train_only=False, smoke_test=False)
    for key in ['init_compound_checkpoint','init_gene_checkpoint','init_orf_gene_checkpoint','init_crispr_gene_checkpoint']:
        config[key] = None
    command = command_from_config('train_cgp_align_replicate', config)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'portable_evaluation_config.json').write_text(json.dumps(config, indent=2))
    if a.dry_run: print(json.dumps(command, indent=2))
    else:
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        subprocess.run(command, check=True, env=env)

if __name__ == '__main__': main()
