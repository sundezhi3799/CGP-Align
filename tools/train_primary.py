"""Portable, opt-in invocation of the original joint-training candidate recipe."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, choices=[31, 37, 41], required=True)
    for name in ['compound-data', 'gene-data', 'protein-data', 'compound-init', 'orf-init', 'crispr-init', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    # Paths are resolved against caller cwd; subprocess cwd is the repository root.
    paths = {k: v.resolve() for k, v in vars(a).items() if isinstance(v, Path)}
    required = [paths[k] for k in ['compound_init', 'orf_init', 'crispr_init']]
    required += [paths['compound_data'] / name for name in ['compound_mocop_entities.parquet', 'compound_mocop_replicates.parquet', 'compound_mocop_replicate_features.npy', 'splits_compound_mocop.json']]
    required += [paths['gene_data'] / name for name in ['gene_mocop_entities.parquet', 'gene_mocop_replicates.parquet', 'gene_mocop_replicate_features.npy', 'splits_gene_mocop.json']]
    required += [paths['protein_data'] / 'protein_sequence_embeddings.npy']
    out = paths['output']
    command = [sys.executable, str(ROOT / 'scripts/train_cgp_align_replicate.py')]
    options = {'compound_data_dir': paths['compound_data'], 'gene_data_dir': paths['gene_data'],
               'gene_protein_embedding_dir': paths['protein_data'],
               'init_compound_checkpoint': paths['compound_init'],
               'init_orf_gene_checkpoint': paths['orf_init'], 'init_crispr_gene_checkpoint': paths['crispr_init'],
               'run_name': f'raw3180_staged4_branchinit_joint_no_anchor_300_sparseckpt_splitseed{a.seed}',
               'checkpoint_dir': out / 'checkpoints', 'log_dir': out / 'logs', 'output_dir': out / 'eval',
               'compound_split_name': 'cold_compound', 'gene_split_name': 'cold_gene',
               'profile_source_mode': 'compound_gene_modality', 'epochs': 300,
               'stage1_epochs': 0, 'stage2_epochs': 0, 'min_selection_epoch': 40,
               'save_candidate_epochs': '40,120,240,300', 'anchor_weight_stage2': 0,
               'anchor_weight_stage3': 0, 'cg_teacher_weight': 0, 'profile_structure_weight': 0,
               'gene_batch_size': 256, 'val_max_gene_entities': 2048, 'weight_decay': 0.01,
               'gene_temperature': 0.07, 'seed': a.seed, 'device': a.device}
    for key, value in options.items():
        command.extend(['--' + key, str(value)])
    command.append('--split_gene_modality_branches')
    if a.device.startswith('cuda'):
        command.append('--amp')
    missing = [str(x) for x in required if not x.is_file()]
    print(json.dumps({'command': command, 'missing_inputs': missing,
                      'recipe_status': 'parameters match recovered historical checkpoint configs; preprocessing audit requires resolution'}, indent=2))
    if missing:
        raise SystemExit(2)
    if a.execute:
        subprocess.run(command, cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
