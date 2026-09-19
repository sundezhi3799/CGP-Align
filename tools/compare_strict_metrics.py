"""Compare paired historical/strict retrieval metrics after protocol checks."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def compare(old_path, new_path):
    old, new = [json.loads(p.read_text()) for p in (old_path, new_path)]
    if old['split'] != new['split'] or new['split'] != 'test': raise ValueError('Test split mismatch')
    if old['selection_metric'] != new['selection_metric']: raise ValueError('Selection metric mismatch')
    directions = {'compound': ('compound_to_profile', 'profile_to_compound'),
                  'gene': ('gene_to_profile', 'profile_to_gene')}
    for group, keys in directions.items():
        for field in ('num_entities', 'num_profile_replicates', 'modality_counts'):
            if old[group].get(field) != new[group].get(field): raise ValueError('Population mismatch: ' + field)
        for key in keys:
            a, b = [m[group]['mocop_protocol_sampled'][key]['1:100'] for m in (old, new)]
            for field in ('num_queries', 'num_gallery', 'ratio', 'repeats', 'positive_count_mean'):
                if a[field] != b[field]: raise ValueError('Protocol mismatch: ' + field)
    keys = ('compound_to_profile', 'profile_to_compound', 'gene_to_profile', 'profile_to_gene', 'mean', 'hmean')
    return dict(protocol_checks='PASS', historical_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
                strict_sha256=hashlib.sha256(new_path.read_bytes()).hexdigest(),
                historical_checkpoint_epoch=old['checkpoint_epoch'], strict_checkpoint_epoch=new['checkpoint_epoch'],
                selection_metric=new['selection_metric'],
                sampled_top10_1_to_100={k: dict(historical=old['top10_100'][k], strict=new['top10_100'][k],
                    difference_percentage_points=100*(new['top10_100'][k]-old['top10_100'][k])) for k in keys},
                scope='Paired saved-result comparison for one seed; not a three-seed statistical conclusion. Split hashes are verified separately in preprocessing audits.')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, choices=(31,37,41), default=31)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = compare(ROOT / f'reference_metrics/primary_seed{a.seed}.json', ROOT / f'reference_metrics/strict/primary_seed{a.seed}.json')
    content = json.dumps(result, indent=2) + '\n'
    if a.output: a.output.write_text(content)
    print(content)

if __name__ == '__main__': main()
