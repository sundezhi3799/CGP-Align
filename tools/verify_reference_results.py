"""Compare original run records to the paper's saved source data; no ML dependencies."""
from pathlib import Path
import csv
import json
import math

ROOT = Path(__file__).resolve().parents[1]

def main():
    expected = {(seed, direction) for seed in (31, 37, 41)
                for direction in ('C-P', 'P-C', 'G-P', 'P-G')}
    mapping = {'C-P': 'compound_to_profile', 'P-C': 'profile_to_compound',
               'G-P': 'gene_to_profile', 'P-G': 'profile_to_gene'}
    metrics = {seed: json.loads((ROOT / f'reference_metrics/primary_seed{seed}.json').read_text())
               for seed in (31, 37, 41)}
    path = ROOT / 'paper_snapshot/source_data/figure3c_sampled_retrieval_run_values.csv'
    seen = set()
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['setting'] != 'CGP-Align':
                continue
            key = (int(row['independent_run']), row['short'])
            if key in seen or key not in expected:
                raise ValueError(f'Unexpected or duplicate run/direction: {key}')
            seen.add(key)
            saved = float(metrics[key[0]]['top10_100'][mapping[key[1]]])
            if not math.isclose(saved, float(row['top10']), rel_tol=0, abs_tol=1e-12):
                raise ValueError(f'Metric mismatch for {key}')
    if seen != expected:
        raise ValueError(f'Missing rows: {expected - seen}')
    means = {direction: sum(m['top10_100'][direction] for m in metrics.values()) / 3
             for direction in mapping.values()}
    print(json.dumps({'status': 'PASS', 'matched_run_direction_values': len(seen),
                      'mean_top10': means,
                      'scope': 'saved metrics vs saved source data; not model rerun'}, indent=2))

if __name__ == '__main__':
    main()
