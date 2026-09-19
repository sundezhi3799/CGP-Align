"""Summarize frozen-model aggregation sensitivity with seed-level uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--previous-run', type=Path, required=True)
    a = p.parse_args()
    records = []
    for seed in (31, 37, 41):
        for method in ('cgp', 'morgan', 'molformer_xl', 'chemberta'):
            path = a.run_root / f'seed{seed}' / method
            new = json.loads((path / 'metrics.json').read_text())
            old = json.loads((a.previous_run / f'seed{seed}/eval' / method / 'metrics.json').read_text())
            for key in ('checkpoint_sha256', 'negative_candidates_sha256', 'prepared_audit_sha256', 'test_count'):
                assert new[key] == old[key], key
            assert new['aggregation'] == 'mean_encoded'
            ranks = np.load(path / 'query_ranks.npz')
            for direction, metrics in new['directions'].items():
                for setting in ('100', '1000', 'full'):
                    rank = ranks[direction + '_' + setting]
                    expected = (metrics['full_gallery']['Recall@10'] if setting == 'full'
                                else metrics['sampled'][setting]['Recall@10']['mean'])
                    assert abs(float((rank <= 10).mean()) - expected) < 1e-12
            for aggregation, result in [('encode_mean', old), ('mean_encoded', new)]:
                for setting in ('100', '1000', 'full'):
                    values = [(m['full_gallery']['Recall@10'] if setting == 'full'
                               else m['sampled'][setting]['Recall@10']['mean'])
                              for m in result['directions'].values()]
                    records.append(dict(seed=seed, method=method, aggregation=aggregation,
                                        setting=setting, bidirectional_top10=float(np.mean(values))))
    summary = []
    for method in ('cgp', 'molformer_xl', 'chemberta', 'morgan'):
        for aggregation in ('encode_mean', 'mean_encoded'):
            for setting in ('100', '1000', 'full'):
                values = [r['bidirectional_top10'] for r in records if
                          (r['method'], r['aggregation'], r['setting']) == (method, aggregation, setting)]
                summary.append(dict(method=method, aggregation=aggregation, setting=setting,
                                    mean=float(np.mean(values)), sd_ddof1=float(np.std(values, ddof=1)), n=3))
    output = dict(protocol_selected_after_test_sensitivity_analysis=True,
                  weights_and_candidates_unchanged=True, rank_metric_checks='passed',
                  seed_records=records, summary=summary)
    (a.run_root / 'summary.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
