"""Build or verify Figure 3F/G tables from adopted frozen-model metric records."""
import argparse
import csv
import io
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'revision_candidates/encoded_mean_20260920'
METHODS = [('cgp', 'cgp_align_main', 'CGP-Align'),
           ('molformer_xl', 'matched_molformer_smiles', 'SMILES-LM profile (MoLFormer)'),
           ('chemberta', 'matched_chemberta_smiles', 'SMILES-LM profile (ChemBERTa)'),
           ('morgan', 'matched_rdkit_morgan', 'Fingerprint-profile baseline')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    f, g = [], []
    for method, key, label in METHODS:
        values = {setting: [] for setting in ('100', '1000', 'full')}
        for seed in (31, 37, 41):
            record = json.loads((RESULTS / f'seed{seed}' / method / 'metrics.json').read_text())
            assert record['protocol'] == 'strict_entity_latent_mean_v1'
            assert record['test_count'] == 11578
            for setting in values:
                scores = [(d['full_gallery']['Recall@10'] if setting == 'full' else
                           d['sampled'][setting]['Recall@10']['mean']) for d in record['directions'].values()]
                values[setting].append(statistics.mean(scores))
        stats = {k: (statistics.mean(v), statistics.stdev(v)) for k, v in values.items()}
        mean, sd = stats['100']
        f.append(dict(method_key=key, method_label=label, n_independent_runs=3,
                      mean_top10=mean, sd_top10=sd, label=f'{mean*100:.1f}%', label_x=mean+sd+.01))
        for setting, metric, title, offset in [('1000', 'top10_1000', 'Sampled 1:1000\nTop-10', .008),
                                              ('full', 'full_r10', 'Full-gallery\nRecall@10', .003)]:
            mean, sd = stats[setting]
            g.append(dict(method_key=key, method_label=label, top10_1000_sd=stats['1000'][1],
                          full_r10_sd=stats['full'][1], metric=metric, value=mean, sd=sd,
                          metric_label=title, label=f'{mean*100:.1f}%', label_x=mean+sd+offset))
    for name, rows in [('figure3_panelB_retrieval_performance.csv', f),
                       ('figure3_panelC_difficult_retrieval.csv', g)]:
        target = ROOT / 'figures/figure3/source_data' / name
        if args.check:
            with target.open(newline='', encoding='utf-8') as handle:
                actual = list(csv.DictReader(handle))
            assert len(actual) == len(rows)
            for expected, saved in zip(rows, actual):
                for key, value in expected.items():
                    if isinstance(value, (float, int)):
                        assert abs(float(saved[key])-value) < 1e-12, (name, key)
                    else:
                        assert saved[key] == value, (name, key)
        else:
            with target.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    print('PASS: Figure 3F/G agree with 12 adopted frozen-model evaluations')


if __name__ == '__main__':
    main()
