"""Build S1/S4/S5 sources from saved strict training logs and Figure 3 metrics."""
import csv
import argparse
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures/supplementary_strict'
SOURCE = ROOT / 'revision_candidates/figure3_strict_20260920'
OUT = FIG / 'source_data'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--check', action='store_true', help='Compare regenerated CSV content without writing')
CHECK = parser.parse_args().check


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write(name, rows):
    f = io.StringIO(newline='')
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    data = f.getvalue().encode('utf-8')
    if CHECK:
        assert (OUT / name).read_bytes() == data, name
    else:
        (OUT / name).write_bytes(data)


OUT.mkdir(exist_ok=True)
sources = json.loads((SOURCE / 'strict_sources.json').read_text())
losses, validation, selected, sampled, full = [], [], [], [], []
directions = [('compound', 'compound_to_profile', 'C-P', 'c2p'),
              ('compound', 'profile_to_compound', 'P-C', 'p2c'),
              ('gene', 'gene_to_profile', 'G-P', 'g2p'),
              ('gene', 'profile_to_gene', 'P-G', 'p2g')]
for source in sources:
    seed = source['seed']
    log = [json.loads(x) for x in (FIG / f'source_logs/seed{seed}.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in log] == list(range(1, 301))
    eligible = [r for r in log if r.get('selection_active') and r.get('val_hmean_Top10') is not None]
    best = max(eligible, key=lambda r: r['val_hmean_Top10'])
    assert best['epoch'] == source['epoch']
    for r in log:
        # The implementation averages ORF/CRISPR losses within the gene term.
        assert math.isclose(r['train_gene_loss'], (r['train_gene_orf_loss'] + r['train_gene_crispr_loss'])/2, abs_tol=2e-6)
        assert math.isclose(r['train_total_loss'], r['train_compound_loss']+r['train_gene_loss'], abs_tol=2e-6)
        for component, key in [('Total objective', 'train_total_loss'), ('Compound', 'train_compound_loss'),
                               ('ORF', 'train_gene_orf_loss'), ('CRISPR', 'train_gene_crispr_loss')]:
            losses.append(dict(seed=seed, epoch=r['epoch'], component=component, loss=r[key]))
        if r.get('val_hmean_Top10') is not None:
            validation.append(dict(seed=seed, epoch=r['epoch'], value=r['val_hmean_Top10'],
                                   selection_active=int(bool(r.get('selection_active'))),
                                   selected=int(r['epoch']==best['epoch'])))
    for kind, direction, short, key in directions:
        selected.append(dict(seed=seed, epoch=best['epoch'], direction=short,
                             value=best[f'val_{key}_Top10'], hmean=best['val_hmean_Top10'],
                             checkpoint_sha256=source['checkpoint_sha256']))
    metrics = json.loads((SOURCE / f'strict_seed{seed}_test_metrics.json').read_text())
    for kind, direction, short, key in directions:
        s = metrics[kind]['mocop_protocol_sampled'][direction]['1:100']
        f = metrics[kind]['full_gallery'][direction]
        common = dict(independent_run=seed, direction=direction.replace('_', ' '), short=short,
                      branch='compound-profile' if kind=='compound' else 'gene-profile')
        for k in (1, 10):
            sampled.append(dict(**common, candidate_setting='1:100', metric=f'Top{k}',
                                value=s[f'Top{k}_accuracy_mean'], sd_within_repeats=s[f'Top{k}_accuracy_std'],
                                **{x:s[x] for x in ['num_queries','num_gallery','ratio','repeats','positive_count_mean']}))
        for metric in ('Recall@1', 'Recall@5', 'Recall@10', 'MRR'):
            full.append(dict(**common, candidate_setting='full-gallery', metric=metric,value=f[metric],
                             **{x:f[x] for x in ['num_queries','num_gallery','num_evaluated']}))
write('figureS1_losses.csv', losses)
write('figureS1_validation.csv', validation)
write('figureS1_selected.csv', selected)
write('figureS4_sampled.csv', sampled)
write('figureS4_full.csv', full)
summaries = read_csv(SOURCE / 'branch_summary_run_values.csv')
write('branch_summary.csv', summaries)
controls = read_csv(SOURCE / 'figureS5_random_ranking_negative_control.csv')
write('figureS5_controls.csv', controls)
print('PASS: 900 training epochs; selected epochs 70/150/60; S4/S5 sources from strict Figure 3 records')
