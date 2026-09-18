"""Check revised H arithmetic and provenance; does not rerun model inference."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'figures/figure3'
rows = list(csv.DictReader((ROOT / 'source_data/figure3h_hidden_phenotype_neighbour_recovery.csv').open()))
expected = {'Structure feature baseline': 2.5677782000345486,
            'CGP-Align entity': 4.8695793746761105,
            'CGP entity -> profile anchor': 9.5171779236483}
assert len(rows) == 3 and {r['method'] for r in rows} == set(expected)
for r in rows:
    assert (int(r['n_query']), int(r['n_gallery']), int(r['k'])) == (11578, 11577, 50)
    # Fixed 10 positives: recall * 10 / 50 is precision; random precision = 10/11577.
    enrichment = float(r['recall_at_k']) * 11577 / 50
    assert math.isclose(enrichment, float(r['precision_enrichment']), rel_tol=1e-10)
    assert math.isclose(enrichment, expected[r['method']], rel_tol=1e-10)
meta = json.loads((ROOT / 'provenance/hidden_analysis_metadata.json').read_text())
assert meta['positive_topn'] == 10
inputs = json.loads((ROOT / 'provenance/hidden_input_metadata.json').read_text())
assert inputs['split'] == 'test' and inputs['checkpoint_epoch'] == 120
assert inputs['compound_entities'] == 11578
assert math.isclose(inputs['checkpoint_best_score'], 0.5431782488809668)
print('PASS: revised Figure 3H saved-source arithmetic and single-checkpoint provenance')
