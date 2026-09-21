"""Compare U2OS response distances for retrieved neighbours (Figure 6f-h)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prism_nyan_utils import l2_normalize, load_nyan_similarity


def evaluate(input_dir, features_path, output):
    input_dir, features_path, output = map(Path, (input_dir, features_path, output))
    output.mkdir(parents=True, exist_ok=True)
    matrix_path = input_dir / 'prism_functional_retrieval_matrices.npz'
    z = np.load(matrix_path, allow_pickle=True)
    meta = pd.read_csv(input_dir / 'prism_cgp_overlap_compounds.csv')
    if not np.array_equal(meta.inchikey.to_numpy(), z['inchikey']):
        raise ValueError('Compound metadata and response matrix row order differ.')
    cell = np.flatnonzero(z['cell_lines'].astype(str) == 'ACH-000364')
    if len(cell) != 1:
        raise ValueError('Expected exactly one U2OS (ACH-000364) response column.')
    y = z['prism_response'][:, cell[0]]
    t = z['morgan_tanimoto']
    features = np.load(features_path)
    if features.shape != (len(y), 2059):
        raise ValueError('Expected row-aligned Morgan-2048 + 11 RDKit descriptors.')
    v = l2_normalize(features)
    rdkit = v @ v.T
    np.fill_diagonal(rdkit, -np.inf)
    scores = {'CGP-Align': z['cgp_similarity'],
              'NYAN': load_nyan_similarity(input_dir, meta.canonical_smiles.tolist()),
              'RDKit-Morgan': rdkit}
    if any(s.shape != (len(y), len(y)) for s in scores.values()) or t.shape != rdkit.shape:
        raise ValueError('Pairwise matrices have inconsistent dimensions.')
    rows, controls, pairs = [], [], []
    for i in np.flatnonzero(np.isfinite(y)):
        eligible = ((np.arange(len(y)) != i) & np.isfinite(y) & np.isfinite(t[i])
                    & (t[i] >= 0) & (t[i] < .20))
        common = eligible.copy()
        for score in scores.values():
            common &= np.isfinite(score[i])
        ids = np.flatnonzero(common)
        if len(ids) < 50:
            continue
        for method, score in scores.items():
            selected = ids[np.argsort(-score[i, ids], kind='stable')][:50]
            rows.append(dict(query_idx=int(i), active_query=bool(y[i] <= -1),
                             method=method, mean_absdiff=float(np.abs(y[selected] - y[i]).mean()),
                             mean_tanimoto=float(t[i, selected].mean()), n_candidates=len(ids)))
        # The control is a random expectation conditional on query and Tanimoto bin.
        # Eligible retrieved compounds remain in the control pool.
        cand = np.flatnonzero(eligible & np.isfinite(scores['CGP-Align'][i]))
        selected = cand[np.argsort(-scores['CGP-Align'][i, cand], kind='stable')][:50]
        for width in (.01, .005, .02):
            distances, expected = [], []
            for rank, k in enumerate(selected, 1):
                pool = cand[np.floor(t[i, cand] / width).astype(int) == int(np.floor(t[i, k] / width))]
                dist = float(abs(y[i] - y[k]))
                control = float(np.abs(y[i] - y[pool]).mean())
                distances.append(dist)
                expected.append(control)
                if width == .01:
                    pairs.append(dict(query_idx=int(i), retrieved_idx=int(k), rank=rank,
                                      query_active=bool(y[i] <= -1), morgan_tanimoto=float(t[i, k]),
                                      response_absdiff=dist, control_absdiff=control,
                                      control_pool_size=len(pool)))
            controls.append(dict(query_idx=int(i), query_active=bool(y[i] <= -1),
                                 bin_width=width, n=len(selected), cgp_absdiff=float(np.mean(distances)),
                                 control_absdiff=float(np.mean(expected)),
                                 gain=float(np.mean(expected) - np.mean(distances))))
    if not rows:
        raise ValueError('No queries have 50 eligible neighbours.')
    q = pd.DataFrame(rows)
    q.to_csv(output / 'query_metrics.csv', index=False)
    wide = q.pivot(index=['query_idx', 'active_query'], columns='method', values='mean_absdiff').reset_index()
    wide.to_csv(output / 'paired_queries.csv', index=False)
    summary = []
    for scope, sub in [('active_queries', wide[wide.active_query]), ('all_queries', wide)]:
        for baseline in ['NYAN', 'RDKit-Morgan']:
            delta = sub[baseline] - sub['CGP-Align']
            summary.append(dict(scope=scope, baseline=baseline, n_queries=len(sub),
                                cgp_mean=float(sub['CGP-Align'].mean()), baseline_mean=float(sub[baseline].mean()),
                                mean_gain=float(delta.mean()), median_gain=float(delta.median()),
                                fraction_cgp_better=float((delta > 0).mean())))
    pd.DataFrame(summary).to_csv(output / 'summary.csv', index=False)
    control = pd.DataFrame(controls)
    control[control.bin_width == .01].drop(columns='bin_width').to_csv(output / 'query_comparison.csv', index=False)
    pd.DataFrame(pairs).to_csv(output / 'retrieved_pairs.csv', index=False)
    sensitivity = []
    for width, group in control.groupby('bin_width', sort=False):
        for scope, sub in [('active_queries', group[group.query_active]), ('all_queries', group)]:
            sensitivity.append(dict(scope=scope, bin_width=width, n_queries=len(sub), n_retrievals=int(sub.n.sum()),
                                    cgp_absdiff=float(sub.cgp_absdiff.mean()), control_absdiff=float(sub.control_absdiff.mean()),
                                    mean_gain=float(sub.gain.mean()), median_gain=float(sub.gain.median()),
                                    positive_query_fraction=float((sub.gain > 0).mean())))
    pd.DataFrame(sensitivity).to_csv(output / 'sensitivity.csv', index=False)
    report = dict(cell_line='ACH-000364', active_threshold=-1, topk=50, max_tanimoto=.20,
                  known_u2os_compounds=int(np.isfinite(y).sum()), missing_u2os_compounds=int((~np.isfinite(y)).sum()),
                  matrix_sha256=hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
                  feature_sha256=hashlib.sha256(features_path.read_bytes()).hexdigest(),
                  baseline_comparisons=summary, matched_control=sensitivity)
    (output / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.input_dir, args.features, args.output)
