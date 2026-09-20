# Figure 3: three equal-weight losses

Current panels A-H use the completed equal-objective runs, with strict per-seed
preprocessing. Baseline models and candidate banks are unchanged. Selected epochs
are 140/140/60 for seeds 31/37/41. A/B/H use the seed-41 epoch-60 model.
Source records and hashes are in `../../revision_candidates/equal_three_term_20260920/figure3`.
F/G frozen-model metrics are in its sibling `encoded` directory.

Run from the repository root:

```
python tools/verify_figure3_sources.py
python tools/build_figure3_matched_sources.py --check
Rscript figures/figure3/reproduce.R
```

Current PNG/PDF/SVG files are in `current_exports`; R also exports 600-dpi TIFF.
Panel dimensions and layout follow the preceding version. Old sources remain in
the preceding revision directory and Git commit 6fcf936.
See [results and scope](../../docs/EQUAL_OBJECTIVE_RESULTS.md).
