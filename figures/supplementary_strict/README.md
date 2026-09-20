# Supplementary Figures 1, 4 and 5

These figures use the same equal-weight three-term models as current Figure 3.
S1 contains all 900 training epochs, with selected epochs 140/140/60. Total loss
is compound + ORF + CRISPR. S4 reports sampled/full-gallery retrieval; S5 reports
query-specific random controls and three-seed variability.

```
python tools/build_strict_supplement_sources.py --check
python tools/verify_figure3_sources.py
Rscript figures/supplementary_strict/figureS1.R
Rscript figures/supplementary_strict/figureS4.R
Rscript figures/supplementary_strict/figureS5.R
```

Each R script accepts an output directory. Current PNG/PDF/SVG exports are in
`current_exports`; R also produces TIFF at 600 dpi. Panels retain their existing
183 x 132 mm layout. Raw numeric logs and source hashes are included.
Prior versions are preserved in Git commit 6fcf936 and the historical snapshot.
See [results and remaining scope](../../docs/EQUAL_OBJECTIVE_RESULTS.md).
