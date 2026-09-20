# Strict Supplementary Figures 1, 4 and 5

These September 20 figures use the same strict primary models as revised Figure 3.
Historical Supplementary Figure source data remain unchanged in `paper_snapshot`.

From the repository root:

```bash
python tools/build_strict_supplement_sources.py --check
python tools/verify_figure3_sources.py
Rscript environment/install-figure3.R
Rscript figures/supplementary_strict/figureS1.R
Rscript figures/supplementary_strict/figureS4.R
Rscript figures/supplementary_strict/figureS5.R
```

Each R script accepts an optional output directory; default is
`outputs/supplementary_strict`. Exports are 183 x 132 mm PDF/SVG and 600 dpi
PNG/TIFF with Arial. SVG retains editable text. Run without `--check` to rebuild
CSV sources from the saved raw logs and strict Figure 3 records.

S1 contains 900 joint-training epochs from seeds 31/37/41, with validation-selected
epochs 70/150/60. Training loss is compound loss plus the mean of the ORF/CRISPR
losses, verified against all 900 log entries. The figure presents training,
validation, selection and selected-checkpoint directional validation scores.
The former training-scale ablation was not rerun under strict preprocessing;
it remains in the historical snapshot and is excluded from this revised figure.

S4 uses strict raw sampled and full-gallery metrics. Recall@k means the fraction
of queries with any matching candidate in the top k. The heatmaps express all
metrics, including MRR, as percentages. S5 uses exact query-specific random hit
probabilities already verified for Figure 3. Directional summaries are computed
within each seed; error bars show sample SD across the three seeds. No hypothesis
tests are reported. The S5 enrichment panel uses points on logarithmic axes.

Plots S4/S5 adapt the original R supplementary figure workflow, correcting the
random control, using deterministic jitter and preserving the original panel
arrangement. Compact CSVs and full numeric training logs are included. Source log
hashes are in `source_provenance.json`; checkpoint hashes are in the Figure 3
strict source record. This workflow redraws saved evaluations, not model training.

The local Windows R installation writes all outputs but returns a native failure
on shutdown; exports were inspected and no dropped-data plotting warnings occurred.
Only locale startup warnings were reported. A clean exit on another installation
remains unverified. Automated pipelines should retain process failure reporting.

The scope excludes Supplementary Figures 2/3 and downstream main Figures 4-6.
See [the manuscript consistency audit](../../docs/SUPPLEMENT_STRICT_AUDIT.md) for
remaining manuscript work, including a newly identified objective-weight mismatch.
