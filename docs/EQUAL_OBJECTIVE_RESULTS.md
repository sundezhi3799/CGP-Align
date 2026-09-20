# Equal-weight three-term training results

Seeds 31/37/41 completed 300 joint epochs each with strict per-seed preprocessing.
The objective is compound + ORF + CRISPR, with unit weights and no auxiliary loss.
Joint initialization reused the completed strict branch pretraining checkpoints;
no preceding joint checkpoint was resumed. Validation selected epochs 140/140/60.

## Matched benchmark

The adopted evaluation individually encodes corrected replicate profiles, averages
by entity and L2-normalizes the mean. Test IDs and candidate banks match those of
the fixed baseline records. Values are three-seed bidirectional means and sample SD.

- Sampled 1:100 Top-10: CGP-Align 68.90% +/- 1.74%; MoLFormer 59.71% +/- 0.49%;
  ChemBERTa 56.78% +/- 0.58%; RDKit-Morgan 56.59% +/- 0.42%.
- CGP-Align sampled 1:1000 Top-10: 28.41% +/- 2.03%.
- CGP-Align full-gallery Recall@10: 8.02% +/- 0.92%.
- Relative to the preceding strict mean-reduction model's 69.48% matched Top-10,
  the new mean differs by -0.58 percentage points. This is descriptive, not a
  statistical significance claim.

Figure 3C-E uses the separate replicate-gallery protocol. Its four-direction
sampled Top-10 HMean is 53.48% +/- 1.05%. Do not compare this directly with 68.90%.
Figure 3H reports a single seed-41 checkpoint: structure baseline 2.5678-fold,
CGP entity 4.7376-fold, entity-to-profile-anchor 9.1972-fold enrichment.

## Reproduction and provenance

- All new numeric records: `revision_candidates/equal_three_term_20260920`.
- Active figures and source CSVs: `figures/figure3` and `figures/supplementary_strict`.
- Training logs and source hashes: `figures/supplementary_strict/source_logs` and
  `source_provenance.json`.
- Checkpoint hashes, selected epochs and exact server paths: new Figure 3
  `strict_sources.json`. These paths document provenance, not public download URLs.
- New checkpoint downloads and prepared input delivery remain outstanding.
- Original historical weights are unchanged. Preceding strict figure sources
  remain in `revision_candidates/figure3_strict_20260920` and prior commit 6fcf936.
- Baselines were not retrained; their identical frozen-model records are retained.

Run `python tools/verify_figure3_sources.py`,
`python tools/build_figure3_matched_sources.py --check`, and
`python tools/build_strict_supplement_sources.py --check` for numeric checks.
The existing R scripts rebuild the four graphics. Current PNG/PDF/SVG exports are
included in each figure directory's `current_exports`.

Four rendered PNGs were visually inspected. Windows R processes produced all
exports and completion markers but returned 3221225477 during shutdown; this is
not recorded as a clean renderer exit. Word exports receive separate layout QA.
This update does not validate Supplementary Figures 2/3 or downstream Figures 4-6.
