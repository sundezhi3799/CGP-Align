# Revised Figure 3

All panels now use the strict, train-fitted preprocessing experiment family.
A-E/H were regenerated on September 20; F/G retain the adopted encode-then-average
results from the same strict checkpoints and matched comparators.

```bash
python tools/verify_figure3_sources.py
python tools/build_figure3_matched_sources.py --check
Rscript environment/install-figure3.R
Rscript figures/figure3/reproduce.R
```

Outputs are `outputs/figure3/figure3.{png,pdf,svg,tiff}` and session information.
Optional arguments specify an output directory and font (default Arial).
SVG retains editable text; PNG/TIFF use 600 dpi. The panel layout is preserved,
with horizontal A/B legends placed above the point clouds.

## Sources and interpretation

- A/B: same 1,200 test compounds and 1,200 test genes from seed 41. Before joint
  training, pretrained entity encoders are loaded and the shared profile encoder
  retains its seed-41 initialization. After training, the validation-selected
  epoch-60 joint checkpoint is used. Replicate profiles are encoded individually,
  averaged by entity and L2-normalized. A separate full-SVD PCA is fitted to the
  combined entity/profile vectors within each stage. Axis variances are recorded.
- C-E: seeds 31/37/41, raw strict retrieval metrics. Random hit probabilities use
  each query's integer positive count and hypergeometric sampling, then average
  over queries. E averages directions within seed before reporting mean/sample SD.
- F/G: twelve frozen-model evaluations in `revision_candidates/encoded_mean_20260920`.
  All methods encode individual corrected replicates, average and L2-normalize.
  Both directions and ten sampled candidate repeats are averaged within seed;
  error bars are sample SD across three seeds. See
  [protocol and sensitivity record](../../docs/STRICT_MATCHED_BENCHMARK.md).
- H: seed-41 epoch-60 model, 11,578 test compounds and 11,577 candidates per query.
  Ten reference neighbours are defined by cosine similarity of L2-normalized means
  of corrected CellProfiler profiles. Enrichment is mean precision@50 divided by
  10/11,577: structure 2.5677782, CGP entity 4.6635972, profile-anchor 9.2771987.
  The profile-anchor readout uses gallery profile embeddings. This measures
  phenotype-defined neighbourhood recovery from one checkpoint.

A/B/H share checkpoint SHA256
`44bbee704a4bbb1b0b8eb9df5a514f201504d74cab6dea5f5b8bb5dfbee19cea`.
Detailed source records, query-level H results and upstream instructions are in
[the strict export package](../../revision_candidates/figure3_strict_20260920/README.md).
Historical filenames retain their original figure numbers. The prior composite
is preserved in `historical_before_strict_20260920`; the earlier F/G archive is
`historical_20260918`. Historical source hashes remain in the provenance manifest;
`copied_sha256` tracks the current deliberately revised files.

## Execution scope

The included tables reproduce the graphic and allow independent arithmetic checks.
Upstream inference additionally requires the exact prepared data and checkpoints;
server paths in provenance are records, not public download URLs.
This update does not regenerate other manuscript figures or supplementary figures.

Known local runtime issue: Windows R 4.5.3 exits with native status `0xC0000005`
after writing all images and session information. The exports were visually
verified, but a clean exit on another R installation remains unverified. Do not
suppress process failure in automated pipelines.
