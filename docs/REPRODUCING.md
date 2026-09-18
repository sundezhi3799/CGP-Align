# Reproduction workflows

## 1. Check archived numbers

```bash
python tools/verify_reference_results.py
Rscript paper_snapshot/scripts/reproduce_article_results.R
```

The first command compares 12 direction/run values with raw saved run metrics. The second recomputes numerical summaries/statistics from processed source tables. Neither command trains a model.

## 2. Check the original model implementation

```bash
python tools/smoke_test_model.py
python scripts/train_cgp_align_replicate.py --help
```

Synthetic inputs are used only for a CPU implementation check. No synthetic score is a paper result.

## 3. Prepare full training inputs

Recover the exact upstream preparation workflow and artifacts listed in `DATA.md`. Included scripts cover shared-feature inspection, gene replicate construction/correction, split generation and protein feature computation. The compound preparation gaps remain documented in `RELEASE_AUDIT.md`.

## 4. Branch initialization and joint training

Original entry points are `train_compound_mocop_replicate.py`, `train_gene_mocop.py` and `train_cgp_align_replicate.py`. The joint recipe below preserves the selected historical launcher arguments and requires three real initializer checkpoints:

```bash
python tools/train_primary.py --seed 31 --compound-data data/seed31/compound --gene-data data/seed31/gene --protein-data data/protein --compound-init checkpoints/seed31/compound.pt --orf-init checkpoints/seed31/orf.pt --crispr-init checkpoints/seed31/crispr.pt --output output/seed31
```

This prints a portable command and checks required files. Add `--execute` to train after inspecting it. Repeat with seeds 37 and 41 and their corresponding artifacts. Use `--device cpu` for debugging; full CPU runtime is not benchmarked. CUDA is the historical training device. This wrapper is a candidate recipe pending checkpoint-config comparison, not a guarantee of exact retraining results.

## 5. Evaluation and downstream analyses

Entry points and current coverage are in `manifests/reproduction_coverage.csv`. Start with `--help`. Intrinsic evaluation and hidden-neighbour export now accept explicit data path overrides; see [PORTABILITY_UPDATE.md](PORTABILITY_UPDATE.md). Other historical entry points may still require path adaptation. Do not run upstream analysis using a different model family merely because its file format is compatible.

## 6. Revised Figure 3

The September Word revision adopts hidden phenotype neighbours in H. See
[the complete redraw workflow](../figures/figure3/README.md) and
`python tools/verify_figure3_sources.py`. The August snapshot and old paired-gain
source tables are retained without alteration. The H checkpoint differs from the
primary seed-41 checkpoint; this difference is documented, not resolved by renaming.
