# Benchmark and figure reproduction

This directory supplies the current Figure 3–6 source tables, plotting code,
matched compound–gene splits, and extended toxicity/PRISM comparisons.
The frozen input arrays are distributed in the
[benchmark-inputs release](https://github.com/sundezhi3799/CGP-Align/releases/tag/benchmark-inputs-20261112).
Third-party inputs retain the terms described in `INPUT_ATTRIBUTION.md`.

## Prepared inputs

From the repository root:

```bash
python scripts/download_data.py --manifest configs/downloads/benchmark_inputs_20261112.json --output ../cgp-benchmarks --groups relation-features extended-benchmark-inputs
```

The manifest records SHA-256 checksums for both archives and every extracted file.
Use the environment in `requirements-downstream.txt`.

The optional `nyan-matched` group contains three trained NYAN alignment
checkpoints, frozen NYAN features, evaluation embeddings, ranks and provenance:

```bash
python scripts/download_data.py --manifest configs/downloads/benchmark_inputs_20261112.json --output ../cgp-benchmarks --groups nyan-matched
```

## Compound–gene relation prediction

```bash
python reproduction/relation_benchmark/run_interaction.py --features ../cgp-benchmarks/features --output ../cgp-results/relations32 --dimension 32 --workers 1
```

Use a separate output folder and `--dimension 64` for the sensitivity analysis.
`--methods RDKit` and `--split broad_31_dgidb` select a small reproduction run.
The supplied 30 splits cover five held-out annotation sources, three seeds,
and two relation graphs. Each partition has equal positive and unlabelled-control
counts with exactly matched compound and gene marginals. Pairs are disjoint;
individual compounds and genes can recur across partitions. The two relation
graphs overlap and are evaluated separately.

Imputation, scaling and PCA are fitted to unique training entities. The classifier
uses all cross-modal products and main effects, with fixed logistic-regression
C=1. The main comparison uses 13 molecular representations with a common ESM2
gene representation; two additional CGP-Align gene-feature controls are also
included. Published source tables are in `figures/figure4/source_data/`.

## Thirty toxicity endpoints

```bash
python reproduction/encoding/run_tox30.py --inputs ../cgp-benchmarks --output ../cgp-results/tox30 --preflight-only
python reproduction/encoding/run_tox30.py --inputs ../cgp-benchmarks --output ../cgp-results/tox30 --workers 1
```

Nine representations use the same two repeats of five outer scaffold folds.
Four ExtraTrees configurations are selected by pooled inner-fold AUROC using
two inner scaffold folds. AUPRC is computed as average precision on pooled
outer-fold predictions, averaged across repeats, then equally across endpoints.
RDKit features precede embedding features in concatenated toxicity inputs.
The preflight checks all 300 outer partitions and 600 inner partitions against
the supplied assignments. The complete run fits 24,300 classifiers.

## Five focused toxicity endpoints and PRISM

Download the original `downstream-inputs` and `downstream-embedding-cache` groups
as described in the main README, then run:

```bash
python reproduction/encoding/run_quick_benchmarks.py --inputs ../cgp-extras/downstream --features ../cgp-benchmarks/baseline_features --embeddings ../cgp-extras/embeddings --output ../cgp-results/extended --task all
```

This adds ChemBERTa and MoLFormer comparisons and checks the unchanged RDKit
toxicity and CGP-Align PRISM results. The five-endpoint comparison uses the
original stratified folds and fixed classifier settings; it is separate from
the nested 30-endpoint benchmark. Current combined results, including the
original methods, are in `figures/summary/`.

Downstream MoLFormer features use deterministic evaluation. The earlier matched
retrieval benchmark uses its originally encoded features, preserving the reported
comparison. `provenance/` contains the original encoding and job scripts, including
their machine-specific paths; it is not a set of portable entry points.

## Figures

Install R packages `ggplot2`, `dplyr`, `tidyr`, `patchwork`, `scales`, `png`,
`svglite`, `ragg` and `ggrepel`. Install Arial and the `rsvg-convert` executable
(or set `RSVG_CONVERT` to its path). Tested with R 4.5.3 on Linux.

```bash
Rscript scripts/plot_figure3.R ../cgp-results/figure3
Rscript scripts/plot_figure4.R ../cgp-results/figure4
Rscript scripts/plot_figure5.R ../cgp-results/figure5
Rscript scripts/plot_figure6.R ../cgp-results/figure6
```

SVG is the common source for PDF and PNG conversion, preserving font layout.
Figure 4 has three main panels; the PCA is supplied as Figure S9 and the
64-dimensional comparison as Figure S8. Figure 5 combines the 30-endpoint
comparison, its aggregate and paired results, the five-endpoint heatmap and
structural-alert complementarity. Figure 6f compares response differences and
6g compares retrieval with query- and chemical-similarity-matched controls.
The source tables preserve the manuscript values; plotting does not refit models.
