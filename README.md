# CGP-Align

Phenotype-anchored representation learning linking chemical and genetic perturbations.

Associated manuscript: **CGP-Align links chemical and genetic perturbations through phenotype-anchored representation learning**.

## Training objective

The selected manuscript objective is `L_compound + L_ORF + L_CRISPR`, with all three weights equal to 1. The training wrapper uses this objective and a distinct run name. Three strict-preprocessing runs have completed 300 joint epochs each, with test evaluation and updated Figure 3–6 and S1–S5. See [new results](docs/FINAL_ARCHITECTURE_RESULTS.md). Historical weights retain their original objective. See [objective provenance and migration](docs/TRAINING_OBJECTIVE.md).

The current Figure 3–6 and S1–S5 use the submission architecture: one 512-unit hidden layer in gene/profile MLPs and no LayerNorm in the compound projection. Use the versioned runtime described in the results document; historical training scripts remain unchanged.

## Current release

The [final-architecture release](https://github.com/sundezhi3799/CGP-Align/releases/tag/final-architecture-20260920)
provides three primary models, nine branch initializers, six S3 ablation models,
protein features and strict seed-specific prepared data. See
[download and evaluation commands](docs/FINAL_ARTIFACTS.md) and
[Figure 4–6/S2/S3 results](docs/DOWNSTREAM_RESULTS.md).

```bash
python tools/download_final_artifacts.py --output ../CGP-Align-artifacts
python tools/prepare_final_architecture_runtime.py --output ../CGP-Align-final-runtime
python ../CGP-Align-final-runtime/tools/evaluate_final_artifacts.py --artifacts ../CGP-Align-artifacts --seed 41 --output ../CGP-Align-evaluation/seed41
```

Install the core dependencies first (see Quick verification below). The default
download contains seed-41 prepared data and all primary/branch checkpoints.
The packaged three-seed evaluations matched all 489 shared numeric metric fields within 1e-5 absolute tolerance
in an isolated runtime. Raw upstream reconstruction of every downstream input
and baseline remains outside this validation; see the documented scope.

## Training archive audit and historical weights

The original primary checkpoints and two missing compound preprocessing scripts
have been recovered. **The historical three-seed pipeline reuses a seed-13-fitted
preprocessing matrix across later train/test splits.** See the
[training archive audit](docs/TRAINING_ARCHIVE_AUDIT.md) for measured overlap and
required reanalysis. These artifacts preserve the original experiments; they do
not establish strict held-out preprocessing or a complete reproduction release.

```bash
python tools/download_models.py --output-dir checkpoints/primary
```

Downloads are verified against the exact original SHA256 hashes. These are historical checkpoints. Use the final-architecture release above for
the current weights and matching prepared data.

See [the strict reanalysis workflow](docs/STRICT_REANALYSIS.md) for the seed-31
pilot that refits corrections and retrains every branch from scratch.

## Quick verification

From this directory, check the correspondence between archived run metrics and figure source data (Python standard library only):

```bash
python tools/verify_reference_results.py
```

To exercise the actual research model with small synthetic inputs:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r environment/requirements-core.txt
python tools/smoke_test_model.py
```

The synthetic check covers graph, ORF, CRISPR and profile encoding, multi-positive loss, gradients and checkpoint round-trip. It does not reproduce manuscript performance.

To recompute article-level summaries from the archived processed tables (base R):

```bash
Rscript paper_snapshot/scripts/reproduce_article_results.R
```

Outputs go to `paper_snapshot/results/`. This verifies calculations from saved source data, not upstream data processing or model training.

## Revised Figure 3

The September 20 Word revision uses strict-model results throughout Figure 3 and hidden phenotype neighbours in panel H. Redraw all eight panels from the included processed tables:

```bash
Rscript environment/install-figure3.R
Rscript figures/figure3/reproduce.R
python tools/verify_figure3_sources.py
```

See [Figure 3 instructions and provenance](figures/figure3/README.md). This redraw reproduces the revised graphic; it does not rerun training or upstream inference. The August snapshot remains unchanged. See [the portability update](docs/PORTABILITY_UPDATE.md) for new evaluation options and remaining blockers.

## Revised supplementary figures

Supplementary Figures 1, 4 and 5 now use the strict three-seed logs and metrics.
See [reproduction instructions](figures/supplementary_strict/README.md) and the
[consistency audit](docs/SUPPLEMENT_STRICT_AUDIT.md), and [the current final-architecture results](docs/FINAL_ARCHITECTURE_RESULTS.md).
Current S2/S3 are included with the downstream figures.

## Contents

- `scripts/`: original preprocessing, training, evaluation and downstream analysis scripts with their local Python dependencies.
- `cgp_align/`: shared research utilities.
- `tools/`: portable verification and training-command utilities.
- `paper_snapshot/`: processed source tables and numerical-summary script from the 2026-08-31 article-results package, retaining its own license.
- `reference_metrics/`: three primary run-level retrieval records.
- `figures/figure3/`: revised A-H plotting workflow, processed data and checkpoint provenance.
- `revision_candidates/hidden_neighbours/`: earlier September candidate tables retained for traceability.
- `manifests/`: source file hashes, model provenance, dependencies and reproduction coverage.
- `docs/`: workflow, data requirements and release audit.

The adopted compound-profile comparison uses **encode each replicate, then average
and L2 normalize** for all four methods. See the [protocol and sensitivity
record](docs/STRICT_MATCHED_BENCHMARK.md); these results now supply Figure 3F/G. Historical Figure 3 sources remain archived.

## Training and analysis

Read [REPRODUCING.md](docs/REPRODUCING.md) and [DATA.md](docs/DATA.md). The scripts retain research interfaces; `--help` documents their arguments. Current prepared arrays, exact splits, initializer checkpoints and final weights are distributed through the versioned release; use the staged final runtime. Historical code and records remain labelled as archival.

## Licensing and citation

Original CGP-Align code and its associated documentation are licensed under the [MIT License](LICENSE). Existing packaged source data and scripts in `paper_snapshot/` retain the terms in [paper_snapshot/LICENSE](paper_snapshot/LICENSE). Third-party materials retain their own terms; see [LICENSE_STATUS.md](LICENSE_STATUS.md) for scope and outstanding attribution checks.

Use the manuscript title above for identification. A formal software citation with authors, release version and persistent identifier will be added when those publication fields are confirmed.
