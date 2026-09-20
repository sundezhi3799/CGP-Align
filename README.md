# CGP-Align

Phenotype-anchored representation learning linking chemical and genetic perturbations.

Associated manuscript: **CGP-Align links chemical and genetic perturbations through phenotype-anchored representation learning**.

## Training objective

The selected manuscript objective is `L_compound + L_ORF + L_CRISPR`, with all three weights equal to 1. The training wrapper uses this objective and a distinct run name. Three strict-preprocessing runs have completed 300 joint epochs each, with test evaluation and updated Figure 3/S1/S4/S5. See [new results](docs/FINAL_ARCHITECTURE_RESULTS.md). Historical weights retain their original objective. See [objective provenance and migration](docs/TRAINING_OBJECTIVE.md).

The current Figure 3/S1/S4/S5 use the submission architecture: one 512-unit hidden layer in gene/profile MLPs and no LayerNorm in the compound projection. Use the versioned runtime described in the results document; historical training scripts remain unchanged.

## Release status

This is a development version of the public [CGP-Align repository](https://github.com/sundezhi3799/CGP-Align), assembled on 2026-09-18. Full manuscript reproduction is not yet verified. See [the release audit](docs/RELEASE_AUDIT.md) for specific outstanding artifacts and version discrepancies. A versioned release and archival DOI will be added after the remaining reproduction work is complete.

The repository contains original research training and evaluation code, with documented portability fixes. Historical filenames are preserved for traceability. The primary retrieval records identify runs 31, 37 and 41 of the four-branch, common-3180 model. The older `reproduce_main_benchmark.py` is deliberately excluded because it implements a different benchmark setup.

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

Downloads are verified against the exact original SHA256 hashes. Prepared data
are not yet public; downloading weights alone is insufficient to rerun retrieval.

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
[consistency audit](docs/SUPPLEMENT_STRICT_AUDIT.md), including the outstanding
main-manuscript objective-weighting correction.

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

Read [REPRODUCING.md](docs/REPRODUCING.md) and [DATA.md](docs/DATA.md). The scripts retain research interfaces; `--help` documents their arguments. Prepared arrays, exact split files, initializer checkpoints and final model weights must be supplied separately. This candidate does not fabricate download URLs or substitute newly trained models for the published checkpoints.

## Licensing and citation

Original CGP-Align code and its associated documentation are licensed under the [MIT License](LICENSE). Existing packaged source data and scripts in `paper_snapshot/` retain the terms in [paper_snapshot/LICENSE](paper_snapshot/LICENSE). Third-party materials retain their own terms; see [LICENSE_STATUS.md](LICENSE_STATUS.md) for scope and outstanding attribution checks.

Use the manuscript title above for identification. A formal software citation with authors, release version and persistent identifier will be added when those publication fields are confirmed.
