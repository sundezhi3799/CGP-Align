# Release audit — 2026-09-18

Status: public development repository; not ready to claim end-to-end reproducibility.

## Evidence established

- Original `train_cgp_align_replicate.py` and its graph utilities are included, together with compound/gene pretraining code.
- The three `no_teacher_seed{31,37,41}_test_metrics.json` records identify the primary common-3180 four-branch checkpoint family. They report selection epochs 50, 110 and 120. These are not the old seeds 13/17/23 benchmark.
- Source provenance records hashes of every imported file. Scientific values have not been regenerated or replaced during assembly.
- The numerical-summary package uses the August 31 snapshot. The revised September Figure 3 now has a separate standalone redraw workflow.

## Required before submission

1. **Raw-data reconstruction.** Two scripts referenced by the historical preprocessing launcher are absent from this workspace: `build_compound_mocop_replicate_dataset.py` and `build_compound_mocop_corrected_profile_dataset.py`. Recover the actual scripts from the training host or original archive. Document the exact source/plate/well correction order; a similarly named well-only workflow is not sufficient evidence.
2. **Artifact delivery.** Recover exact prepared tables, entity ordering, protein embeddings, split files, three initializers per run and selected final checkpoints. Add public URLs, byte sizes and SHA256 hashes. Existing `models.json` paths are provenance, not downloadable assets.
3. **Training configuration.** The portable joint-training launcher is transcribed from `run_raw3180_staged4_branchinit_joint_no_anchor_300_saveval.sh`; original defaults are inherited. Compare its fully expanded configuration with each actual checkpoint before declaring exact reproduction. Preserve pretraining configuration and normalization provenance as well.
4. **Downstream model identity.** Some toxicity scripts name `CGP_sep_k32`, while the primary retrieval uses the no-teacher family. Determine which checkpoints actually produced Figures 4–6 and document any differences. Do not silently relabel model families.
5. **Matched baseline pipeline.** The matched benchmark table builder is present, but final MoLFormer/ChemBERTa/fingerprint training commands, feature caches and split alignment remain to be recovered and verified.
6. **Figure 3H revision.** The September 18 Word revision now uses hidden phenotype neighbours. `figures/figure3/` supplies plotting code, processed inputs and provenance. The August paired-gain archive is preserved. Upstream inference remains dependent on unavailable arrays/checkpoints.
7. **Supplementary checkpoint provenance.** Figure S1/PCA metadata give seed-41 score 0.5431782488809668; the primary retrieval metadata give 0.5441922277613108. Trace the checkpoints/configuration rather than assuming equivalence.
8. **Runtime portability.** Original analysis scripts may expect workspace-relative output paths and path-bearing checkpoint configs. Complete parameter overrides and validate from an independent checkout with only public inputs.
9. **Rights and metadata.** MIT selected by the maintainer on 2026-09-18; the full code license is included. Public repository: `https://github.com/sundezhi3799/CGP-Align`. Complete third-party attribution checks, upstream dataset versions/access terms, release tag and archival DOI.
10. **Independent reproduction.** Run the full selected pipeline, compare metrics with reference records, report tolerances, hardware, time and actual failures. The available lightweight checks do not establish full retraining success.

## Excluded from release candidate

Manuscript Word files, correspondence, credentials, local environment files, remote connection settings, caches, raw external repositories, exploratory launchers, full raw datasets and model binaries were not copied wholesale.

## Submission statement

Only after publication and validation should the manuscript and submission portal assert public availability of code supporting all main findings. The repository audit must be resolved or accurately scoped; do not state that every figure can already be reproduced end to end.

## Subsequent portability work

See [PORTABILITY_UPDATE.md](PORTABILITY_UPDATE.md) for completed fixes and validation. The exact primary checkpoint/data paths are absent from the current maintainer workspace; recovery from the training host remains necessary.
