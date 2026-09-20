# Canonical training objective

Selected by the author on 2026-09-20:

`L = L_compound + L_ORF + L_CRISPR`.

All three weights are 1; each branch uses symmetric multi-positive InfoNCE.
The reference `cgp_align/losses.py` is an unchanged copy of the submission package
`zenodo_common_review_package_20260811/model_code/cgp_align/losses.py`.

`tools/train_primary.py` explicitly selects split ORF/CRISPR branches,
`--gene_branch_loss_reduction sum`, compound/gene weights 1, and zero auxiliary
loss weights. New run names start with `cgp_align_equal_three_term_seed`.
The gene term in training logs is now the SUM of raw ORF and CRISPR losses.
Within each branch, the forward/reverse contrastive directions remain averaged.
For a batch containing only one gene modality, that branch retains weight 1.

For historical research reproduction only, the trainer accepts
`--gene_branch_loss_reduction mean`. This mode does not implement the selected
three-term objective. Existing checkpoints lacking this config field must be
interpreted from their archived training provenance, not the new parser default.

## Results requiring regeneration

Historical model releases and the September strict retraining results predate
this change. Do not relabel or overwrite their checkpoints, logs, source CSVs or
figures. Joint training, checkpoint selection and evaluations must be rerun under
the selected objective, using strict per-seed preprocessing and compatible
branch initializers. Keep the selected encode-then-average profile aggregation.
Updated Figure 3 and S1/S4/S5 are pending these runs. Other manuscript figures
still require their own checkpoint provenance audit.

The manuscript's three-term sum formula is retained. No training run has been
launched by this code update. No claim is made that old scores remain unchanged.

## Verification

`python tools/test_training_objective.py` compares both losses and gradients of
the production loss functions against the submission reference with duplicate
positive IDs, unequal modality counts and an absent modality. It also checks the
legacy mean mode and parser/wrapper wiring.
