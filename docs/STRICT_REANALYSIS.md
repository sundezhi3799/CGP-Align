# Strict preprocessing reanalysis

The historical weights remain an archive, not corrected results. The seed-31
pilot preserves the original entity splits, architectures, hyperparameters and
300-epoch budgets while fitting project-level z-score and plate/well corrections
only on the seed-31 training rows. Public upstream profile processing is unchanged.

`tools/prepare_strict_profiles.py` starts from the uncorrected common-3180
profiles. It verifies metadata alignment and disjoint folds, saves fitted
parameters and input/output hashes, and refuses to overwrite an output directory.
`tools/test_strict_preprocessing.py` checks that changing held-out values cannot
change fitted parameters or training outputs, and compares the correction math
with the original implementation for identical fit rows.

The pilot controller `tools/run_strict_seed31.py` trains compound, ORF and CRISPR
branches from scratch on three GPUs, waits for all three processes to finish
successfully, then trains and evaluates the joint model using their new best
validation checkpoints. It never initializes from historical trained weights.
Archived initializer configurations provide the original training recipes;
runtime metadata dictionaries are converted back to their CLI mode values.

Example for the original research server (requires the raw data and frozen
protein embeddings; these inputs are not bundled in this repository):

```bash
python tools/run_strict_seed31.py \
  --research-root /home/sdz/projects/cgp_align_cpg_full \
  --run-root /data3/sdz/cgp_align_strict_20260918/seed31 \
  --gpus 0,1,2
```

Use a fresh run directory. The controller writes `status.json`, preprocessing
audits, per-branch logs, configurations and checkpoints, then joint evaluation
outputs. It must run detached for long server jobs. A retained exclusive lock
prevents accidental duplicate launches; failed jobs require inspection rather
than automatic overwrite or resume. The controller does not launch seeds 37/41.

After the pilot, compare retrieval metrics with the historical seed-31 results,
verify model selection and downstream evaluation, then extend the same procedure
to seeds 37/41. Corrected manuscript figures and conclusions must use the new
completed experiments. No corrected performance result is claimed yet.
