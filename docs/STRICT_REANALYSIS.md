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
than automatic overwrite or resume. The generalized `tools/run_strict_seed.py --seed 37` (or `--seed 41`) uses that seed's archived branch and joint recipes and its own original split. Each run needs a separate run directory and three distinct available GPUs.

After the pilot, compare retrieval metrics with the historical seed-31 results,
verify model selection and downstream evaluation, then extend the same procedure
to seeds 37/41. Corrected manuscript figures and conclusions must use the new
completed experiments. No corrected performance result is claimed yet.

## Joint-stage mixed precision fix

The initial seed-31 joint launch failed before completing epoch 1 on PyTorch 2.1:
LayerNorm produced float32 hidden states while message projection used float16.
Message aggregation now casts messages to the accumulator dtype before index_add_.
This preserves full-precision behavior and architecture/checkpoint compatibility.
The six-layer forward/backward regression check covers autocast and full precision;
run `python tools/test_gnn_amp.py --device cuda:0` on the training server.
`tools/restart_strict_joint.py` checks completed branches and preprocessing audits,
then launches a fresh joint attempt, preserving the failed attempt and its logs.

## Seed-31 completed pilot

Seed 31 completed all branch training, 300 joint epochs and test evaluation.
The best joint checkpoint was selected at epoch 70 by validation harmonic mean
Top-10 (historical checkpoint: epoch 50). The corrected result is stored separately
at `reference_metrics/strict/primary_seed31.json`; historical reference files are
unchanged. Run `python tools/compare_strict_metrics.py --seed 31` to verify the
same test counts, sampled retrieval protocol and selection metric before comparison.

For the 1:100 sampled protocol, historical to strict Top-10 percentages are:
- Compound to profile: 83.5222 to 83.8737 (+0.3515 percentage points).
- Profile to compound: 51.1727 to 53.8830 (+2.7103 percentage points).
- Gene to profile: 68.9590 to 66.7358 (-2.2232 percentage points).
- Profile to gene: 37.4879 to 36.5192 (-0.9687 percentage points).
- Four-direction arithmetic mean: 60.2854 to 60.2529 (-0.0325 percentage points).

These are one-seed descriptive differences, not evidence of statistical equivalence
or the final three-seed manuscript result. This run also includes the documented
mixed-precision compatibility fix; numerical differences cannot be attributed
solely to preprocessing. Downstream analyses and baseline/ablation preprocessing
must be audited before replacing manuscript figures or claiming full reproduction.

For the remaining two seeds, launch separate controllers on disjoint GPUs:

```bash
python tools/run_strict_seed.py --seed 37 --research-root /home/sdz/projects/cgp_align_cpg_full --run-root /data3/sdz/cgp_align_strict_20260918/seed37 --gpus 0,1,2
python tools/run_strict_seed.py --seed 41 --research-root /home/sdz/projects/cgp_align_cpg_full --run-root /data3/sdz/cgp_align_strict_20260918/seed41 --gpus 3,4,5
```

Each controller refits its own preprocessing from the uncorrected profiles,
trains all three branches from scratch and then runs joint training/evaluation.
No seed-31 fitted preprocessing parameters or trained weights are reused.
