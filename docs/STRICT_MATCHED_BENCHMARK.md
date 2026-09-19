# Strict matched compound-profile comparison

This replaces the legacy seed-13/17/23 comparison as a new experiment. Do not
combine its scores with the primary replicate-gallery metrics. The old 71.1%
CGP comparator was a different model family and protocol.

## Fixed protocol before inspecting new baseline test scores

- Seeds 31/37/41 use the exact same compound IDs and train/validation/test
  assignments as their corrected primary runs. Each has 92,629 training,
  11,579 validation and 11,578 test compounds.
- Target is the arithmetic mean of a compound's strictly corrected replicate
  profiles. Every model encodes that mean vector once. This is **not** the mean
  of encoded replicate vectors. No statistics are fitted on validation/test rows.
- Molecular caches (Morgan/RDKit, MoLFormer, ChemBERTa) must have identical IDs,
  SMILES and order, complete coverage, finite values and no all-zero rows. Record
  SHA256 and dimensions. Cache lineage predates this run; this check verifies
  alignment/content validity, not a new audit of foundation-model pretraining.
- Evaluation is bidirectional, test-only, one positive per query: full gallery
  plus 100 and 1,000 unique sampled negatives, 10 repeats. Candidate banks are
  frozen per seed, shared across models/directions, exclude the positive, and
  use the first 100 of the 1,000 candidates for the smaller setting. Similarity
  ties use ascending gallery row index. Save all per-query ranks.
- Use the existing validation-selected strict CGP checkpoints (epochs 70,150,60).
  Do not choose a different CGP checkpoint using these new test results.
- The three molecular-profile comparators retain the archived recipe: original
  `train_cgp_align_base_intrinsic.py`, 160 epochs, batch 512, AdamW learning rate
  1e-4 and weight decay 1e-4, dropout 0.1, compound-profile loss only. Select via
  validation compound-to-profile Recall@10 every five epochs, capped at 2,048
  validation queries, as in that trainer. Recipes are stored in manifests.

This is a framework comparison with matched data partitions and **evaluation**
targets, not a compute-matched or encoder-only ablation. CGP retains its earlier
replicate-level multimodal training and additional branch pretraining; comparators
train on entity means. Their training budgets and selection objectives differ
and must be disclosed in the figure legend/methods. Molecular caches can be
reused, but all comparator projection/profile models are trained from scratch.

## Execution

`tools/run_matched_benchmark.py` prepares three independent cohorts, then runs
one worker per GPU. Each worker evaluates CGP and trains/evaluates Morgan,
MoLFormer and ChemBERTa sequentially. Preparation validates raw IDs, strict fit
audit/split hashes, creates new mean profiles, and preserves all old data.
Run into a new directory; the controller refuses to overwrite an existing run.

```bash
python tools/run_matched_benchmark.py \
  --research-root /home/sdz/projects/cgp_align_cpg_full \
  --strict-root /data3/sdz/cgp_align_strict_20260918 \
  --run-root /data3/sdz/cgp_align_matched_20260919 \
  --gpus 3,4,5
```

Do not launch onto occupied GPUs. The root status file tracks preparation and
workers; each `seed*/status.json` identifies its current method/stage, with logs,
recipes, trained checkpoints, test embeddings, per-query ranks and metric JSON.
The frozen 10-repeat negative bank is approximately 463 MB per seed. Prepared
inputs and results are not automatically published by this controller.

`tools/test_matched_retrieval.py` independently checks full/sampled ranks using
stable sorting, including tied scores. Before full execution also validate the
actual prepared data with the original trainer's isolated two-epoch smoke mode.
Smoke checkpoints must never be used in the reported comparison.
