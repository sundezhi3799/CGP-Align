# Submission-architecture results

Figure 3 and S1/S4/S5 now use the completed submission-architecture runs for seeds
31, 37 and 41, selected at joint epochs 220, 90 and 160. Nine branch pretraining
runs and three joint runs each completed 300 epochs on strict per-seed data.

The gene and profile MLPs have one 512-unit hidden layer followed by a 256-unit
output. The compound projection is 256–512–256 without LayerNorm. Pretraining
gene MLPs use GELU. The joint objective is the sum of the three equally weighted
symmetric multi-positive losses. No teacher term is enabled. Initializer loading
requires exact coverage and matching shapes for each mapped encoder.

The submission package did not include a full historical training recipe. These
runs use the documented strict-data training workflow with the submission model
architecture and loss; they are not claimed to recover an unavailable historical
training process. Compound branch pretraining uses the submission single-hidden-layer
profile encoder without a source embedding; the joint profile encoder includes
the three-valued source embedding and is freshly initialized.

## Retrieval results

Replicates are encoded separately, averaged per compound, then L2-normalized.
Each seed uses 11,578 test compounds. Numbers below are mean ± sample SD across
three seeds, after averaging the two directions within each seed.

- Sampled 1:100 Top-10: **68.8675% ± 1.1194%**.
- Sampled 1:1000 Top-10: **29.3089% ± 1.0427%**.
- Full-gallery Recall@10: **8.9394% ± 0.6953%**.
- Baseline 1:100 means: MoLFormer 59.7091%, ChemBERTa 56.7772%, Morgan 56.5935%.

The prior deeper-architecture equal-loss results were 68.8972%, 28.4112% and
8.0238%. Candidate-bank, prepared-data and replicate hashes match across this
comparison; all nine baseline records are unchanged. Changes involve multiple
architecture/pretraining corrections and optimization, not an isolated layer
ablation. The historical approximately 71.1% result uses different seeds and data
and is not a controlled comparator here.

## Reproducible records and runtime

`revision_candidates/final_architecture_20260920` contains metric JSONs,
source tables, checkpoint hashes, source-code hashes, the original submission
model package, four revised training modules, and the numerical-equivalence audit.
The audit passed 19 graph, output, gradient and multi-positive-loss checks.
The `training_configs/` directory contains all 12 branch/joint configurations
and three initializer-provenance records. The archived controller and audit
script preserve the actual server paths.

Root `scripts/` retains the historical implementation. Stage a separate runtime
before loading the current checkpoints:

```bash
python tools/prepare_final_architecture_runtime.py --output ../CGP-Align-final-runtime
```

Use the staged `scripts/train_cgp_align_replicate.py` with
`--gene_hidden_dims 512 --profile_hidden_dims 512`, separate ORF/CRISPR branches,
source embedding enabled, source adapters disabled, and
`--gene_branch_loss_reduction sum`. Use new compatible branch initializers,
not historical checkpoints. The staged compound pretrainer directly uses the
submission profile encoder. Full run configuration remains embedded in checkpoint
provenance and the archived controller; server paths need adapting to local data.
The staged evaluation tool accepts `--aggregation mean_encoded --replicate-data`.

Figure source checks run in the repository itself:

```bash
python tools/verify_figure3_sources.py
python tools/build_figure3_matched_sources.py --check
python tools/build_strict_supplement_sources.py --check
```

Current PNG/PDF/SVG exports are under `figures/figure3/current_exports` and
`figures/supplementary_strict/current_exports`. The manuscript update preserves
the original figure dimensions and paragraph/run formatting.

Figure 4–6 and S2/S3 remain outside this completed result update. Public delivery
of the new weights and prepared data is also outstanding; the historical model
download command does not provide these new checkpoints. Full-paper reproduction
is not yet verified.
