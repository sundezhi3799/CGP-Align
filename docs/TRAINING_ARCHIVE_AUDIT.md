# Training archive recovery and preprocessing audit

The original training archive was inspected on 2026-09-18. All three primary
checkpoints were recovered with matching epoch and validation score. Their exact
SHA256 digests and sizes are recorded in `manifests/models.json`. Each file is
34,777,442 bytes; all three load strictly into the published four-branch model
(8,684,100 parameters). The checkpoint configs are archived separately for review.

The two previously missing compound preprocessing scripts were recovered verbatim
and included with source hashes. Their recovery establishes source availability,
not a new execution of raw-data preparation.

## Important preprocessing discrepancy

The seed-31/37/41 compound and gene directories link to the same corrected feature
arrays. The correction audit and original split files identify seed 13 as the
split used to fit the normalization and plate/well correction. The later seeds
change the training/validation/test assignments without refitting those arrays.

Among 11,578 held-out compounds in each later split, the number present in the
earlier correction-fit set is 9,283 (seed 31), 9,215 (seed 37), and 9,233 (seed 41).
Gene split files also have overlap: 1,853/2,325, 1,880/2,331, and 1,868/2,370
respectively, before the loader's additional filtering. Machine-readable counts
are in `manifests/preprocessing_split_audit.json`.

These are unsupervised preprocessing overlaps. They are not evidence that downstream
target/toxicity/PRISM labels were used for training, nor do they quantify performance
inflation. They do mean that the historical three-run pipeline cannot currently be
described as fitting every preprocessing step exclusively on each run's own training
partition. An entity-held-out model split alone does not remove this overlap.

To audit the original workspace (exit code 2 deliberately flags overlap):

```bash
python tools/audit_preprocessing_splits.py --data-root /path/to/research/workspace --output outputs/preprocessing_split_audit.json
```

The audit follows the original feature-file symlinks and confirms identical entity
tables before comparing row indices. Copying a dataset without its original
correction-fit splits loses information needed for this audit.

## Required resolution

Preserve these checkpoints and scores as historical artifacts. For a strict
held-out evaluation, construct each seed's split before fitting normalization and
plate/well statistics, apply those statistics to validation/test data, then rerun
the affected branch pretraining, joint training and evaluations. Report the new
results separately and update the manuscript consistently. The size of any effect
is unknown until this comparison is performed; relabelling files or changing only
the evaluation script cannot resolve the methodological issue.

Prepared profile arrays total approximately 9.8 GB before compression and are not
yet provided as public downloads. Initializer weights, feature ordering, protein
embeddings, raw-data versions and downstream provenance also remain part of the
release work. No full retraining claim is made by this recovery update.

## Platform checks

The original Linux checkpoints contain pickled `PosixPath` objects. The portable
loader maps stored concrete path objects to pure path objects, enabling Windows
loading without rewriting the checkpoints. Both Linux and Windows path variants
have regression coverage. Checkpoints still use pickle and must be trusted.

The archive's older PyTorch 1.13 environment does not accept
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. For that environment, clear this
variable explicitly before invoking evaluation; newer environments should be
validated separately. This is an environment compatibility issue, not a change
to the trained weights.

A server-side inference check using the public checkout, the actual seed-31 checkpoint and original arrays completed on 64 held-out compounds and 64 held-out gene entities (one negative-sampling repeat). This confirms execution on real inputs, not agreement with the full manuscript metrics. The full three-seed evaluation and strict-preprocessing retraining have not been rerun.

## Public weight archive

The three exact historical checkpoints are supplied in the `historical-models-20260918` prerelease. Use `python tools/download_models.py` to download and verify them. This archive preserves the original experiment; it is not a corrected-preprocessing model release. `manifests/prepared_data_artifacts.json` records exact sizes, shared array targets and SHA256 for 40 prepared-input paths on the training host. Public prepared-input URLs remain unavailable.
