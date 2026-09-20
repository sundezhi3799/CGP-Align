# Final architecture artifacts

Release: [final-architecture-20260920](https://github.com/sundezhi3799/CGP-Align/releases/tag/final-architecture-20260920).
The manifest `manifests/final_artifacts_20260920.json` records SHA256 and size for
every compressed part and every extracted file. These are the models used in
the September 20 revision, with selected joint epochs 220, 90 and 160 for seeds
31, 37 and 41. Historical model downloads refer to different checkpoints.

## Download and evaluate

Run from the repository root. The downloader requires only the Python standard
library. The default selection downloads all main/branch weights, protein
features and seed-41 prepared data (about 10.3 GB compressed). Keep additional
space for extracted arrays; the cache is retained for checksum verification.

```bash
python tools/download_final_artifacts.py --output ../CGP-Align-artifacts
python tools/prepare_final_architecture_runtime.py --output ../CGP-Align-final-runtime
python -m pip install -r environment/requirements-core.txt
python ../CGP-Align-final-runtime/tools/evaluate_final_artifacts.py \
  --artifacts ../CGP-Align-artifacts --seed 41 --output ../CGP-Align-evaluation/seed41
```

The runtime must be staged before evaluation: root research scripts preserve
the historical implementation. The portable launcher replaces archived server
paths with the downloaded paths, loads the matching joint checkpoint and
evaluates the complete held-out test partitions. Results are written as
`joint_test_metrics.json`. Linux with a CUDA-enabled PyTorch installation is
the validated environment; `--device cpu` is available but much slower.

For all three seeds, download both additional groups and evaluate each seed:

```bash
python tools/download_final_artifacts.py --output ../CGP-Align-artifacts \
  --groups prepared-seed31 prepared-seed37
python ../CGP-Align-final-runtime/tools/evaluate_final_artifacts.py \
  --artifacts ../CGP-Align-artifacts --seed 31 --output ../CGP-Align-evaluation/seed31
python ../CGP-Align-final-runtime/tools/evaluate_final_artifacts.py \
  --artifacts ../CGP-Align-artifacts --seed 37 --output ../CGP-Align-evaluation/seed37
```

`--verify-only` checks extracted files. `--offline --cache PATH` verifies and
extracts previously downloaded parts without network access. Extraction checks
every archive member against the manifest, disallows links and unexpected paths,
and verifies the content hash before installing a file.

For S3, download `--groups ablation-models` and evaluate using
`--variant no_source_indicator` or `--variant no_pretraining` together with the
matching `--seed` and a separate output directory. The variant's embedded
configuration controls the architecture switch; the same prepared test data
are used for each matched seed.

## Contents

- `final-models`: three joint checkpoints and nine branch initializers, with
  their exact training configurations.
- `prepared-seed31`, `prepared-seed37`, `prepared-seed41`: replicate profile
  arrays, entity/replicate metadata, cold-compound/cold-gene split files,
  feature columns, fitted normalization/correction parameters and fit audits.
  Each seed has its own training-fitted corrections.
- `protein-features`: the ESM2-derived protein feature matrix and its summary,
  preserving the original row order used by gene metadata.
- `ablation-models`: six selected S3 checkpoints and configurations, covering
  no source indicator and no branch pretraining for each of the three seeds.

Each group is a gzip tar stream split into parts of at most 1 GiB. The downloader
streams the parts in numeric order without creating an additional combined tar.
Never extract an individual multi-part segment as a standalone gzip archive.
The manifest's `code_commit` identifies the training implementation; the release
tag additionally includes artifact tools and the downstream results update.

## Training from prepared inputs

The staged runtime also includes a portable training launcher:

```bash
python ../CGP-Align-final-runtime/tools/train_final_artifacts.py \
  --artifacts ../CGP-Align-artifacts --seed 41 --stage all \
  --output ../CGP-Align-retraining/seed41 --dry-run
```

`--dry-run` validates the four configurations against their actual argument
parsers and writes commands without training. Remove it and choose a new output
directory to execute three branch pretraining stages followed by joint training
sequentially. `--stage joint` instead starts from the released branch initializers.
`--variant no_source_indicator` or `--variant no_pretraining` selects the S3
control. The launcher's configuration round-trip was checked for all four stages;
it has not been used for an additional full 1,200-epoch training rerun.
Both packaged ablation variants were also reevaluated for seed 41, matching
their six Top-10 summary values within 1e-5. See `ablation_portable_audit.json`.

## Validation and scope

The release archives were extracted into a separate directory and loaded using
a separately staged runtime. The three-seed evaluations matched all 489 shared
numeric metric fields within 1e-5 absolute tolerance in the same GPU environment. See
`revision_candidates/downstream_20260920/portable_evaluation_audit.json`.
Seed 41 matched exactly. Seven numeric fields for seeds 31/37 differed, with
maximum absolute difference 0.000001571; these do not change the reported
rounded values. The differences are retained in the audit rather than silently
replacing the reference results.
This verifies checkpoint/data packaging and path portability, not a second
training run or an independent reconstruction from raw Cell Painting downloads.

The release contains the prepared inputs for intrinsic retrieval and training.
Figure 4–6 additionally require original relation, ToxRIC and PRISM inputs and
baseline feature resources. Their current computed source tables, results,
analysis scripts and protocol details are available in
[the downstream reproduction guide](DOWNSTREAM_RESULTS.md). The earlier
`paper_snapshot` is an explicitly historical archive.

## Attribution

Original CGP-Align software and released CGP-Align checkpoints use the repository
MIT License. Cell Painting Gallery inputs retain their
[CC0 terms](https://github.com/broadinstitute/cellpainting-gallery/blob/main/LICENSE);
please follow the Gallery's [citation guidance](https://broadinstitute.github.io/cellpainting-gallery/citing.html)
and cite the JUMP/cpg0016 resource. Third-party molecular annotations, protein
inputs and foundation models retain their source terms; MIT does not relicense
those sources. This release redistributes derived prepared features, not raw
microscopy images or the ESM2 foundation-model weights.
