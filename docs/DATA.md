# Required data artifacts

No raw-data download URL is invented by this candidate. The manuscript identifies Cell Painting Gallery cpg0016 / JUMP, ToxRIC, PRISM Repurposing and external compound–gene relation resources. Exact file versions, access terms and hashes remain to be assembled.

## Prepared compound directory

The original trainer reads:

- `compound_mocop_entities.parquet`
- `compound_mocop_replicates.parquet`
- `compound_mocop_replicate_features.npy`
- `splits_compound_mocop.json` (key `cold_compound`)

## Prepared gene directory

- `gene_mocop_entities.parquet`
- `gene_mocop_replicates.parquet`
- `gene_mocop_replicate_features.npy`
- `splits_gene_mocop.json` (key `cold_gene`)

The protein feature directory must supply `protein_sequence_embeddings.npy` in the exact original gene index order. The ESM2-650M mean/CLS feature convention must be recorded with the sequence mapping and upstream model version.

Replicate tables use `entity_index` and `feature_index`. Input shapes, row order, feature order and entity metadata are part of the reproducibility contract. The paper uses common 3,180-dimensional morphology profiles and 256-dimensional output embeddings.

`make_raw3180_seeded_split_dirs.py` reconstructs splits from a fixed ordered entity table and uses symlinks for payloads. On Windows, symlink support may require Developer Mode; run in Linux/WSL or supply already prepared directories. Final archived splits should also be distributed directly and checked for entity/gene overlap.

## Delivery manifest to complete

For each source dataset, processed array, split, baseline feature matrix and checkpoint, record its purpose, original source, exact version, filename, size, SHA256, license, public download URL and producing command. Large files belong in a persistent artifact repository, with small manifests retained in Git.

## Historical normalization and split order

The original seeded directories reuse shared corrected arrays fitted on the earlier seed-13 split. This introduces overlap with later held-out entities during unsupervised preprocessing. See [TRAINING_ARCHIVE_AUDIT.md](TRAINING_ARCHIVE_AUDIT.md). Reusing `make_raw3180_seeded_split_dirs.py` on those corrected arrays reproduces the historical order; it does not fix the overlap. A strict reanalysis must split raw profiles before fitting per-seed corrections.
