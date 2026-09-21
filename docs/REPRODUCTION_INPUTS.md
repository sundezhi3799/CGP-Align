# Recomputing the matched and downstream results

The supplementary [reproduction-inputs release](https://github.com/sundezhi3799/CGP-Align/releases/tag/reproduction-inputs-20260921)
adds the nine trained comparator checkpoints, molecular feature matrices,
seed-specific entity-mean profiles and frozen negative candidate banks, mapped
relation inputs, five toxicity task tables, baseline features, alert activation
matrix, PRISM response inputs and the exact frozen CGP embedding caches.
It supplements the [main model/data release](FINAL_ARTIFACTS.md).
SHA256 checks cover each compressed part and each extracted file.

## 1. Download inputs and stage the implementation

From the repository root:

```bash
# The first release supplies CGP models and strict prepared profiles.
python tools/download_final_artifacts.py --output ../cgp-artifacts \
  --groups final-models protein-features prepared-seed31 prepared-seed37 prepared-seed41

# The supplementary release supplies comparators and analysis inputs.
python tools/download_final_artifacts.py \
  --manifest manifests/reproduction_inputs_20260921.json --output ../cgp-extras \
  --groups matched-models-features matched-seed31 matched-seed37 matched-seed41 \
           downstream-inputs downstream-embedding-cache
python tools/prepare_final_architecture_runtime.py --output ../cgp-runtime
```

The supplementary download is approximately 6.4 GB compressed. Extraction retains
the downloaded parts; allow additional space for arrays and outputs. Relation
enrichment creates three score arrays of about 3–4 GB each. Use a new output
directory for each independent run. Keep all nine parts for each main-release
prepared-data group; the downloader joins streams automatically.

## 2. Figure 3F/G: evaluate all four methods

Use the main model evaluation environment in `FINAL_ARTIFACTS.md` (Python 3.10,
PyTorch 2.1.0, NumPy 1.26.4 and RDKit 2023.9.6 were used). Run once per seed:

```bash
python3.10 -m venv ../cgp-model-env
source ../cgp-model-env/bin/activate
python -m pip install -r environment/requirements-model-observed.txt
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

```bash
python ../cgp-runtime/tools/evaluate_published_matched.py \
  --artifacts ../cgp-artifacts --matched-artifacts ../cgp-extras \
  --seed 41 --output ../cgp-results/matched/seed41 --device cuda:0
```

Repeat for seeds 31 and 37. Each command loads the final CGP checkpoint and the
three matching comparator checkpoints, encodes each corrected replicate before
averaging, and uses the archived common candidate bank. All dataset paths are
relocated explicitly. The command writes metrics, ranks and test embeddings for
each method. Hard links save temporary disk space where available; copying is
used when the filesystem does not support links.

These are the trained projection/profile comparators using fixed MoLFormer,
ChemBERTa and RDKit-Morgan molecular inputs, not copies of the foundation-model
weights. Their training command records and complete prepared inputs are included.

To reconstruct a comparator's training command from its archived recipe:

```bash
python ../cgp-runtime/tools/train_published_matched.py \
  --artifacts ../cgp-artifacts --matched-artifacts ../cgp-extras \
  --seed 41 --method morgan --output ../cgp-results/retrain/seed41_morgan --dry-run
```

Choose `morgan`, `molformer_xl` or `chemberta` and seed 31, 37 or 41. Remove
`--dry-run` and choose a new output directory to run the 160-epoch recipe.
The dry run checks arguments against the actual training parser; it is not an
additional training replication. Primary-model training is documented in
[FINAL_ARTIFACTS.md](FINAL_ARTIFACTS.md).

## 3. Generate CGP embeddings from weights

In the same model environment:

```bash
python ../cgp-runtime/tools/reproduce_downstream.py --task embeddings \
  --inputs ../cgp-extras/downstream --artifacts ../cgp-artifacts \
  --output ../cgp-results/from_weights --device cuda:0
```

This encodes all compound and gene entities for three seeds and the toxicity and
PRISM compound lists for seed 41. It requires the three prepared-data groups.
The optional released `../cgp-extras/embeddings` cache records the exact inputs
used for the manuscript readouts. It can be used to reproduce those readouts
without rerunning neural inference. Tiny floating-point differences in regenerated
features can affect near-tied ranks or tree splits; both regenerated and archived
input paths are explicit rather than silently substituting one for the other.

## 4. Figure 4: mapped relations and PCA

Use the model environment:

```bash
python ../cgp-runtime/tools/reproduce_downstream.py --task relations \
  --inputs ../cgp-extras/downstream --artifacts ../cgp-artifacts \
  --embeddings ../cgp-extras/embeddings --output ../cgp-results/readouts
```

This runs all three seeds, broad and strict graphs, five relation sources,
leave-one-source-out classifiers and the fixed reference PCA/overlap panels.
To use regenerated inputs, pass `--embeddings ../cgp-results/from_weights/embeddings`.

## 5. Figures 5/6: pinned downstream environment

The toxicity/PRISM analyses originally used newer RDKit and scikit-learn versions
than the neural-model environment. RDKit version changes can alter canonical
SMILES and the eligible cohort. Create a separate environment rather than replacing
the model environment's packages:

```bash
python3.10 -m venv ../cgp-downstream-env
source ../cgp-downstream-env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements-downstream-observed.txt
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

python ../cgp-runtime/tools/reproduce_downstream.py --task toxicity \
  --inputs ../cgp-extras/downstream --artifacts ../cgp-artifacts \
  --embeddings ../cgp-extras/embeddings --output ../cgp-results/readouts
python ../cgp-runtime/tools/reproduce_downstream.py --task prism \
  --inputs ../cgp-extras/downstream --artifacts ../cgp-artifacts \
  --embeddings ../cgp-extras/embeddings --output ../cgp-results/readouts
```

The toxicity command refits all six representations under random/scaffold folds,
then refits the three alert readouts for all five CV seeds. The PRISM command
recomputes hit enrichment, U2OS activity, continuous retention and gene-neighbour
annotations from the released response matrices, structure features and embeddings.
The internal historical PRISM key `RDKit2D` denotes **RDKit-Morgan**, a 2059-vector
of Morgan-2048 plus 11 RDKit descriptors. The corrected Figure 6 and manuscript
use the accurate display name. Figure 5's genuine RDKit2D descriptors are a
separate baseline and retain their name.

Both NYAN toxicity CV modes use the official checkpoint cache in
`downstream/toxicity/scaffold_nyan.npz`; its filename records its earlier use,
not a separate scaffold-specific encoder. The original scaffold protocol skips
partitions whose train or test labels contain only one class: 24 evaluable folds
across five endpoints are retained for each representation, compared with 25
random folds. The same exclusions apply to every representation.

Alert Lift@50 uses a released secondary ordering for exactly equal predicted
scores (`original_alert_tie_order.npz`). It serializes the original Windows
NumPy score-sort order; the new predictions remain the primary sorting key.
This makes the original tie treatment portable across CPU/NumPy sort
implementations. The ordering was derived from predicted scores without using
labels and was checked against all 75 original runs. The CLI saves new
out-of-fold probabilities and ranks for inspection. `--task alerts` reruns just
this component; `--task toxicity` includes it automatically.

## Scope and upstream provenance

These commands begin from versioned prepared inputs, including baseline feature
caches and the PRISM overlap response matrix. They do not reconstruct all public
database snapshots, retrain foundation models, or repeat Cell Painting image
processing. The original preprocessing and feature-building code remains in
`scripts/`; upstream sources and license boundaries are listed in
[INPUT_ATTRIBUTION.md](INPUT_ATTRIBUTION.md). No author-private directory is
required by the portable commands above.

Current source-table redraw commands remain in [DOWNSTREAM_RESULTS.md](DOWNSTREAM_RESULTS.md).
Validation records distinguish recomputation from exact frozen caches and fresh
neural inference; numerical differences are retained in the audit.
