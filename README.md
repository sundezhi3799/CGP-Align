# CGP-Align

Code for **CGP-Align links chemical and genetic perturbations through phenotype-anchored representation learning**.

CGP-Align learns a shared representation of compounds, genetic perturbations and Cell Painting profiles through phenotype-based alignment.

## Installation

Create a Python 3.10 environment and install the dependencies:

```bash
conda create -n cgp-align python=3.10
conda activate cgp-align
git clone https://github.com/sundezhi3799/CGP-Align.git
cd CGP-Align
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

These commands use CUDA 11.8. For CPU execution, replace `cu118` with `cpu`. Run the commands below from the repository root. Toxicity and PRISM use a separate environment described below.

## Data and pretrained models

Download the pretrained models and prepared data:

```bash
python scripts/download_data.py --output ../cgp-data
```

The default download includes three joint models, nine branch initializers, protein features and the seed-41 profiles (about 10.3 GB compressed). The files are also available in the [model release](https://github.com/sundezhi3799/CGP-Align/releases/tag/final-architecture-20260920).

For seeds 31 and 37 and the ablation experiments:

```bash
python scripts/download_data.py --output ../cgp-data --groups prepared-seed31 prepared-seed37 ablation-models
```

For the baseline comparisons:

```bash
python scripts/download_data.py --manifest configs/downloads/reproduction_inputs_20260921.json --output ../cgp-extras --groups matched-models-features matched-seed31 matched-seed37 matched-seed41
```

For downstream analysis:

```bash
python scripts/download_data.py --manifest configs/downloads/reproduction_inputs_20260921.json --output ../cgp-extras --groups downstream-inputs downstream-embedding-cache
```

The [baseline and downstream data](https://github.com/sundezhi3799/CGP-Align/releases/tag/reproduction-inputs-20260921) total about 6.4 GB compressed. Downloads are checked against SHA256 hashes and extracted automatically. Allow space for both the archives and extracted files. Use `--verify-only` to check existing files, or `--offline --cache PATH` to use cached archives.

## Training

Run branch pretraining followed by joint training:

```bash
python scripts/train.py --artifacts ../cgp-data --seed 41 --stage all --output ../cgp-training/seed41
```

Use `--stage joint` to train from the supplied branch encoders. Reference configurations for seeds 31, 37 and 41 are in `configs/`. The training command reads `configuration.json` from `../cgp-data/models/seed<seed>/<stage>/`; edit those files to change the training settings. Data and output paths are set by the command-line arguments. Use a new output directory for each run; `--dry-run` prepares the run without starting training.

Joint training sums the compound, ORF and CRISPR alignment losses with equal weights. Gene and profile MLPs have a 512-unit hidden layer and a 256-dimensional output. Checkpoints are selected by validation HMean Top-10.

## Evaluation

Evaluate a pretrained model:

```bash
python scripts/evaluate.py --artifacts ../cgp-data --seed 41 --output ../cgp-results/seed41
```

Use `--seed 31` or `--seed 37` with the corresponding data. For ablations, add `--variant no_source_indicator` or `--variant no_pretraining`. Use `--device cpu` to evaluate on CPU.

### Baseline comparisons

Run the matched compound-profile benchmark:

```bash
python scripts/evaluate_matched.py --artifacts ../cgp-data --matched-artifacts ../cgp-extras --seed 41 --output ../cgp-results/matched/seed41
```

Profile replicates are encoded individually, averaged by entity and L2-normalized. All methods use the same candidate banks.

To train a baseline:

```bash
python scripts/train_baseline.py --artifacts ../cgp-data --matched-artifacts ../cgp-extras --seed 41 --method morgan --output ../cgp-training/morgan41
```

Available methods are `morgan`, `molformer_xl` and `chemberta`. Use a new output directory for each run. Add `--dry-run` to prepare the inputs and inspect the training command without running it.

### PCA

The checkpoints, encoder code and selected test profiles for Figure 3A/B are available in the [PCA release](https://github.com/sundezhi3799/CGP-Align/releases/tag/figure3-pca-20260921). Export the coordinates in the `cgp-align` environment:

```bash
python scripts/download_data.py --manifest configs/downloads/pca_inputs_20260921.json --groups pca-inputs --output ../cgp-pca
python -I ../cgp-pca/pca/run_pca.py --output ../cgp-results/pca --device cuda:0
```

The download is about 476 MB and includes the seed-41 epoch-120 model, its branch encoders and 1,200 test compounds and 1,200 test gene perturbations. Use `--device cpu` for CPU execution. Details of the PCA model and inputs are in the archive's README.

## Downstream analysis

The analyses use prepared profiles, mapped annotations and molecular features. Image processing and foundation-model training are performed upstream.

### Embeddings and compound-gene relations

Generate embeddings in the `cgp-align` environment. This command processes seeds 31, 37 and 41, so download `prepared-seed31` and `prepared-seed37` first using the command above. For seed 41 alone, add `--seeds 41`; pass the same option to relation analysis.

```bash
python scripts/analyze.py --task embeddings --inputs ../cgp-extras/downstream --artifacts ../cgp-data --output ../cgp-results/from_weights
```

Run relation analysis using the supplied embeddings:

```bash
python scripts/analyze.py --task relations --inputs ../cgp-extras/downstream --artifacts ../cgp-data --embeddings ../cgp-extras/embeddings --output ../cgp-results/relations
```

To use the generated embeddings instead, set `--embeddings ../cgp-results/from_weights/embeddings`. Relation analysis includes source mapping, enrichment and held-out-source classification. Its score arrays require several GB of disk space.

### Toxicity and PRISM environment

Create a separate environment for these analyses. The RDKit version affects molecule parsing and cohort selection.

```bash
conda create -n cgp-downstream python=3.10
conda activate cgp-downstream
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-downstream.txt
```

The commands below use the supplied embedding cache. To use your generated embeddings, replace the `--embeddings` path as above.

### Toxicity prediction

```bash
python scripts/analyze.py --task toxicity --inputs ../cgp-extras/downstream --artifacts ../cgp-data --embeddings ../cgp-extras/embeddings --output ../cgp-results/toxicity
```

Scaffold evaluation excludes single-class folds for all methods. This command also runs the structural-alert analysis below.

### Structural alerts

To run only the structural-alert analysis:

```bash
python scripts/analyze.py --task alerts --inputs ../cgp-extras/downstream --artifacts ../cgp-data --embeddings ../cgp-extras/embeddings --output ../cgp-results/alerts
```

Alert Lift@50 uses the supplied secondary ordering to break ties in predicted scores.

### PRISM drug response

```bash
python scripts/analyze.py --task prism --inputs ../cgp-extras/downstream --artifacts ../cgp-data --embeddings ../cgp-extras/embeddings --output ../cgp-results/prism
```

The PRISM input key `RDKit2D` denotes Morgan-2048 fingerprints plus 11 RDKit descriptors. The toxicity RDKit2D baseline uses a separate feature set.

## Acknowledgements

This work uses the following datasets, models and resources:

- [Cell Painting Gallery](https://github.com/broadinstitute/cellpainting-gallery): JUMP/cpg0016 profiles and molecular metadata, under CC0; cite the Gallery and JUMP resources.
- [MotiVE](https://github.com/carpenter-singh-lab/2024_Arevalo_NeurIPS_MotiVE): mapped compound–gene annotations (Arevalo, Su et al., NeurIPS 2024) and underlying DGIdb, DrugRep, Hetionet, OpenBioLink and PharMeBINet sources. Their [annotation archive](https://zenodo.org/records/18197517) documents mixed-source terms; retain source-specific conditions.
- [TOXRIC](https://toxric.bioinforai.tech/): toxicity labels; cite [Wu et al., Nucleic Acids Research](https://doi.org/10.1093/nar/gkac1074). TOXRIC and underlying assay terms apply; these inputs are not covered by the code's MIT license.
- [PRISM Repurposing](https://depmap.org/repurposing/): 19Q4 primary-screen response profiles; cite Corsello et al., Nature Cancer (2020). DepMap-generated data use CC BY 4.0.
- Structural-alert inputs retain RDKit, ChEMBL, Toxtree and literature-source terms. The supplied alert metadata records library IDs and references.
- MoLFormer, ChemBERTa and ESM2 inputs retain their source terms. Distributed comparator checkpoints are projection/profile models trained for this study; foundation-model weights and raw microscopy images are not included.

## License

The code is released under the [MIT License](LICENSE). Third-party data, models and libraries retain their respective licenses.
