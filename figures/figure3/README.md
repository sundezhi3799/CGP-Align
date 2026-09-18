# Revised Figure 3

This workflow redraws panels A-H from the archived processed tables. It matches
the September 18 Word revision: H is hidden phenotype-neighbour recovery, not
the paired-gain panel retained in the August `paper_snapshot`.

From the repository root:

```bash
Rscript environment/install-figure3.R
Rscript figures/figure3/reproduce.R
python tools/verify_figure3_sources.py
```

Outputs: `outputs/figure3/figure3.{png,pdf,svg,tiff}`, the plotted H table, and
`sessionInfo.txt`. An optional first argument changes the output directory; an
optional second argument changes the font (default Arial). For example:

```bash
Rscript figures/figure3/reproduce.R outputs/figure3 sans
```

The original plot layout and calculations are preserved. The changes make paths
relative to the script, send generated files to the output directory rather than
overwriting source data, and allow font substitution. Font changes can change
text layout. The recorded Windows/R 4.5.3/Arial run produced a pixel-identical PNG
to the revised manuscript image (4322 by 4440 pixels).

Known local runtime issue: this Windows R installation exits with native status
`0xC0000005` after writing all four images and session information. This also occurs
for PNG-only export. The generated PNG was checked visually and pixel-compared,
but a clean process exit is not yet verified. Do not suppress that failure in an
automated pipeline. A different R installation/platform still needs verification.

## Inputs and interpretation

- A-B: `figure2ab_four_branch_main_pca_source_data.csv` and its metadata. Historical
  filenames retain their original figure numbers. These are stored PCA coordinates;
  redrawing does not regenerate embeddings or fit PCA from model weights.
- C: direction-level run values and random-ranking expectations.
- D: full-gallery random-ranking negative-control source.
- E: branch summary means and standard deviations.
- F-G: matched and difficult compound-profile retrieval source tables.
- H: `hidden_phenotype_neighbour_recovery_compact.csv`, filtered to compound-to-compound,
  k=50 and the three displayed methods.

H uses 11,578 test compounds. Each query has 10 reference neighbours defined by
cosine similarity of entity-mean, normalized 3,180-dimensional CellProfiler
profiles; the query is excluded from its 11,577-candidate gallery. Enrichment
is mean precision@50 divided by 10/11,577. Values are 2.5677782, 4.8695794 and
9.5171779 for structure features, CGP-Align entity similarity and the profile-anchor
readout. The last method accesses gallery profile embeddings. These results
measure phenotype-defined neighbourhood recovery, not independent biological labels.

H uses the seed-41 `raw3180_4branch_cwcl_sep_k32_w100_seed41` checkpoint at epoch
120, score 0.5431782488809668. It is not the main three-run retrieval summary;
the primary seed-41 score is 0.5441922277613108. The exact relationship between
these checkpoint families remains unresolved. Do not transfer the old H paired
statistics or three-run uncertainty to this panel. Provenance JSON files preserve
the historical input paths; those paths are not public downloads.

## Upstream computation

With the exact checkpoint and prepared inputs available:

```bash
python scripts/export_raw3180_hidden_phenotype_inputs.py --checkpoint checkpoints/hidden_seed41.pt --compound-data-dir data/seed41/compound --gene-data-dir data/seed41/gene --gene-protein-embedding-dir data/protein --output-dir outputs/hidden_inputs
python scripts/run_hidden_phenotype_neighbour_recovery.py --embedding-npz outputs/hidden_inputs/final_raw3180_test_embeddings.npz --profile-input-npz outputs/hidden_inputs/final_raw3180_test_profile_inputs.npz --structure-features data/compound_structure_features.npy --positive-topn 10 --eval-ks 10,50,100,200 --output-dir outputs/hidden_recovery
```

The checkpoint and prepared arrays are not yet publicly supplied, so these upstream
commands are documented but not independently reproduced. The default protein-feature
path is only relevant to additional gene-scope baselines, not the displayed compound H.
Structure feature row ordering must match the compound indices exported with the embeddings.
