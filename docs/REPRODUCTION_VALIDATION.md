# Reproduction validation, 21 September 2026

The supplementary release contains 149 extracted files in six groups, with
SHA256 values for all files and all nine compressed parts. It complements the
September 20 primary-model/prepared-data release. The portable runtime was
staged separately from the original training workspace, and analysis inputs
were loaded from extracted packages. See [commands](REPRODUCTION_INPUTS.md).

## What was recomputed

- **Matched retrieval (Figure 3F/G):** all 12 evaluations (four methods and
  three seeds), comparing 720 numeric metric fields. The nine comparator
  evaluations matched exactly. CGP had 48 small differences; maximum Recall
  difference was 0.0000086371 (0.000864 percentage points). Mean-rank differences
  are recorded separately and are not claimed to meet that Recall bound.
  Checkpoint and candidate-bank hashes matched.
- **Neural embeddings:** all three compound/gene embedding sets and the
  seed-41 toxicity/PRISM sets were regenerated from released checkpoints and
  prepared inputs. Gene vectors matched exactly; the largest compound-vector
  component difference was below 5e-7. Exact frozen caches are also released.
- **Figure 4:** 30 enrichment runs, 30 leave-one-source-out classifiers
  (60 validation/test records), the fixed-reference PCA and source-overlap
  counts were recomputed. With the exact frozen embeddings, all 600 classifier
  numeric fields matched exactly. Newly inferred embeddings retained identical
  enrichment Lift@50 and overlap counts; classifier AUPRC differed by at most
  0.000947 per validation/test record. PCA-coordinate differences were below
  5.5e-7. Both comparisons are retained.
- **Figure 5:** all six representations were refitted on random and scaffold
  partitions, yielding 150 and 144 evaluable fold records, respectively.
  All 120 endpoint AUPRC/AUROC means matched within 5e-16. All 75 alert readouts
  were also refitted; their 225 AUPRC/AUROC/Lift@50 values matched exactly after
  preserving the original equal-score ordering. The full 345-value comparison
  is in `toxicity_audit.json`; the portability details are below.
- **Figure 6:** 19 PRISM/mechanism CSV tables were recomputed. Hit counts,
  U2OS readouts, retention@50, gene-mechanism summaries and selected cases
  matched. Small score differences and RDKit-Morgan near-tie ranking differences
  remain: aggregate NDCG differences were at most 3.60e-6; individual-query
  NDCG10 differences reached 0.00755. The audit does not claim every query
  metric is bitwise identical.
- **Training launchers:** all nine comparator recipes passed argument
  round-trip validation against the original training parser. Primary/ablation
  training-launcher checks and checkpoint evaluations remain in the September
  20 audit. This pass did not repeat full training.

The machine-readable records are in
`revision_candidates/reproduction_20260921`. The `readouts` directory preserves
selected recomputed tables. The new receipts supplement the immutable
September 20 records; older receipts describe their original versions.

## Portability corrections

The model environment uses RDKit 2023.9.6; the toxicity/PRISM readouts use
RDKit 2025.9.6. Using the model environment for toxicity changed parsing of a
few molecules and failed cohort alignment. Separate pinned environments now
preserve the actual cohort definition.

The initial new NYAN random-fold wrapper selected a historical cache. It was
corrected to use the official checkpoint cache used in the paper; all five
NYAN tasks were rerun, and the unused historical cache was removed from this
release. No manuscript baseline was replaced by the failed wrapper output.

Original alert ranking used NumPy's default score sort. Equal-score ordering
differed between the original Windows execution and Linux even when AUPRC and
AUROC matched. The release serializes the original score-sort priority and
uses it only as a secondary key for equal newly predicted scores. Priorities
were derived from archived probabilities without labels, and reproduced all
75 original Lift@50 values in the original environment. The portable command
saves new probabilities and full rankings so the effect is inspectable.

The PCA stage was resumed after the wrapper gained an explicit embedding-cache
argument. The final CLI accepts both regenerated and released embeddings.

## Manuscript correspondence and scope

The Word v11 revision corrects Figure 6's structure baseline to RDKit-Morgan
(2048 Morgan bits plus 11 RDKit descriptors), records the tie-order convention
in Methods, and links both public releases in Availability of data and materials.
Figure 5's separate RDKit2D baseline is unchanged. Figure 6 plot values are
unchanged; only its display label was corrected. Figure 1/2 exact manuscript
schematics are included in `figures/schematics`.

The main manuscript retains 28 pages and the original paragraph/run/section
properties, field instructions and embedded image dimensions. Changed pages
were visually reviewed after rendering; `word_integrity.json` records the file
hashes and changed parts. The supplementary v5 document is unchanged.

Validation starts from versioned prepared inputs. It does not independently
rebuild all external database snapshots, rerun Cell Painting image processing,
retrain third-party foundation models, or update the earlier Zenodo archive.
Third-party inputs retain their [original terms](INPUT_ATTRIBUTION.md).
