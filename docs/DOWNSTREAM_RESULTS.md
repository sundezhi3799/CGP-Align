# Figure 4–6 and S2/S3 update

The September 20 downstream update uses the final submission architecture and
strict per-seed preprocessing. Figure 4 aggregates seeds 31, 37 and 41. Figure
5/6 use seed 41 at epoch 160. S2 displays seed 41 and includes audits for all
three seeds. S3 compares three matched seeds for each of three variants.

## Current results

- Figure 4: mean cosine Lift@50 **2.45184** across five sources. The mean
  leave-one-source-out AUPRC is **0.48144**, versus **0.18891** for cosine ranking.
  The graph definitions and three model seeds are equally averaged.
- Figure 5: random-fold mean AUPRC **0.49367** and AUROC **0.86284**;
  scaffold-fold mean AUPRC **0.46593**, versus RDKit2D **0.43082**.
  CGP plus alerts has mean AUPRC **0.53351**, versus latent-only **0.48768**.
- Figure 6: at Morgan Tanimoto <0.20, top-50 response-correlation hits are
  **1965**, versus **353** for random ranking (**5.56657-fold**). The U2OS
  active-fraction gain over RDKit-Morgan is **0.15679** (bootstrap CI 0.13208–0.18417)
  over 187 active queries. The BIIB021/R547 example shares 39 top-50 genes.
- S2: all audited cross-split entity/gene-symbol overlaps are zero. Seed-41
  mean residual plate/well R² is 0.00117/0.00208 for compounds and
  0.00078/0.00188 for gene perturbations.
- S3: mean test HMean Top-10 is approximately **52.6%** for the full model,
  **50.7%** without the source indicator and **47.2%** without pretraining.

## Redraw the current figures

```bash
Rscript environment/install-downstream.R
Rscript figures/downstream_final/reproduce.R
python tools/verify_downstream_sources.py
```

R scripts, exact input tables and PNG/PDF/SVG/TIFF outputs are in
`figures/downstream_final`. PubChem structure PNGs used in Figure 6 are cached
alongside source tables, so redrawing does not require refetching structures.
Current supplementary table 1/4/6/7 CSVs are in its `supplementary_tables`
directory. Archived August tables retain their original values and filenames.

## Protocol details tied to the numerical implementation

Figure 4b ranks the full eligible galleries, collapsing gene-entity scores to
gene symbols by maximum cosine over ORF/CRISPR entities. Lift@50 is the observed
probability of at least one annotated hit divided by the mean random hit
probability without replacement; it is not top-50 positive-edge density.
Compound-query and gene-query lifts are averaged. Figure 4c/d instead average
encoded gene entities by symbol and L2-normalize before constructing pair
features. Logistic regression uses 5 sampled negatives per positive, up to
10,000 training positives and 3,000 evaluation positives, and sampling seed 13.
The PCA uses a fixed reference entity/pair subset reprojected through seed 41;
the labelled pairs are not selected as the new model's best predictions.

Figure 5a–c reuse the fixed non-CGP comparator records and rerun CGP features on
the same benchmark loader and five folds (seed 41; ExtraTrees 160 trees).
Figure 5d reruns all three alert/latent/combined readouts (240 trees) with five
cross-validation seeds. The encoder checkpoint is fixed, so these are not five
independently trained encoders. Toxicity confidence intervals resample five
endpoint-level paired deltas after averaging cross-validation seeds, using
20,000 bootstrap samples. The 3,167-rule alert matrix is unchanged.

Figure 6b and d use pan-cancer correlations across 578 PRISM cell lines; c uses
U2OS activity at LFC ≤ −1. Retention@50 is (retrieved mean correlation minus
random expected correlation)/(oracle mean correlation minus random expected
correlation), computed per query under identical filters and then averaged.
Gene interpretation uses 7,441 gene symbols and mean-encoded, normalized
ORF/CRISPR vectors. Gene-programme labels are rule-based annotations, not a
causal or experimental mechanism assay. Displayed examples are selected from
the 40 leading high-confidence retrieved cases using shared-gene counts.

S3 uses 300 joint epochs for every control, validation selection from epoch 40,
the same data and equal three-term objective. No-source controls retain the
branch initializers. No-pretraining controls initialize all entity encoders
from scratch. The former adapter/bridge-loss experiments are not reused as
evidence for this final architecture.

## Analysis records and upstream reruns

`revision_candidates/downstream_20260920` contains current per-run results,
training configurations/log records, input hashes, embedding provenance, and
the actual research orchestration scripts. Those orchestration snapshots retain
the source workspace paths for traceability. Numerical implementations are in
root `scripts/`; stage the final runtime before loading these checkpoints.
The supplementary September 21 release supplies the mapped relation tables,
ToxRIC tasks/alert matrix, PRISM overlap responses and baseline feature caches.
[Portable commands](REPRODUCTION_INPUTS.md) relocate these inputs explicitly and
recompute the analyses. They start from versioned prepared inputs; a reconstruction
of every source database or foundation-model feature from raw downloads has not
been verified. See the [recomputation audit](REPRODUCTION_VALIDATION.md).

The September 21 label correction identifies Figure 6's 2059-dimensional
baseline as **RDKit-Morgan** (2048 Morgan bits plus 11 RDKit descriptors).
Historical CSV method keys remain `RDKit2D` to preserve compatibility; the R
script maps them to the correct display label. Figure 5 uses a separate, genuine
RDKit2D baseline. This correction changes no source-table values.

The manuscript update preserves paragraph/run/section properties, field
instructions and embedded image dimensions. Main and supplementary documents
remain 28 and 6 pages. Figures and revised pages were rendered for visual QA.
