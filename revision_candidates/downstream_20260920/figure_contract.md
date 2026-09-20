# Figure update contract

Backend: established R workflow only. Numeric analyses use Python; no Python plots.
Dimensions, typography, colours and panel arrangement follow the preceding manuscript.
Exports: editable SVG/PDF, PNG and 600-dpi TIFF; retain source tables and provenance.

- Figure 4, quantitative grid: quantify relation enrichment and held-out-source
  classification using the final model. A documents the relation universe; B reports
  bidirectional enrichment; C/D classify source-held-out pairs and paired gains;
  E projects final embeddings for the same reference entities and pairs. Reference
  pairs are not labelled as the highest-scoring pairs of the new model.
- Figure 5, quantitative grid: measure toxicity prediction and complementarity with
  expert alerts. Preserve endpoints, folds, estimator settings and comparator cohorts.
  Update CGP-containing features; verify matched controls before reuse.
- Figure 6, schematic plus quantitative grid: evaluate structure-diverse response
  neighbours and gene-neighbour overlap. Keep PRISM response matrix and thresholds.
  B uses pan-cancer response correlations; C uses U2OS active-query retrieval.
  D uses the implemented correlation-based oracle-normalized retention definition.
  E uses the fixed historical gene universe restricted to mapped current genes;
  average encoded ORF/CRISPR entity vectors per symbol then normalize. F examples
  must satisfy newly recomputed criteria.
- S2, quantitative grid: describe current cohort and train-only preprocessing.
  Display seed 41, archive all three seeds. Report feature-wise residual R2 rather
  than interpreting it as independent causal batch-effect evidence.
- S3, quantitative grid: compare full model, no profile-source indicator and no
  branch pretraining under the same seed-specific splits and 300 joint epochs.
  Do not retain claims about adapter-count or shared-gene controls not rerun here.

Checks: IDs/row order, candidate sets, split hashes, checkpoint hashes, query-level
pairing, definitions of uncertainty, no reuse of stale CGP caches, visual inspection
at final Word size. Conclusions and selected examples follow new results.
