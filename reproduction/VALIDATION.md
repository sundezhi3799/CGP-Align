# Reproduction checks

Checked on Linux on 12 November 2026 using the pinned downstream dependencies
and R 4.5.3:

- All 66 figure data files match the manuscript source-data archive byte for byte.
- Figure 3–6 and the included supplementary plots export successfully.
- All 300 outer and 600 inner toxicity partitions pass the scaffold and label
  checks; regenerated inner assignments equal the supplied assignments.
- A 32-dimensional RDKit / broad / seed-31 / DGIdb relation run reproduces
  AUPRC, AUROC and query-average AP within 1e-12, using the exact published split.
- The five focused toxicity endpoints reproduce the 15 added/control method
  summaries. PRISM reproduces all 20 method/cutoff summaries and 9,280 query-level
  U2OS records within numerical precision.
- Five existing public-input tests pass.
- Every downloadable archive matches its manifest size and SHA-256 checksum.

These checks validate the portable entry points and their supplied inputs.
The full 24,300-fit toxicity benchmark and all relation models were not refitted
as part of this packaging check; their published source tables are supplied.
