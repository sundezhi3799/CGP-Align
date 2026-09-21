# Figure 6 inputs

The CSV files contain the plotted source values for panels b–e. `workflow.png` is the schematic in panel a. Molecular coordinates and bonds are provided in `chemical_vectors/`; the plot draws editable structures from these coordinates.

`shared_gene_annotations.csv` records all 39 genes shared by BIIB021 and R547, their ranks, functional assignments and UniProt accessions. Each gene has one manually assigned display category. The bars count genes; they are not enrichment scores or category prevalence across compound pairs.

Panels f–h are calculated from PRISM responses and model similarities by `scripts/evaluate_prism_response_similarity.py`, also called by the PRISM analysis command. The primary cohort is 187 U2OS queries with LFC ≤ −1. The response-distance comparison uses top-50 neighbours at Morgan Tanimoto <0.20. The matched control averages absolute response differences within query-specific Tanimoto bins of width 0.01, including retrieved compounds. Additional bin widths and the all-query analysis are exported alongside the main results.

The 0.488 response correlation in panel e is across 578 cell lines. Panels f–h use only U2OS (ACH-000364). The plots show descriptive query-level results without pair-independent significance tests.
