# Figure 5

The six `figure5*.csv` tables provide the values plotted in panels a–d.

For panel c, `scaffold_all_folds.csv` contains 150 evaluations: six representations, five endpoints and five folds. `scaffold_fold_assignments.csv.gz` records the canonical SMILES, binary label, scaffold group and test fold of each molecule within each endpoint. All representations use these same folds. Acyclic molecules are grouped by canonical SMILES.

The panel c dots are mean average precision across five folds for each endpoint; bars are equal-weight means across the five endpoints. `sd_endpoint_auprc` describes variation across endpoints. `CGP_sep_k32` is the CGP-Align representation.

Run `Rscript scripts/plot_figure5.R <output directory>` from the repository root to export PDF, SVG and 600-dpi PNG/TIFF.

Toxicity labels originate from TOXRIC and the underlying assays; their data terms apply. See the repository acknowledgements.
