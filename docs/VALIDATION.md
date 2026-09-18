# Local validation — 2026-09-18

Environment: Windows, CPython 3.13.5, R 4.5.3. Observed Python package versions are in `environment/requirements-tested.txt`; they describe this check environment, not the historical GPU training environment. A fresh Linux installation and GPU retraining have not been verified.

Passed checks:

- `python tools/verify_reference_results.py`: all 12 run/direction Top-10 values agree with the original saved run metrics within absolute tolerance 1e-12. Means are 0.8433696090, 0.5376190501, 0.6468732396 and 0.3461344798 for compound-to-profile, profile-to-compound, gene-to-profile and profile-to-gene.
- `python tools/smoke_test_model.py`: original graph, ORF, CRISPR and shared-profile modules; finite loss, unit-norm embeddings, finite nonzero gradients in each branch and strict state-dictionary round-trip. Uses reduced hidden sizes and synthetic inputs, not pretrained weights.
- `Rscript paper_snapshot/scripts/reproduce_article_results.R`: exit code 0; 68 source files scanned, 164 numerical result rows and 4 statistical summary rows generated. Both output CSVs exactly match the previously archived CSVs after parsing. R emitted locale startup warnings but completed successfully.
- Python AST compilation of included source and helper scripts: no syntax errors.

These checks verify implementation and saved-result consistency. They do not establish that full training or downstream analyses can yet be reproduced using only public artifacts. See `RELEASE_AUDIT.md`.
