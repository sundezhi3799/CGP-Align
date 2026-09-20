# Current Figure 4–6 and S2/S3

From the repository root:

```bash
Rscript environment/install-downstream.R
Rscript figures/downstream_final/reproduce.R
python tools/verify_downstream_sources.py
```

`scripts/` reads only local `source_data/` and writes `figures/`. The current
numeric protocols and checkpoint provenance are described in
[DOWNSTREAM_RESULTS.md](../../docs/DOWNSTREAM_RESULTS.md). Original plotting
code uses MIT; adapted archived source-data materials retain the corresponding
[source-data license](../../paper_snapshot/LICENSE). Original-source terms
continue to apply to third-party benchmark annotations.
