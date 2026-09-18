# Portability update 2026-09-18

## Changes and scope

1. Revised Figure 3 now has a standalone R redraw workflow and all of its processed
   source tables. The August snapshot is preserved. See `figures/figure3/README.md`.
2. Intrinsic checkpoint evaluation and hidden-neighbour export accept three explicit
   data-directory overrides. They clear all four branch-initializer paths before
   constructing the model, then load the full checkpoint with `strict=True`.
   Training initialization and architecture are unchanged.
3. PRISM functional retrieval and the NYAN cache conversion script are included
   with their local import dependencies. The former is historical tri-branch code;
   its inclusion does not establish equivalence to the final four-branch model.
4. PRISM gene interpretation now requires explicit input/output paths and supports
   `--help` without executing analysis. `tabulate`, used by its report writer, is
   declared in the analysis environment. Embedding provenance still needs checking;
   matching dimensions alone are not evidence of a shared model/coordinate system.
5. CI now exercises the original four-branch synthetic model, portable full-checkpoint
   loading, strict rejection of incomplete weights, and revised-H saved-source arithmetic.
   These checks do not establish manuscript-level retraining performance.

Original hashes remain in `source_provenance.csv`; edited import hashes are recorded
separately in `portability_changes.json`. Only the listed portability changes were
made to imported research code.

## Commands

```bash
python -m pip install -r environment/requirements-core.txt
python tools/smoke_test_model.py
python tools/test_portable_inference.py
python scripts/eval_raw3180_branch_retrieval_from_checkpoints.py --rows seed31:checkpoints/seed31.pt --compound-data-dir data/seed31/compound --gene-data-dir data/seed31/gene --gene-protein-embedding-dir data/protein --output_dir outputs/retrieval/seed31 --max_compounds 0 --max_genes 0
```

Use matching seed-specific splits and the exact published checkpoint. The example
evaluates the full held-out entity set; sampling, repeat counts and seeds must be
matched to a specific reference experiment before comparing numerical results.
Only load trusted checkpoint files (the historical format uses Python pickle).

```bash
python -m pip install -r environment/requirements-analysis.txt
python scripts/run_prism_functional_analogue_retrieval.py --help
python scripts/build_prism_official_nyan_cache.py --help
python scripts/run_prism_gene_mechanism_interpretation.py --prism-dir data/prism_analysis --prism-embeddings data/prism_compounds.npz --gene-embeddings data/genes.npz --output-dir outputs/prism_mechanism
```

The last command requires the existing overlap/pair CSV files plus compatible
embeddings; these inputs are not bundled. No PRISM scientific result was rerun in
this update. NYAN requires its official encoder output; the cache builder is not
a replacement implementation of that encoder.

## Remaining blockers

The three exact primary checkpoint paths and their prepared data directories from
`manifests/models.json` were checked on the maintainer's current workspace and are
absent. They must be recovered from the training host/original archive. Do not replace
them with similarly named local runs. The two missing compound preprocessing scripts,
branch initializers, exact protein features and ordering, raw data versions, downstream
model provenance and additional figure workflows remain unresolved. A public weight
URL cannot be supplied until the correct artifact has actually been recovered and
published. This update is still a development version, not a complete reproduction release.
