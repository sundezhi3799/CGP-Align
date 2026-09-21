# Reproduction input sources and attribution

The repository MIT license covers original CGP-Align code and trained CGP-Align
and comparator projection/profile models. It does not relicense third-party
datasets, foundation models or curated rule libraries. The supplementary release
preserves the exact scientific inputs used here; each file has a SHA256 in
`manifests/reproduction_inputs_20260921.json`.

- **Cell Painting and molecular identity metadata:** JUMP/cpg0016 in the
  [Cell Painting Gallery](https://github.com/broadinstitute/cellpainting-gallery).
  Gallery data are CC0. The previous main release contains seed-specific prepared
  profile arrays and training-fitted corrections; microscopy images are not included.
- **Compound–gene annotations:** mapped broad/strict inputs and source subsets
  derived from [MotiVE](https://github.com/carpenter-singh-lab/2024_Arevalo_NeurIPS_MotiVE),
  using the Gallery `cpg0034-arevalo-su-motive` public data. Cite Arevalo, Su et al.,
  NeurIPS 2024 (arXiv:2406.08649) and the underlying DGIdb, DrugRep, Hetionet,
  OpenBioLink and PharMeBINet sources. The upstream
  [annotation archive](https://zenodo.org/records/18197517) explains its mixed-source
  terms and warns that its newer snapshots can differ from the original S3 data.
  Do not silently substitute a newer annotation file. No blanket MIT or CC0 claim
  is made for these source-database contents.
- **Toxicity labels:** the five `Endocrine Disruption_SR-*` task CSVs are exact
  files obtained from [TOXRIC](https://toxric.bioinforai.tech/). Cite Wu et al.,
  *Nucleic Acids Research* 51(D1), D1432–D1445,
  [doi:10.1093/nar/gkac1074](https://doi.org/10.1093/nar/gkac1074).
  Its authors describe freely accessible, reusable CSV downloads. A specific
  blanket open-data license was not established during this audit; retain TOXRIC
  and underlying assay-source terms, rather than treating these files as MIT data.
- **Toxicity structural features:** archived RDKit/Avalon/Morgan/AtomPair and NYAN
  feature arrays, with their exact molecule order. These are numerical inputs,
  not third-party model weights. The alert activation matrix preserves 3,167
  columns; `expert_alert_metadata.csv` identifies source libraries, rule IDs and
  references. RDKit, ChEMBL, Toxtree and literature rule sources retain their
  respective terms; the CGP-Align code license does not override them.
- **PRISM:** the 1,856-compound, 578-cell-line overlap matrix derives from the
  [Broad PRISM Repurposing](https://depmap.org/repurposing/) 19Q4 primary screen,
  `primary-screen-replicate-collapsed-logfold-change.csv` and treatment metadata.
  Cite Corsello et al., *Nature Cancer* (2020), and the Broad DepMap resource.
  DepMap-generated data use CC BY 4.0. The compound overlap metadata and exact
  archived NYAN/Morgan-RDKit input features accompany the matrix.
- **MoLFormer/ChemBERTa features:** numerical molecular embeddings from the
  archived comparator pipeline. The released checkpoints are the projection and
  profile networks trained for this study, not the pretrained language models.
  Feature hashes, molecule order and training recipes are retained.

Access is anonymous through GitHub Releases. Source citations and license
boundaries are separate from the software license. This record documents
provenance and observed terms; it does not assert that heterogeneous source
databases share a single open-data license.
