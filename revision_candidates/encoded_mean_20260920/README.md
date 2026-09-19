# Adopted encode-then-average evaluation, 2026-09-20

All 12 frozen-model evaluations completed. Test cohort: 11,578 compounds per
seed (31, 37, 41), one matched phenotype per compound. Each phenotype is the
L2-normalized mean of individually encoded, strictly corrected replicates.
Both directions use identical frozen negative banks (100/1,000 negatives,
10 repeats) and the full test gallery. Higher Recall@10 means the paired
compound/phenotype ranks among the ten most similar candidates.

`summary.csv` and `summary.json` report bidirectional seed means and sample SD
across three seeds; `seed_records.csv` preserves individual runs. Metrics are
fractions, not percentages. All 12 new 100-negative scores reproduce the prior
diagnostic exactly. Saved per-query ranks independently reproduce every new
Recall@10 metric. Checkpoint, prepared-audit and candidate hashes match the
historical evaluation. `historical_encode_mean/` retains all 12 old metric files.

This aggregation choice followed test-set sensitivity analysis. Both protocols
are retained; these results do not reproduce the legacy 614-feature 71.1% result.
Training remains different across frameworks (replicate-level multimodal CGP
versus entity-mean comparator training); see the protocol documentation.

Full embeddings and per-query ranks remain on gpu80 under
`/data3/sdz/cgp_align_matched_encoded_mean_20260920/seed{seed}/{method}/`.
They are not included in this Git repository. The code and metric records alone
do not replace the separate input-data and checkpoint release requirements.
The archived manuscript and Figure 3 plotting inputs have not been overwritten.
