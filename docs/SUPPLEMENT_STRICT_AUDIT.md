> Historical audit of the preceding mean-reduction runs. Current three-term results and figure sources are documented in [EQUAL_OBJECTIVE_RESULTS.md](EQUAL_OBJECTIVE_RESULTS.md).

# Supplementary figure consistency audit

Completed on September 20, 2026:

- S1 now uses the strict three-seed training logs and selected epochs 70/150/60.
- S4 sampled/full-gallery metrics and S5 random controls match revised Figure 3.
- The supplement captions define positive candidates, query-level random controls,
  seed-level variability and the actual training objective.
- Main manuscript references to S1 now describe selected-model validation rather
  than a training-scale experiment. The intrinsic retrieval text now correctly
  states all same-entity positives plus 100 negatives.
- The historical scale experiment is preserved, not relabelled as a strict run.
- Original source records and earlier Word files are retained.

## Remaining discrepancy identified from the training logs

The implementation `modality_balanced_contrastive_loss_parts` averages ORF and
CRISPR losses. The joint loop adds this gene term to the compound term. All 900
strict log records confirm:

`L_total = L_compound + (L_ORF + L_CRISPR) / 2`.

On September 20, the author selected the submission implementation's three-term
unit-weight sum as the canonical objective. The training code now implements that
choice; the existing strict results above retain their original mean-reduction
provenance. S1/S4/S5 and revised Figure 3 require fresh results before they can be
presented as evaluations of the selected objective. Preserve the manuscript sum
formula; do not relabel existing weights or logs as equal-weight training.

The complete scientific audit also still requires provenance checks for S2/S3
and main Figures 4-6. This update is not full-manuscript reproduction sign-off.
