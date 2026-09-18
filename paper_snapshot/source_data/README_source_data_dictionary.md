# Source Data Dictionary

This directory contains the panel-level source data used to generate the main and supplementary figures for the CGP-Align study. Files are organized by figure and panel. Unless otherwise stated, downstream labels used in these files were not used for CGP-Align encoder training or checkpoint selection.

## General conventions

- `method` or `method_label`: representation or model family shown in the figure.
- `independent_run`: repeated independent run identifier where applicable.
- `direction`: retrieval direction, for example compound-to-profile or profile-to-compound.
- `Top10`, `Recall@10`, `MRR`: retrieval metrics calculated on held-out entities or external labels.
- `AUPRC`: area under the precision-recall curve.
- `AUROC`: area under the receiver operating characteristic curve.
- `Lift@50`: enrichment of positive labels among the top 50 ranked candidates relative to the mapped candidate-universe expectation.
- Error bars are figure-specific and are defined in the corresponding figure legends.

## Main figures

### Figure 1

- `figure1a_resource_composition.csv`: entity and replicate-level profile counts supporting the profile-level Cell Painting resource described in the text.
- `figure1b_preprocessing_label_firewall.csv`: preprocessing, entity-held-out split and downstream-label isolation elements used to define the study design.
- `figure1c_architecture_nodes.csv`: schematic node definitions for the CGP-Align encoder architecture.
- `figure1c_contrastive_matrix_schematic.csv`: plotting coordinates for the mini-batch contrastive-alignment matrix schematic.
- `figure1d_evidence_ladder.csv`: staged frozen-embedding evaluation tasks shown conceptually in the Fig. 1c evaluation panel.

Analysis unit: dataset component, model component or evaluation stage. The final Figure 1 is a conceptual framework schematic; quantitative claims supported by these files are the resource-count summaries reported in the text.

### Figure 2

- `figure2_architecture_source.csv`: panel-level architecture descriptors for the shared profile encoder, compound/ORF/CRISPR entity encoders and symmetric contrastive objectives shown in the Fig. 2 schematic.

Analysis unit: architecture component. Figure 2 is a schematic model-architecture figure documenting the shared profile encoder, branch-specific entity encoders and symmetric contrastive-alignment objective.

### Figure 3

- `figure3a_b_pca_source_data.csv`: PCA coordinates used for the before/after joint-alignment visual diagnostics in Fig. 3a,b.
- `figure3a_b_pca_diagnostics.csv`: PCA variance and sampling diagnostics for Fig. 3a,b.
- `figure3a_b_pca_metadata.json`: provenance metadata for the PCA diagnostic source data.
- `figure3c_sampled_retrieval_run_values.csv`: direction-level sampled 1:100 Top-10 retrieval values by independent run.
- `figure3c_random_ranking_top10.csv`: positive-count-adjusted random-ranking expectations for sampled retrieval.
- `figure3e_branch_summary_mean_sd.csv`: branch-level Mean/HMean Top-1 and Top-10 summaries, standard deviations across independent runs and matched random-ranking expectations.
- `figure3f_matched_retrieval_performance.csv`: bidirectional sampled 1:100 compound-profile retrieval under a unified 3,180-dimensional CellProfiler profile protocol.
- `figure3g_difficult_retrieval.csv`: sampled 1:1000 and full-gallery compound-profile retrieval results.
- `figure3h_paired_run_direction.csv`: paired CGP-Align minus MoLFormer differences across run-direction units.

Analysis unit: held-out retrieval direction, independent run or matched compound-profile benchmark unit. Random-ranking values account for the average number of same-entity positive profiles in the sampled retrieval gallery. Matched compound-profile panels use same-task baselines trained against the same 3,180-dimensional CellProfiler profile target and evaluated on the same held-out split.

### Figure 4

- `figure4a_relation_source_upset_intersections_main.csv`: intersection sizes for mapped external compound-gene relation sources used in the UpSet-style source-overlap panel.
- `figure4a_relation_source_upset_matrix_main.csv`: source-membership matrix for relation-source intersections in Fig. 4a.
- `figure4a_relation_source_upset_metadata_main.csv`: source-universe metadata for Fig. 4a.
- `figure4a_relation_source_upset_set_sizes_main.csv`: mapped edge counts for each external relation source in Fig. 4a.
- `figure4b_target_transfer_lift50.csv`: filtered five-source audit input for relation-transfer Lift@50.
- `figure4b_target_transfer_lift50_main.csv`: cosine-ranking Lift@50 for frozen CGP-Align embeddings across the five target-oriented external compound-gene relation benchmarks.
- `figure4c_target_transfer_leave_one_source_out_classifier_main.csv`: leave-one-source-out frozen-feature classifier summaries.
- `figure4_relation_transfer_summary_main.csv`: text-level summary values for Fig. 4b,c.
- `figure4d_target_transfer_metric_gain_main.csv`: absolute AUPRC and AUROC gains of the frozen-feature classifier over cosine ranking shown in Fig. 4d.
- `figure4e_merged_database_entity_space_pca_coordinates.csv`: PCA coordinates for mapped compound and gene entities in the frozen CGP-Align latent space.
- `figure4e_merged_database_entity_space_pca_links.csv`: representative known compound-gene pairs highlighted in the Fig. 4e latent-space projection.

Analysis unit: external relation benchmark, mapped compound-gene edge, or leave-one-source-out relation evaluation. Relation labels are introduced only after encoder training. Mapped compound-gene edges are downstream positives within each source-specific mapped candidate universe. Supervised results use frozen CGP-Align embeddings and test downstream leave-one-source-out calibration, not relation-label pretraining.

### Figure 5

- `figure5a_sr5_representation_mean_metrics.csv`: mean AUPRC and AUROC across five phenotype-related ToxRIC endpoints.
- `figure5b_sr5_endpoint_auprc_heatmap.csv`: endpoint-level AUPRC values for the same SR5 panel.
- `figure5c_sr5_scaffold_split_endpoint_dots.csv`: scaffold-disjoint endpoint-level AUPRC values.
- `figure5c_sr5_scaffold_split_mean_metrics.csv`: scaffold-disjoint mean AUPRC summary.
- `figure5d_expert_alert_complementarity_summary.csv`: alert-only, CGP-Align-only and combined summary metrics.
- `figure5d_sr5_expert_alert_complementarity_by_run.csv`: repeated-run values after endpoint-level averaging.

Analysis unit: five phenotype-related ToxRIC endpoints: SR-ARE, SR-MMP, SR-HSE, SR-p53 and SR-ATAD5. Conclusions are bounded to these endpoints and do not imply uniform performance gains across all TOXRIC labels.

### Figure 6

- `figure6_panel_b_functional_neighbour_enrichment_source.csv`: high-confidence PRISM functional-neighbour retrieval summaries across Morgan Tanimoto filters, response-correlation thresholds and top-k cutoffs.
- `figure6_panel_c_cgp_vs_rdkit2d_source.csv`: query-level paired comparison between CGP-Align and RDKit2D under the low-structure U2OS active-neighbour retrieval setting.
- `figure6_panel_d_pancancer_source.csv`: pan-cancer oracle-normalized retention@50 across structural-similarity filters.
- `figure6_panel_e_gene_mechanism_summary_source.csv`: gene-neighbour mechanism summaries used to select and document shared-gene evidence for structure-diverse retrieved compound pairs.
- `figure6_panel_e_mechanism_category_summary_source.csv`: mechanism-category summaries used for the CGP-Align program bubble plot.
- `figure6_panel_f_high_confidence_cases_source.csv`: high-confidence CGP-Align functional-neighbour case pairs with PRISM response correlation and Morgan Tanimoto similarity.
- `figure6_panel_f_mechanism_cases_source.csv`: gene-neighbour mechanism annotations for representative CGP-Align case pairs.
- `figure6_panel_f_discovery_landscape_source.csv`: discovery-landscape source data for Panel f, including high-confidence CGP-Align response-neighbour pairs, PRISM response correlation, Morgan similarity and matched gene-neighbour mechanism annotations where available.
- `figure6_manifest.json`: source and analysis metadata for Figure 6.

Analysis unit: PRISM query compound, structural-similarity filter, representation method, gene-neighbour summary or retrieved compound pair. Functional-neighbour enrichment is calculated from high-confidence PRISM response-neighbour hits relative to random ranking under the same Morgan Tanimoto filter. In the query-level panel, identical coordinate pairs are aggregated for plotting; `n_query` stores the number of U2OS active queries represented by each point. Oracle-normalized retention@50 is normalized to the best available response-similarity ranking under the same structural filter. Gene-neighbour mechanism summaries are based on shared top-50 CGP-Align gene neighbours and predefined mechanism-category gene sets. Panels e and f visualize computational mechanism-prioritization evidence for selected structure-diverse response neighbours; they do not establish experimental confirmation of shared mechanism.

## Supplementary figures

- `figureS1_training_log_public.jsonl` and `figureS1_training_log_public_README.md`: public training-log subset and field definitions.
- `figureS1a_training_loss_curve.csv`: training objective and branch-level loss curves.
- `figureS1b_branch_validation_curves.csv` and `figureS1b_validation_selection_curve.csv`: validation retrieval trajectories and checkpoint-selection curve.
- `figureS1c_checkpoint_selection.csv` and `figureS1_checkpoint_selection_summary.csv`: candidate checkpoint metrics and selected checkpoint summary.
- `figureS1d_training_scale_ablation.csv`, `figureS1_training_scale_family_summary.csv` and `figureS1_training_scale_train_summary.csv`: training-scale ablation summaries.
- `figureS2a_dataset_composition.csv`: processed entity and replicate-level profile counts.
- `figureS2b_entity_replicate_distribution.csv` and `figureS2b_entity_replicate_summary.csv`: replicate support per biological entity.
- `figureS2c_split_composition.csv` and `figureS2c_split_overlap_check.csv`: entity-held-out split composition and overlap checks.
- `figureS2d_post_correction_r2.csv` and `figureS2e_normalization_audit.csv`: normalization and quality-control audits.
- `figureS3a_main_ablation_table.csv`: main architecture and loss-design ablation summary.
- `figureS3b_key_ablation_run_level_metrics.csv`: run-level metrics for selected model variants.
- `figureS3c_historical_ablation_context.csv`: sanitized historical architecture-control context.
- `figureS4_fig2_matched_sampled_retrieval_metrics.csv`: Figure 2-matched sampled retrieval audit.
- `figureS4_fig2_matched_full_gallery_metrics.csv`: Figure 2-matched full-gallery retrieval metrics.
- `figureS4_fig2_matched_branch_summary.csv`: branch-level summaries for the Figure 2-matched audit.
- `figureS4_fig2_matched_checkpoint_sources.csv`: checkpoint-source metadata for Supplementary Figure 4.
- `figureS5_random_ranking_negative_control.csv`: random-ranking controls for sampled and full-gallery retrieval.
- `statistical_tests.csv`: paired bootstrap and sign-test summaries reported in the associated article text.

Supplementary figure files retain the natural analysis units described in their legends. Exploratory files not used by the current article figures are excluded from this source-data package.
