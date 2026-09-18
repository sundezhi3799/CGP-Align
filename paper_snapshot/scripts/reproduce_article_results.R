# Reproduce article-level numerical results from packaged source data.
#
# This script uses only files distributed inside this package. It does not
# require raw third-party datasets, model checkpoints or training code.

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) > 0) {
  script_file <- normalizePath(sub("^--file=", "", file_arg[1]), mustWork = TRUE)
  package_dir <- normalizePath(file.path(dirname(script_file), ".."), mustWork = TRUE)
} else {
  package_dir <- normalizePath(getwd(), mustWork = TRUE)
  if (basename(package_dir) == "scripts") {
    package_dir <- normalizePath(file.path(package_dir, ".."), mustWork = TRUE)
  }
}

source_dir <- file.path(package_dir, "source_data")
table_dir <- file.path(package_dir, "supplementary_tables")
result_dir <- file.path(package_dir, "results")
dir.create(result_dir, showWarnings = FALSE, recursive = TRUE)

read_source <- function(filename) {
  path <- file.path(source_dir, filename)
  if (!file.exists(path)) {
    stop("Missing source-data file: ", filename, call. = FALSE)
  }
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

num <- function(x) suppressWarnings(as.numeric(x))

metric_rows <- list()
add_metric <- function(figure, panel, analysis, metric, value, unit, source_file) {
  metric_rows[[length(metric_rows) + 1L]] <<- data.frame(
    figure = figure,
    panel = panel,
    analysis = analysis,
    metric = metric,
    value = value,
    unit = unit,
    source_file = source_file,
    stringsAsFactors = FALSE
  )
}

stat_rows <- list()
add_stat <- function(figure, comparison, metric, unit, n, mean_delta,
                     ci95_low, ci95_high, sign_pos, sign_neg, sign_ties, p_value,
                     source_file) {
  stat_rows[[length(stat_rows) + 1L]] <<- data.frame(
    figure = figure,
    comparison = comparison,
    metric = metric,
    unit = unit,
    n = n,
    mean_delta = mean_delta,
    ci95_low = ci95_low,
    ci95_high = ci95_high,
    sign_pos = sign_pos,
    sign_neg = sign_neg,
    sign_ties = sign_ties,
    p_value = p_value,
    source_file = source_file,
    stringsAsFactors = FALSE
  )
}

bootstrap_mean_ci <- function(delta, n_boot = 20000L, seed = 1L) {
  delta <- delta[is.finite(delta)]
  set.seed(seed)
  boot <- replicate(n_boot, mean(sample(delta, length(delta), replace = TRUE)))
  c(mean = mean(delta), low = unname(quantile(boot, 0.025)),
    high = unname(quantile(boot, 0.975)))
}

sign_test_p <- function(pos, neg) {
  total <- pos + neg
  if (total == 0L) return(NA_real_)
  binom.test(pos, total, p = 0.5, alternative = "two.sided")$p.value
}

# Figure 1: processed resource composition.
fig1 <- read_source("figure1a_resource_composition.csv")
for (i in seq_len(nrow(fig1))) {
  add_metric("Figure 1", "a", fig1$modality[i], fig1$quantity[i],
             num(fig1$n[i]), "count", "figure1a_resource_composition.csv")
}

# Figure 3: intrinsic retrieval and matched compound-profile benchmark.
fig3c <- read_source("figure3c_sampled_retrieval_run_values.csv")
for (direction in unique(fig3c$direction)) {
  vals <- num(fig3c$top10[fig3c$direction == direction])
  add_metric("Figure 3", "c", direction, "Top-10 accuracy mean",
             mean(vals), "fraction", "figure3c_sampled_retrieval_run_values.csv")
  add_metric("Figure 3", "c", direction, "Top-10 accuracy sd",
             stats::sd(vals), "fraction", "figure3c_sampled_retrieval_run_values.csv")
}
fig3c_null <- read_source("figure3c_random_ranking_top10.csv")
for (i in seq_len(nrow(fig3c_null))) {
  add_metric("Figure 3", "c", fig3c_null$direction[i], "Random-ranking Top-10",
             num(fig3c_null$random_ranking_top10[i]), "fraction",
             "figure3c_random_ranking_top10.csv")
}
fig3e <- read_source("figure3e_branch_summary_mean_sd.csv")
for (i in seq_len(nrow(fig3e))) {
  add_metric("Figure 3", "e", fig3e$metric[i], "Observed mean",
             num(fig3e$mean[i]), "fraction", "figure3e_branch_summary_mean_sd.csv")
  add_metric("Figure 3", "e", fig3e$metric[i], "Random-ranking mean",
             num(fig3e$random_ranking_mean[i]), "fraction", "figure3e_branch_summary_mean_sd.csv")
}
fig3f <- read_source("figure3f_matched_retrieval_performance.csv")
for (i in seq_len(nrow(fig3f))) {
  add_metric("Figure 3", "f", fig3f$method_label[i],
             "Bidirectional 1:100 Top-10 accuracy",
             num(fig3f$mean_top10[i]), "fraction",
             "figure3f_matched_retrieval_performance.csv")
}
fig3g <- read_source("figure3g_difficult_retrieval.csv")
for (i in seq_len(nrow(fig3g))) {
  add_metric("Figure 3", "g", fig3g$method_label[i], fig3g$metric[i],
             num(fig3g$value[i]), "fraction",
             "figure3g_difficult_retrieval.csv")
}
fig3h <- read_source("figure3h_paired_run_direction.csv")
delta3 <- num(fig3h$delta)
ci3 <- bootstrap_mean_ci(delta3)
add_stat("Figure 3", "CGP-Align minus MoLFormer", "Retrieval delta",
         "run-direction", length(delta3), ci3["mean"], ci3["low"], ci3["high"],
         sum(delta3 > 0), sum(delta3 < 0), sum(delta3 == 0),
         sign_test_p(sum(delta3 > 0), sum(delta3 < 0)),
         "figure3h_paired_run_direction.csv")

# Figure 4: external compound-gene relation transfer.
fig4meta <- read_source("figure4a_relation_source_upset_metadata_main.csv")
if ("n_edges_in_selected_sources" %in% names(fig4meta)) {
  add_metric("Figure 4", "a", "Mapped external relation sources",
             "Mapped compound-gene edges", num(fig4meta$n_edges_in_selected_sources[1]),
             "count", "figure4a_relation_source_upset_metadata_main.csv")
}
fig4sets <- read_source("figure4a_relation_source_upset_set_sizes_main.csv")
for (i in seq_len(nrow(fig4sets))) {
  add_metric("Figure 4", "a", fig4sets$name[i], "Mapped source edges",
             num(fig4sets$set_edges[i]), "count",
             "figure4a_relation_source_upset_set_sizes_main.csv")
}
fig4b <- read_source("figure4b_target_transfer_lift50_main.csv")
for (i in seq_len(nrow(fig4b))) {
  add_metric("Figure 4", "b", fig4b$name[i], "Mean Lift@50",
             num(fig4b$mean_lift50[i]), "fold",
             "figure4b_target_transfer_lift50_main.csv")
}
fig4summary <- read_source("figure4_relation_transfer_summary_main.csv")
for (i in seq_len(nrow(fig4summary))) {
  add_metric("Figure 4", "summary", fig4summary$metric[i], "Summary value",
             num(fig4summary$value[i]), "reported unit",
             "figure4_relation_transfer_summary_main.csv")
}
fig4c <- read_source("figure4c_target_transfer_leave_one_source_out_classifier_main.csv")
for (i in seq_len(nrow(fig4c))) {
  add_metric("Figure 4", "c", fig4c$name[i], "Frozen-feature classifier AUPRC",
             num(fig4c$leave_one_source_out_auprc[i]), "fraction",
             "figure4c_target_transfer_leave_one_source_out_classifier_main.csv")
  add_metric("Figure 4", "c", fig4c$name[i], "Cosine AUPRC",
             num(fig4c$cosine_auprc[i]), "fraction",
             "figure4c_target_transfer_leave_one_source_out_classifier_main.csv")
}
fig4d <- read_source("figure4d_target_transfer_metric_gain_main.csv")
for (i in seq_len(nrow(fig4d))) {
  add_metric("Figure 4", "d", fig4d$name[i], fig4d$metric[i],
             num(fig4d$gain[i]), "absolute gain",
             "figure4d_target_transfer_metric_gain_main.csv")
}

# Figure 5: toxicity benchmark and structural-alert complementarity.
fig5a <- read_source("figure5a_sr5_representation_mean_metrics.csv")
for (i in seq_len(nrow(fig5a))) {
  add_metric("Figure 5", "a", fig5a$display[i], "Mean AUPRC",
             num(fig5a$auprc_mean[i]), "fraction",
             "figure5a_sr5_representation_mean_metrics.csv")
  add_metric("Figure 5", "a", fig5a$display[i], "Mean AUROC",
             num(fig5a$auroc_mean[i]), "fraction",
             "figure5a_sr5_representation_mean_metrics.csv")
}
fig5c <- read_source("figure5c_sr5_scaffold_split_mean_metrics.csv")
for (i in seq_len(nrow(fig5c))) {
  add_metric("Figure 5", "c", fig5c$display[i], "Scaffold-disjoint mean AUPRC",
             num(fig5c$mean_auprc[i]), "fraction",
             "figure5c_sr5_scaffold_split_mean_metrics.csv")
}
fig5d <- read_source("figure5d_expert_alert_complementarity_summary.csv")
for (i in seq_len(nrow(fig5d))) {
  add_metric("Figure 5", "d", fig5d$method_label[i], "AUPRC",
             num(fig5d$mean_auprc[i]), "fraction",
             "figure5d_expert_alert_complementarity_summary.csv")
  add_metric("Figure 5", "d", fig5d$method_label[i], "Lift@50",
             num(fig5d$mean_lift_at_top50[i]), "fold",
             "figure5d_expert_alert_complementarity_summary.csv")
}
fig5d_run <- read_source("figure5d_sr5_expert_alert_complementarity_by_run.csv")
run_ids <- unique(fig5d_run$independent_run)
get_by_run <- function(label, field) {
  out <- numeric(length(run_ids))
  for (i in seq_along(run_ids)) {
    row <- fig5d_run[fig5d_run$independent_run == run_ids[i] &
                       fig5d_run$method_label == label, , drop = FALSE]
    out[i] <- num(row[[field]][1])
  }
  out
}
delta5_auprc <- get_by_run("CGP + alerts", "run_mean_auprc") -
  get_by_run("CGP-Align latent", "run_mean_auprc")
delta5_lift <- get_by_run("CGP + alerts", "run_mean_lift_at_top50") -
  get_by_run("CGP-Align latent", "run_mean_lift_at_top50")
ci5a <- bootstrap_mean_ci(delta5_auprc)
ci5l <- bootstrap_mean_ci(delta5_lift)
add_stat("Figure 5", "CGP + alerts minus CGP-Align latent", "AUPRC",
         "independent run after endpoint-level averaging", length(delta5_auprc),
         ci5a["mean"], ci5a["low"], ci5a["high"],
         sum(delta5_auprc > 0), sum(delta5_auprc < 0), sum(delta5_auprc == 0),
         sign_test_p(sum(delta5_auprc > 0), sum(delta5_auprc < 0)),
         "figure5d_sr5_expert_alert_complementarity_by_run.csv")
add_stat("Figure 5", "CGP + alerts minus CGP-Align latent", "Lift@50",
         "independent run after endpoint-level averaging", length(delta5_lift),
         ci5l["mean"], ci5l["low"], ci5l["high"],
         sum(delta5_lift > 0), sum(delta5_lift < 0), sum(delta5_lift == 0),
         sign_test_p(sum(delta5_lift > 0), sum(delta5_lift < 0)),
         "figure5d_sr5_expert_alert_complementarity_by_run.csv")

# Figure 6: PRISM response-profile retrieval.
fig6b <- read_source("figure6_panel_b_functional_neighbour_enrichment_source.csv")
main6b <- fig6b[fig6b$topk == 50 & fig6b$corr_threshold == 0.25 &
                  abs(num(fig6b$low_tanimoto) - 0.20) < 1e-8, , drop = FALSE]
for (i in seq_len(nrow(main6b))) {
  add_metric("Figure 6", "b", main6b$method[i],
             "Active hits per 1,000 top-50 ranked neighbours",
             num(main6b$hit_yield_per_1000_ranked[i]), "hits per 1,000",
             "figure6_panel_b_functional_neighbour_enrichment_source.csv")
  add_metric("Figure 6", "b", main6b$method[i], "Unique pair hits",
             num(main6b$unique_pair_hits[i]), "count",
             "figure6_panel_b_functional_neighbour_enrichment_source.csv")
}
fig6c <- read_source("figure6_panel_c_cgp_vs_rdkit2d_source.csv")
weights6 <- if ("n_query" %in% names(fig6c)) as.integer(fig6c$n_query) else rep(1L, nrow(fig6c))
delta6 <- rep(num(fig6c$`CGP-Align`) - num(fig6c$RDKit2D), weights6)
ci6 <- bootstrap_mean_ci(delta6)
add_stat("Figure 6", "CGP-Align minus RDKit2D", "Top-50 active fraction",
         "matched active U2OS query", length(delta6), ci6["mean"], ci6["low"], ci6["high"],
         sum(delta6 > 0), sum(delta6 < 0), sum(delta6 == 0),
         sign_test_p(sum(delta6 > 0), sum(delta6 < 0)),
         "figure6_panel_c_cgp_vs_rdkit2d_source.csv")
fig6d <- read_source("figure6_panel_d_pancancer_source.csv")
for (i in seq_len(nrow(fig6d))) {
  add_metric("Figure 6", "d", paste(fig6d$method[i], fig6d$setting[i]),
             "Oracle-normalized retention@50", num(fig6d$retention50[i]),
             "fraction", "figure6_panel_d_pancancer_source.csv")
}
fig6e <- read_source("figure6_panel_e_gene_mechanism_summary_source.csv")
for (i in seq_len(nrow(fig6e))) {
  add_metric("Figure 6", "e", fig6e$method[i], "Mean shared top-50 genes",
             num(fig6e$mean_shared_top50_genes[i]), "count",
             "figure6_panel_e_gene_mechanism_summary_source.csv")
  add_metric("Figure 6", "e", fig6e$method[i], "Mean shared-gene -log10(P)",
             num(fig6e$mean_shared_gene_neglog10p[i]), "-log10(P)",
             "figure6_panel_e_gene_mechanism_summary_source.csv")
}

# Supplementary result hooks.
figS1c <- read_source("figureS1c_checkpoint_selection.csv")
selected <- figS1c[figS1c$is_selected %in% c(TRUE, "TRUE", "True", "true", 1, "1"), , drop = FALSE]
if (nrow(selected) > 0) {
  add_metric("Supplementary Figure 1", "c", "Selected checkpoint",
             "Epoch", num(selected$epoch[1]), "epoch",
             "figureS1c_checkpoint_selection.csv")
  add_metric("Supplementary Figure 1", "c", "Selected checkpoint",
             "Validation Branch HMean Top-10", num(selected$val_hmean_Top10[1]),
             "fraction", "figureS1c_checkpoint_selection.csv")
}
figS2c <- read_source("figureS2c_split_overlap_check.csv")
add_metric("Supplementary Figure 2", "c", "Split overlap check",
           "Maximum overlapping entities", max(num(figS2c$overlap_entities), na.rm = TRUE),
           "count", "figureS2c_split_overlap_check.csv")
figS3a <- read_source("figureS3a_main_ablation_table.csv")
if ("Balanced val score" %in% names(figS3a)) {
  best_i <- which.max(num(figS3a$`Balanced val score`))
  add_metric("Supplementary Figure 3", "a", figS3a$`Model / ablation`[best_i],
             "Best balanced validation score", num(figS3a$`Balanced val score`[best_i]),
             "reported unit", "figureS3a_main_ablation_table.csv")
}
figS4 <- read_source("figureS4_fig2_matched_branch_summary.csv")
for (metric in unique(figS4$metric)) {
  vals <- num(figS4$value[figS4$metric == metric])
  add_metric("Supplementary Figure 4", "summary", metric, "Mean",
             mean(vals, na.rm = TRUE), "fraction",
             "figureS4_fig2_matched_branch_summary.csv")
}
figS5 <- read_source("figureS5_random_ranking_negative_control.csv")
for (metric in unique(figS5$metric)) {
  rows <- figS5[figS5$metric == metric, , drop = FALSE]
  add_metric("Supplementary Figure 5", "summary", metric,
             "Mean observed over random-ranking fold enrichment",
             mean(num(rows$fold_enrichment), na.rm = TRUE), "fold",
             "figureS5_random_ranking_negative_control.csv")
}

main_results <- do.call(rbind, metric_rows)
stat_results <- do.call(rbind, stat_rows)

write.csv(main_results, file.path(result_dir, "reproduced_main_results.csv"),
          row.names = FALSE)
write.csv(stat_results, file.path(result_dir, "reproduced_statistical_summaries.csv"),
          row.names = FALSE)

source_files <- list.files(source_dir, recursive = TRUE, full.names = TRUE)
source_files <- source_files[file.info(source_files)$isdir == FALSE]
source_dir_norm <- gsub("\\\\", "/", normalizePath(source_dir, winslash = "/", mustWork = TRUE))
source_file_norm <- gsub("\\\\", "/", source_files)
inventory <- data.frame(
  file = sub(paste0("^", source_dir_norm, "/?"), "", source_file_norm),
  size_bytes = file.info(source_files)$size,
  stringsAsFactors = FALSE
)
inventory$rows <- NA_integer_
inventory$columns <- NA_integer_
for (i in seq_len(nrow(inventory))) {
  if (grepl("\\.csv$", inventory$file[i], ignore.case = TRUE)) {
    dat <- try(read.csv(file.path(source_dir, inventory$file[i]), nrows = 1,
                        stringsAsFactors = FALSE, check.names = FALSE), silent = TRUE)
    if (!inherits(dat, "try-error")) {
      inventory$columns[i] <- ncol(dat)
      inventory$rows[i] <- max(0L, length(readLines(file.path(source_dir, inventory$file[i]))) - 1L)
    }
  }
}
write.csv(inventory, file.path(result_dir, "source_data_file_inventory.csv"),
          row.names = FALSE)

log_lines <- c(
  "CGP-Align article-results reproduction completed.",
  paste0("Source-data files scanned: ", nrow(inventory)),
  paste0("Main result rows written: ", nrow(main_results)),
  paste0("Statistical summary rows written: ", nrow(stat_results)),
  "Raw third-party data, model checkpoints and training code were not required."
)
writeLines(log_lines, file.path(result_dir, "reproducibility_log.txt"))
cat(paste(log_lines, collapse = "\n"), "\n")
