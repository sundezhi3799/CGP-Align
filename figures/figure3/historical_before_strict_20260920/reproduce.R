#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(patchwork)
  library(scales)
})

cli <- commandArgs(trailingOnly = TRUE)
script_arg <- grep("^--file=", commandArgs(), value = TRUE)[1]
base_dir <- dirname(normalizePath(sub("^--file=", "", script_arg), winslash = "/"))
source_dir <- file.path(base_dir, "source_data")
figure_dir <- if (length(cli) >= 1) cli[1] else file.path(base_dir, "../../outputs/figure3")
font_family <- if (length(cli) >= 2) cli[2] else "Arial"
out_prefix <- file.path(figure_dir, "figure3")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

pal <- c(
  ink = "#17212E",
  muted = "#6F7B8A",
  grid = "#E6ECF2",
  cgp = "#0B4C78",
  gene = "#128A7D",
  smiles = "#7353A2",
  smiles2 = "#9B83BF",
  fingerprint = "#C96A3A",
  neutral = "#9AA7B4"
)

method_levels <- c(
  "CGP-Align",
  "SMILES-LM profile (MoLFormer)",
  "SMILES-LM profile (ChemBERTa)",
  "Fingerprint-profile baseline"
)

method_colors <- c(
  "CGP-Align" = pal[["cgp"]],
  "SMILES-LM profile (MoLFormer)" = pal[["smiles"]],
  "SMILES-LM profile (ChemBERTa)" = pal[["smiles2"]],
  "Fingerprint-profile baseline" = pal[["fingerprint"]]
)

method_axis_labels <- c(
  "CGP-Align" = "CGP-Align",
  "SMILES-LM profile (MoLFormer)" = "MoLFormer",
  "SMILES-LM profile (ChemBERTa)" = "ChemBERTa",
  "Fingerprint-profile baseline" = "RDKit-Morgan"
)

theme_set(
  theme_classic(base_size = 7.0, base_family = font_family) +
    theme(
      axis.line = element_line(linewidth = 0.32, colour = pal[["ink"]]),
      axis.ticks = element_line(linewidth = 0.28, colour = pal[["ink"]]),
      axis.title = element_text(size = 7.2, colour = pal[["ink"]]),
      axis.text = element_text(size = 6.7, colour = pal[["ink"]]),
      plot.title = element_text(face = "bold", size = 8.0, colour = pal[["ink"]],
                                margin = margin(b = 3)),
      strip.text = element_text(size = 6.6, face = "bold", colour = pal[["ink"]]),
      strip.background = element_blank(),
      panel.grid.major.x = element_line(linewidth = 0.25, colour = pal[["grid"]]),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "none",
      plot.margin = margin(5, 8, 5, 8)
    )
)

fmt_pct <- function(x, digits = 1) sprintf(paste0("%.", digits, "f%%"), 100 * as.numeric(x))

panel_title <- function(letter, title) paste0(letter, "  ", title)

# a-b. PCA embedding diagnostics ------------------------------------------------
pca_source <- file.path(source_dir, "figure2ab_four_branch_main_pca_source_data.csv")
pca_metadata <- file.path(source_dir, "figure2ab_four_branch_main_pca_metadata.json")

latent_df <- read_csv(pca_source, show_col_types = FALSE)

branch_cols <- c(
  "Compound" = pal[["cgp"]],
  "ORF gene" = pal[["gene"]],
  "CRISPR gene" = "#D56E32"
)

if (!"branch_group" %in% names(latent_df)) {
  latent_df$branch_group <- dplyr::case_when(
    latent_df$pair_kind == "compound" ~ "Compound",
    latent_df$perturbation_modality == "orf" ~ "ORF gene",
    latent_df$perturbation_modality == "crispr" ~ "CRISPR gene",
    TRUE ~ NA_character_
  )
}

if (!"representation" %in% names(latent_df)) {
  latent_df$representation <- dplyr::case_when(
    latent_df$modality %in% c("compound_structure", "gene_embedding") ~ "Entity encoder",
    TRUE ~ "Profile encoder"
  )
}

latent_df <- latent_df %>%
  mutate(
    plot_x = pca1,
    plot_y = pca2,
    stage_short = dplyr::recode(
      if ("stage_key" %in% names(.)) stage_key else stage_label,
      "branch_pretrained" = "Before joint training",
      "joint_aligned" = "After joint training",
      "Native single-branch embeddings" = "Before joint training",
      "CGP-Align joint embedding" = "After joint training",
      .default = stage_label
    ),
    stage_short = factor(stage_short, levels = c("Before joint training", "After joint training")),
    branch_group = factor(branch_group, levels = c("Compound", "ORF gene", "CRISPR gene")),
    representation = factor(representation, levels = c("Entity encoder", "Profile encoder"))
  ) %>%
  filter(!is.na(stage_short), !is.na(branch_group), !is.na(representation))

set.seed(1701)
latent_entity <- latent_df %>%
  filter(representation == "Entity encoder") %>%
  group_by(stage_short, branch_group) %>%
  group_modify(~ dplyr::slice_sample(.x, n = min(360, nrow(.x)))) %>%
  ungroup()

latent_profile_cloud <- latent_df %>%
  filter(representation == "Profile encoder") %>%
  group_by(stage_short, branch_group) %>%
  group_modify(~ dplyr::slice_sample(.x, n = min(900, nrow(.x)))) %>%
  ungroup()

extract_pca_variance <- function(metadata_path) {
  if (!file.exists(metadata_path)) {
    return(data.frame(stage_short = character(), pc1 = numeric(), pc2 = numeric()))
  }
  metadata_text <- paste(readLines(metadata_path, warn = FALSE), collapse = "\n")
  extract_stage <- function(stage_key, stage_short) {
    pattern <- paste0(
      '"', stage_key, '"\\s*:\\s*\\{[^\\}]*?"explained_variance_ratio"\\s*:\\s*\\[\\s*',
      '([0-9eE+\\.-]+)\\s*,\\s*([0-9eE+\\.-]+)'
    )
    match <- regexec(pattern, metadata_text, perl = TRUE)
    parsed <- regmatches(metadata_text, match)[[1]]
    if (length(parsed) < 3) {
      return(data.frame(stage_short = stage_short, pc1 = NA_real_, pc2 = NA_real_))
    }
    data.frame(stage_short = stage_short, pc1 = as.numeric(parsed[2]), pc2 = as.numeric(parsed[3]))
  }
  rbind(
    extract_stage("branch_pretrained", "Before joint training"),
    extract_stage("joint_aligned", "After joint training")
  )
}

pca_variance <- extract_pca_variance(pca_metadata)

pca_axis_label <- function(stage_value, pc) {
  row <- pca_variance %>% filter(stage_short == stage_value)
  value <- if (pc == 1) row$pc1 else row$pc2
  if (length(value) == 1 && !is.na(value)) {
    sprintf("PC%d (%.1f%%)", pc, value * 100)
  } else {
    sprintf("PC%d", pc)
  }
}

pca_xlim <- range(latent_df$plot_x, na.rm = TRUE)
pca_ylim <- range(latent_df$plot_y, na.rm = TRUE)
pca_xpad <- diff(pca_xlim) * 0.045
pca_ypad <- diff(pca_ylim) * 0.045
pca_xlim <- pca_xlim + c(-pca_xpad, pca_xpad)
pca_ylim <- pca_ylim + c(-pca_ypad, pca_ypad)
pca_x_breaks <- pretty(pca_xlim, n = 4)
pca_y_breaks <- pretty(pca_ylim, n = 4)

make_pca_panel <- function(stage_value, letter, title_text) {
  pca_legend <- data.frame(
    branch_group = factor(c("Compound", "ORF gene", "CRISPR gene"),
                          levels = c("Compound", "ORF gene", "CRISPR gene")),
    label = c("Compound", "ORF", "CRISPR"),
    plot_x = pca_xlim[1] + diff(pca_xlim) * 0.035,
    plot_y = pca_ylim[2] - diff(pca_ylim) * c(0.075, 0.145, 0.215)
  )
  key_x <- pca_xlim[1] + diff(pca_xlim) * 0.040
  key_label_x <- pca_xlim[1] + diff(pca_xlim) * 0.075
  key_y_entity <- pca_ylim[1] + diff(pca_ylim) * 0.145
  key_y_profile <- pca_ylim[1] + diff(pca_ylim) * 0.080

  ggplot() +
    stat_density_2d(
      data = latent_profile_cloud %>% filter(stage_short == stage_value),
      aes(x = plot_x, y = plot_y, fill = branch_group, colour = branch_group),
      geom = "polygon",
      bins = 5,
      alpha = 0.12,
      linewidth = 0,
      show.legend = FALSE
    ) +
    geom_point(
      data = latent_profile_cloud %>% filter(stage_short == stage_value),
      aes(x = plot_x, y = plot_y, colour = branch_group),
      shape = 1,
      size = 0.30,
      stroke = 0.18,
      alpha = 0.34,
      show.legend = FALSE
    ) +
    geom_point(
      data = latent_entity %>% filter(stage_short == stage_value),
      aes(x = plot_x, y = plot_y, colour = branch_group),
      shape = 16,
      size = 0.70,
      alpha = 0.52,
      stroke = 0,
      show.legend = FALSE
    ) +
    geom_text(
      data = pca_legend,
      aes(x = plot_x, y = plot_y, label = label, colour = branch_group),
      hjust = 0,
      size = 2.35,
      fontface = "bold",
      show.legend = FALSE
    ) +
    annotate(
      "point",
      x = key_x, y = key_y_entity,
      shape = 16, size = 1.00,
      colour = pal[["muted"]], alpha = 0.72
    ) +
    annotate(
      "text",
      x = key_label_x, y = key_y_entity,
      label = "Perturbation",
      hjust = 0, vjust = 0.5,
      size = 1.90,
      colour = pal[["muted"]]
    ) +
    annotate(
      "point",
      x = key_x, y = key_y_profile,
      shape = 1, size = 1.00,
      stroke = 0.35,
      colour = pal[["muted"]], alpha = 0.72
    ) +
    annotate(
      "text",
      x = key_label_x, y = key_y_profile,
      label = "Profile",
      hjust = 0, vjust = 0.5,
      size = 1.90,
      colour = pal[["muted"]]
    ) +
    scale_colour_manual(values = branch_cols, guide = "none") +
    scale_fill_manual(values = branch_cols, guide = "none") +
    scale_x_continuous(breaks = pca_x_breaks) +
    scale_y_continuous(breaks = pca_y_breaks) +
    coord_cartesian(xlim = pca_xlim, ylim = pca_ylim, expand = FALSE) +
    labs(
      title = panel_title(letter, title_text),
      x = pca_axis_label(stage_value, 1),
      y = pca_axis_label(stage_value, 2)
    ) +
    theme(
      panel.grid.major = element_line(linewidth = 0.22, colour = pal[["grid"]]),
      panel.grid.minor = element_blank(),
      axis.text = element_text(size = 5.8, colour = pal[["muted"]]),
      axis.ticks = element_line(linewidth = 0.25, colour = pal[["muted"]]),
      axis.ticks.length = grid::unit(1.4, "mm"),
      axis.title = element_text(size = 6.8, colour = pal[["ink"]]),
      plot.margin = margin(5, 8, 5, 8)
    )
}

p_pca_a <- make_pca_panel("Before joint training", "a", "Before joint alignment")
p_pca_b <- make_pca_panel("After joint training", "b", "After CGP-Align training")

# c. Intrinsic sampled retrieval ------------------------------------------------
dir_runs <- read_csv(file.path(source_dir, "figure2c_direction_top10_run_values.csv"),
                     show_col_types = FALSE) %>%
  mutate(
    top10_pct = top10 * 100,
    direction = factor(direction,
                       levels = c("Compound -> profile", "Profile -> compound",
                                  "Gene -> profile", "Profile -> gene")),
    route = factor(route, levels = c("Compound-profile", "Gene-profile"))
  )

dir_summary <- dir_runs %>%
  group_by(direction, route) %>%
  summarise(mean = mean(top10_pct), sd = sd(top10_pct), .groups = "drop")

dir_random <- read_csv(file.path(source_dir, "figure2c_random_ranking_top10.csv"),
                       show_col_types = FALSE) %>%
  mutate(
    random_pct = random_ranking_top10 * 100,
    direction = factor(direction, levels = levels(dir_runs$direction))
  )

p_a <- ggplot(dir_summary, aes(y = direction, x = mean, colour = route)) +
  geom_point(data = dir_random, aes(y = direction, x = random_pct),
             inherit.aes = FALSE, shape = 124, size = 3.7, colour = pal[["neutral"]]) +
  geom_text(data = dir_random %>% filter(direction == "Profile -> gene"),
            aes(y = direction, x = random_pct + 3.0, label = "random"),
            inherit.aes = FALSE, hjust = 0, size = 1.85,
            colour = pal[["neutral"]]) +
  geom_errorbar(aes(xmin = mean - sd, xmax = mean + sd), width = 0.16,
                linewidth = 0.55, orientation = "y") +
  geom_segment(aes(x = mean, xend = mean,
                   y = as.numeric(direction) - 0.16,
                   yend = as.numeric(direction) + 0.16),
               linewidth = 0.95) +
  geom_text(aes(label = sprintf("%.1f", mean), x = pmin(mean + 4.0, 92)),
            hjust = 0, size = 2.25, colour = pal[["ink"]], fontface = "bold") +
  scale_colour_manual(values = c("Compound-profile" = pal[["cgp"]], "Gene-profile" = pal[["gene"]])) +
  scale_y_discrete(labels = c(
    "Compound -> profile" = "C->P",
    "Profile -> compound" = "P->C",
    "Gene -> profile" = "G->P",
    "Profile -> gene" = "P->G"
  )) +
  scale_x_continuous(limits = c(0, 100), breaks = c(0, 25, 50, 75, 100)) +
  coord_cartesian(clip = "off") +
  labs(title = panel_title("c", "Sampled retrieval"),
       x = "Top-10 hit rate (%)", y = NULL) +
  theme(axis.text.y = element_text(size = 6.8),
        plot.margin = margin(5, 12, 5, 5))

# d. Intrinsic full-gallery enrichment -----------------------------------------
full <- read_csv(file.path(source_dir, "figureS5_random_ranking_negative_control.csv"),
                 show_col_types = FALSE) %>%
  filter(candidate_setting == "full gallery", metric == "Recall@10") %>%
  mutate(
    direction = factor(direction,
                       levels = c("Compound to profile", "Profile to compound",
                                  "Gene to profile", "Profile to gene")),
    branch = factor(branch, levels = c("compound-profile", "gene-profile"),
                    labels = c("Compound-profile", "Gene-profile"))
  )

full_summary <- full %>%
  group_by(direction, branch) %>%
  summarise(mean = mean(fold_enrichment), sd = sd(fold_enrichment), .groups = "drop")

p_b <- ggplot(full_summary, aes(y = direction, x = mean, colour = branch)) +
  geom_vline(xintercept = 1, colour = pal[["neutral"]], linewidth = 0.40, linetype = "dashed") +
  annotate("text", x = 3.2, y = 4.25, label = "random", hjust = 0,
           size = 1.85, colour = pal[["neutral"]]) +
  geom_errorbar(aes(xmin = pmax(mean - sd, 0), xmax = mean + sd), width = 0.16,
                linewidth = 0.55, orientation = "y") +
  geom_segment(aes(x = mean, xend = mean,
                   y = as.numeric(direction) - 0.16,
                   yend = as.numeric(direction) + 0.16),
               linewidth = 0.95) +
  geom_text(aes(label = sprintf("%.1f", mean), x = mean + sd + 2.5),
            hjust = 0, size = 2.25, colour = pal[["ink"]], fontface = "bold") +
  scale_colour_manual(values = c("Compound-profile" = pal[["cgp"]], "Gene-profile" = pal[["gene"]])) +
  scale_y_discrete(labels = c(
    "Compound to profile" = "C->P",
    "Profile to compound" = "P->C",
    "Gene to profile" = "G->P",
    "Profile to gene" = "P->G"
  )) +
  scale_x_continuous(limits = c(0, max(full_summary$mean + full_summary$sd) + 11),
                     breaks = c(0, 20, 40, 60, 80)) +
  coord_cartesian(clip = "off") +
  labs(title = panel_title("d", "Full-gallery enrichment"),
       x = "Fold over random", y = NULL) +
  theme(axis.text.y = element_text(size = 6.8),
        plot.margin = margin(5, 12, 5, 5))

# e. Branch summary -------------------------------------------------------------
branch <- read_csv(file.path(source_dir, "figure2d_branch_summary_mean_sd.csv"),
                   show_col_types = FALSE) %>%
  mutate(
    metric = factor(metric, levels = c("Mean Top-1", "HMean Top-1", "Mean Top-10", "HMean Top-10")),
    mean_pct = mean * 100,
    sd_pct = sd * 100,
    random_pct = random_ranking_mean * 100,
    summary_type = factor(summary_type, levels = c("Mean", "HMean"))
  )

p_c <- ggplot(branch, aes(y = metric, x = mean_pct, colour = summary_type)) +
  geom_point(aes(y = metric, x = random_pct),
             inherit.aes = FALSE, shape = 124, size = 3.7, colour = pal[["neutral"]]) +
  geom_text(data = branch %>% filter(metric == "HMean Top-10"),
            aes(y = metric, x = random_pct + 2.8, label = "random"),
            inherit.aes = FALSE, hjust = 0, size = 1.85,
            colour = pal[["neutral"]]) +
  geom_errorbar(aes(xmin = pmax(mean_pct - sd_pct, 0), xmax = mean_pct + sd_pct),
                width = 0.16, linewidth = 0.55, orientation = "y") +
  geom_segment(aes(x = mean_pct, xend = mean_pct,
                   y = as.numeric(metric) - 0.16,
                   yend = as.numeric(metric) + 0.16),
               linewidth = 0.95) +
  geom_text(aes(label = sprintf("%.1f", mean_pct), x = mean_pct + 3.0),
            hjust = 0, size = 2.25, colour = pal[["ink"]], fontface = "bold") +
  scale_colour_manual(values = c("Mean" = pal[["cgp"]], "HMean" = pal[["gene"]])) +
  scale_x_continuous(limits = c(0, 72), breaks = c(0, 20, 40, 60)) +
  coord_cartesian(clip = "off") +
  labs(title = panel_title("e", "Branch summary"),
       x = "Summary retrieval score (%)", y = NULL) +
  theme(axis.text.y = element_text(size = 6.8),
        plot.margin = margin(5, 12, 5, 5))

# f. Matched compound-profile benchmark ----------------------------------------
main_b <- read_csv(file.path(source_dir, "figure3_panelB_retrieval_performance.csv"),
                   show_col_types = FALSE) %>%
  mutate(
    method_label = factor(method_label, levels = rev(method_levels)),
    label = fmt_pct(mean_top10, 1),
    label_x = mean_top10 + sd_top10 + 0.010
  )

p_d <- ggplot(main_b, aes(x = mean_top10, y = method_label, colour = method_label)) +
  geom_segment(aes(x = 0.52, xend = mean_top10, yend = method_label), linewidth = 0.95, alpha = 0.46) +
  geom_errorbar(aes(xmin = mean_top10 - sd_top10, xmax = mean_top10 + sd_top10),
                orientation = "y", width = 0.10, linewidth = 0.34, colour = pal[["ink"]]) +
  geom_errorbar(aes(xmin = mean_top10, xmax = mean_top10),
                orientation = "y", width = 0.30, linewidth = 0.62) +
  geom_text(aes(x = label_x, label = label), hjust = 0, size = 2.2, colour = pal[["ink"]]) +
  scale_colour_manual(values = method_colors) +
  scale_y_discrete(labels = method_axis_labels) +
  scale_x_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0.52, 0.79), breaks = seq(0.55, 0.75, by = 0.05)) +
  coord_cartesian(clip = "off") +
  labs(title = panel_title("f", "Matched benchmark"),
       x = "Top-10 accuracy", y = NULL) +
  theme(axis.text.y = element_text(size = 6.7),
        plot.margin = margin(5, 16, 5, 5))

# g. Harder matched compound-profile retrieval ---------------------------------
hard_df <- read_csv(file.path(source_dir, "figure3_panelC_difficult_retrieval.csv"),
                    show_col_types = FALSE) %>%
  mutate(
    method_label = factor(method_label, levels = rev(method_levels)),
    metric_label = factor(metric_label, levels = c("Sampled 1:1000\nTop-10", "Full-gallery\nRecall@10")),
    label = fmt_pct(value, 1),
    label_x = value + sd + ifelse(metric == "full_r10", 0.003, 0.008)
  )

p_e <- ggplot(hard_df, aes(x = value, y = method_label, colour = method_label)) +
  geom_segment(aes(x = 0, xend = value, yend = method_label), linewidth = 0.82, alpha = 0.44) +
  geom_errorbar(aes(xmin = pmax(value - sd, 0), xmax = value + sd),
                orientation = "y", width = 0.09, linewidth = 0.30, colour = pal[["ink"]]) +
  geom_errorbar(aes(xmin = value, xmax = value),
                orientation = "y", width = 0.28, linewidth = 0.58) +
  geom_text(aes(x = label_x, label = label), hjust = 0, size = 1.95, colour = pal[["ink"]]) +
  facet_wrap(~metric_label, nrow = 1, scales = "free_x") +
  scale_colour_manual(values = method_colors) +
  scale_y_discrete(labels = method_axis_labels) +
  scale_x_continuous(labels = percent_format(accuracy = 1),
                     breaks = c(0, 0.10, 0.20, 0.30, 0.40),
                     expand = expansion(mult = c(0, 0.40))) +
  coord_cartesian(clip = "off") +
  labs(title = panel_title("g", "Difficult retrieval"), x = NULL, y = NULL) +
  theme(axis.text.y = element_text(size = 5.7),
        axis.ticks.y = element_blank(),
        strip.text = element_text(size = 6.2, face = "bold"),
        panel.spacing.x = unit(13, "pt"),
        plot.margin = margin(5, 15, 5, 5))

# h. Hidden phenotype-neighbour recovery ----------------------------------------
hidden_dir <- source_dir

hidden_recovery <- read_csv(file.path(hidden_dir, "hidden_phenotype_neighbour_recovery_compact.csv"),
                            show_col_types = FALSE) %>%
  filter(scope == "compound_to_compound",
         k == 50,
         method %in% c("Structure feature baseline",
                       "CGP-Align entity",
                       "CGP entity -> profile anchor")) %>%
  mutate(
    method_axis = dplyr::recode(
      method,
      "Structure feature baseline" = "Structure\nbaseline",
      "CGP-Align entity" = "CGP-Align\nentity",
      "CGP entity -> profile anchor" = "Profile-anchor\nreadout"
    ),
    method_axis = factor(method_axis,
                         levels = c("Structure\nbaseline",
                                    "CGP-Align\nentity",
                                    "Profile-anchor\nreadout")),
    method_idx = as.numeric(method_axis),
    method_family = dplyr::recode(
      method,
      "Structure feature baseline" = "Structure",
      "CGP-Align entity" = "CGP-Align",
      "CGP entity -> profile anchor" = "Profile anchor"
    ),
    label = paste0(sprintf("%.2f", precision_enrichment), "x"),
    label_x = precision_enrichment + 0.35
  ) %>%
  arrange(method_axis)

write_csv(hidden_recovery,
          file.path(figure_dir, "figure3h_hidden_phenotype_neighbour_recovery.csv"))

hidden_colors <- c(
  "Structure" = pal[["neutral"]],
  "CGP-Align" = pal[["cgp"]],
  "Profile anchor" = pal[["gene"]]
)

p_f <- ggplot(hidden_recovery,
              aes(y = method_idx, x = precision_enrichment, colour = method_family)) +
  geom_vline(xintercept = 1, colour = pal[["neutral"]], linewidth = 0.40,
             linetype = "dashed") +
  annotate("text", x = 1.15, y = 0.70, label = "random", hjust = 0,
           size = 1.85, colour = pal[["neutral"]]) +
  geom_segment(aes(x = 1, xend = precision_enrichment, yend = method_idx),
               linewidth = 2.1, alpha = 0.24) +
  geom_segment(aes(x = precision_enrichment, xend = precision_enrichment,
                   y = method_idx - 0.16, yend = method_idx + 0.16),
               linewidth = 0.75) +
  geom_text(aes(x = label_x, label = label), hjust = 0,
            size = 2.20, colour = pal[["ink"]], fontface = "bold") +
  scale_colour_manual(values = hidden_colors) +
  scale_x_continuous(limits = c(0, 10.8), breaks = c(0, 2.5, 5.0, 7.5, 10.0)) +
  scale_y_continuous(breaks = hidden_recovery$method_idx,
                     labels = hidden_recovery$method_axis,
                     limits = c(0.55, 3.45)) +
  coord_cartesian(clip = "off") +
  labs(title = panel_title("h", "Hidden neighbours"),
       x = "Top-50 enrichment over random", y = NULL) +
  theme(axis.text.y = element_text(size = 6.4, lineheight = 0.94),
        axis.text.x = element_text(size = 6.4),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(linewidth = 0.25, colour = pal[["grid"]]),
        plot.margin = margin(5, 16, 5, 5))

fig_design <- "
AAABBB
CCDDEE
FFGGHH
"

fig <- p_pca_a + p_pca_b + p_a + p_b + p_c + p_d + p_e + p_f +
  plot_layout(design = fig_design, heights = c(1.10, 0.95, 1.05)) &
  theme(plot.background = element_rect(fill = "white", colour = NA),
        plot.margin = margin(5, 6, 5, 6))

save_pub <- function(plot, prefix, width_mm = 183, height_mm = 142, dpi = 600) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4

  ragg::agg_png(paste0(prefix, ".png"), width = w, height = h,
                units = "in", res = dpi, background = "white")
  print(plot)
  invisible(dev.off())

  grDevices::tiff(paste0(prefix, ".tiff"), width = w, height = h,
                  units = "in", res = dpi, bg = "white",
                  compression = "lzw", type = "cairo")
  print(plot)
  invisible(dev.off())

  grDevices::cairo_pdf(paste0(prefix, ".pdf"), width = w, height = h,
                       family = font_family, bg = "white")
  print(plot)
  invisible(dev.off())

  svglite::svglite(paste0(prefix, ".svg"), width = w, height = h,
                    bg = "white")
  print(plot)
  invisible(dev.off())
}

save_pub(fig, out_prefix, width_mm = 183, height_mm = 188, dpi = 600)
message("Wrote merged Figure 3 variant to: ", out_prefix)

writeLines(capture.output(sessionInfo()), file.path(figure_dir, "sessionInfo.txt"))
