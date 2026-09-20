suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(patchwork)
  library(scales)
})

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args_all[grep("^--file=", args_all)][1])
if (is.na(file_arg)) {
  file_arg <- file.path("scripts", "make_figureS2_data_qc_split.R")
}
root <- normalizePath(file.path(dirname(normalizePath(file_arg, winslash = "/", mustWork = TRUE)), ".."), winslash = "/", mustWork = TRUE)
source_dir <- file.path(root, "source_data")
figure_dir <- file.path(root, "figures")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

composition <- read_csv(file.path(source_dir, "figureS2a_dataset_composition.csv"), show_col_types = FALSE) %>%
  mutate(modality = factor(modality, levels = c("Compound", "ORF", "CRISPR")))
rep_dist <- read_csv(file.path(source_dir, "figureS2b_entity_replicate_distribution.csv"), show_col_types = FALSE) %>%
  mutate(modality = factor(modality, levels = c("Compound", "ORF", "CRISPR")))
split_comp <- read_csv(file.path(source_dir, "figureS2c_split_composition.csv"), show_col_types = FALSE) %>%
  mutate(
    modality = factor(modality, levels = c("Compound", "ORF", "CRISPR")),
    split = factor(split, levels = c("train", "val", "test"))
  )
overlap <- read_csv(file.path(source_dir, "figureS2c_split_overlap_check.csv"), show_col_types = FALSE)
r2_df <- read_csv(file.path(source_dir, "figureS2d_post_correction_r2.csv"), show_col_types = FALSE) %>%
  mutate(
    dataset = factor(dataset, levels = c("Compound", "Gene")),
    factor = recode(factor, "entity" = "Biological entity", "plate" = "Plate", "well" = "Well"),
    factor = factor(factor, levels = c("Biological entity", "Plate", "Well"))
  )
theme_set(
  theme_classic(base_size = 8.3, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "#222222"),
      axis.ticks = element_line(linewidth = 0.32, colour = "#222222"),
      axis.text = element_text(colour = "#222222"),
      axis.title = element_text(colour = "#111111", size = 8.4),
      plot.title = element_text(face = "bold", size = 9.2, colour = "#111111", margin = margin(b = 4)),
      legend.title = element_blank(),
      legend.text = element_text(size = 7.2),
      plot.subtitle = element_text(size = 7.4, colour = "#52616F", margin = margin(b = 4)),
      plot.margin = margin(7, 9, 7, 9),
      panel.grid.major.y = element_line(linewidth = 0.22, colour = "#E7ECF2"),
      panel.grid.major.x = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 8.2)
    )
)

pal_mod <- c("Compound" = "#2F80ED", "ORF" = "#009E73", "CRISPR" = "#D55E00")
pal_split <- c("train" = "#102A43", "val" = "#8A99A8", "test" = "#D9E1E8")
pal_factor <- c("Biological entity" = "#102A43", "Plate" = "#B23A48", "Well" = "#F2A65A")

count_long <- composition %>%
  pivot_longer(c(entity_count, profile_count), names_to = "count_type", values_to = "count") %>%
  mutate(
    count_type = recode(count_type, entity_count = "Entities", profile_count = "Profiles"),
    count_label = case_when(
      count >= 1e5 ~ paste0(round(count / 1e3, 0), "k"),
      count >= 1e3 ~ paste0(round(count / 1e3, 1), "k"),
      TRUE ~ as.character(count)
    )
  )

p_a <- ggplot(count_long, aes(modality, count, fill = modality)) +
  geom_col(width = 0.62, colour = NA) +
  geom_text(aes(label = count_label), vjust = -0.45, size = 2.7, family = "Arial") +
  facet_wrap(~count_type, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = pal_mod) +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale()), expand = expansion(mult = c(0, 0.12))) +
  labs(title = "Raw3180 perturbation resource scale", x = NULL, y = "Count") +
  theme(legend.position = "none")

rep_plot_df <- rep_dist %>%
  mutate(num_replicates_cap = pmin(num_replicates, 25))

rep_summary <- rep_dist %>%
  group_by(modality) %>%
  summarise(median_rep = median(num_replicates), p90_rep = quantile(num_replicates, 0.9), .groups = "drop")

p_b <- ggplot(rep_plot_df, aes(modality, num_replicates_cap, fill = modality)) +
  geom_violin(width = 0.82, alpha = 0.28, colour = NA, trim = TRUE) +
  geom_boxplot(width = 0.18, outlier.shape = NA, linewidth = 0.35, alpha = 0.85) +
  geom_text(
    data = rep_summary,
    aes(x = modality, y = pmin(p90_rep, 25) + 1.2, label = paste0("median ", median_rep)),
    inherit.aes = FALSE, size = 2.7, family = "Arial"
  ) +
  scale_fill_manual(values = pal_mod) +
  scale_y_continuous(breaks = c(1, 5, 10, 15, 20, 25), labels = c("1", "5", "10", "15", "20", ">=25")) +
  labs(title = "Replicate support per entity", x = NULL, y = "Replicates per entity") +
  theme(legend.position = "none")

split_long <- split_comp %>%
  pivot_longer(c(entity_count, profile_count), names_to = "count_type", values_to = "count") %>%
  mutate(count_type = recode(count_type, entity_count = "Entities", profile_count = "Profiles"))

overlap_label <- paste0("pairwise split overlap = ", max(overlap$overlap_entities), " entities")

p_c <- ggplot(split_long, aes(modality, count, fill = split)) +
  geom_col(width = 0.62, colour = "white", linewidth = 0.2) +
  facet_wrap(~count_type, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = pal_split) +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale())) +
  labs(title = "Entity-held-out cold splits", subtitle = overlap_label, x = NULL, y = "Count") +
  theme(legend.position = "bottom")

p_d <- ggplot(r2_df, aes(factor, mean_r2, colour = factor, group = factor)) +
  geom_errorbar(aes(ymin = median_r2, ymax = p90_r2), width = 0.12, linewidth = 0.45) +
  geom_point(size = 2.2) +
  facet_wrap(~dataset, nrow = 1) +
  scale_colour_manual(values = pal_factor) +
  scale_y_continuous(labels = percent_format(accuracy = 0.1), expand = expansion(mult = c(0.02, 0.10))) +
  labs(title = "Batch residuals are suppressed after correction", x = NULL, y = expression("Post-correction "*R^2)) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 25, hjust = 1)
  )

fig <- (p_a | p_b) / (p_c | p_d) +
  plot_layout(heights = c(1, 1), widths = c(1, 1)) +
  plot_annotation(tag_levels = "a") &
  theme(
    plot.margin = margin(7, 9, 7, 9),
    plot.tag = element_text(size = 13, face = "bold", colour = "#111111"),
    plot.tag.position = c(0, 1),
    plot.title = element_blank(),
    plot.subtitle = element_blank()
  )

out_base <- file.path(figure_dir, "figureS2_data_qc_split")
width_mm <- 183
height_mm <- 132
w <- width_mm / 25.4
h <- height_mm / 25.4

ggsave(paste0(out_base, ".png"), fig, width = w, height = h, dpi = 600, bg = "white", device = ragg::agg_png)
ggsave(paste0(out_base, ".tiff"), fig, width = w, height = h, dpi = 600, bg = "white", device = ragg::agg_tiff, compression = "lzw")
ggsave(paste0(out_base, ".pdf"), fig, width = w, height = h, bg = "white", device = cairo_pdf, family = "Arial")
ggsave(paste0(out_base, ".svg"), fig, width = w, height = h, bg = "white", device = svglite::svglite)

message("Wrote: ", out_base)
