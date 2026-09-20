suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(grid)
  library(patchwork)
})

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- args_all[grepl("^--file=", args_all)]
if (length(file_arg) > 0) {
  script_path <- normalizePath(sub("^--file=", "", file_arg[1]), mustWork = TRUE)
} else {
  script_path <- normalizePath(
    "output/cgp_align/paper/manuscript_cgp_align_profile/bib_submission_build_v3/scripts/make_figure4_integrated_relation_transfer.R",
    mustWork = TRUE
  )
}

build_dir <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
source_dir <- file.path(build_dir, "source_data")
figure_dir <- file.path(build_dir, "figures")
dir.create(figure_dir, showWarnings = FALSE, recursive = TRUE)

ink <- "#1F2D3A"
muted_ink <- "#5D6B78"
grid_col <- "#E9EEF3"
axis_col <- "#263542"
font_family <- "sans"

source_levels <- c("DGIdb", "DrugRep", "Hetionet", "OpenBioLink", "PharMeBINet")
source_cols <- c(
  DGIdb = "#3F6A9A",
  DrugRep = "#5A8F35",
  Hetionet = "#B85A6A",
  OpenBioLink = "#8C5D9E",
  PharMeBINet = "#A9852E"
)
metric_cols <- c("AUPRC gain" = "#2A9D8F", "AUROC gain" = "#7E5AA6")
entity_cols <- c(Compound = "#2368A2", Gene = "#557F2C")
bg_cols <- c(Compound = "#7FB8DE", Gene = "#A8C86A")

theme_fig <- function(base_size = 7.6) {
  theme_classic(base_size = base_size, base_family = font_family) +
    theme(
      axis.line = element_line(linewidth = 0.28, colour = axis_col),
      axis.ticks = element_line(linewidth = 0.22, colour = axis_col),
      axis.text = element_text(size = base_size * 0.92, colour = muted_ink),
      axis.title = element_text(size = base_size * 1.02, colour = ink),
      legend.title = element_text(size = base_size * 0.90, colour = ink),
      legend.text = element_text(size = base_size * 0.86, colour = ink),
      panel.grid.major = element_line(linewidth = 0.12, colour = grid_col),
      panel.grid.minor = element_blank(),
      plot.tag = element_text(
        face = "bold",
        size = base_size * 1.75,
        family = font_family,
        colour = "black"
      ),
      plot.tag.position = c(0.005, 0.998),
      plot.margin = margin(3, 4, 3, 4)
    )
}

panel_tag <- function(label, x = -Inf, y = Inf, size = 4.4) {
  labs(tag = label)
}

fmt_k <- function(x) {
  ifelse(x >= 1000, paste0(format(round(x / 1000, 1), trim = TRUE), "k"), as.character(round(x)))
}

up_intersections <- read_csv(
  file.path(source_dir, "figure4a_relation_source_upset_intersections_main.csv"),
  show_col_types = FALSE
) %>%
  mutate(pattern_id = as.integer(pattern_id))
up_matrix <- read_csv(
  file.path(source_dir, "figure4a_relation_source_upset_matrix_main.csv"),
  show_col_types = FALSE
) %>%
  mutate(
    pattern_id = as.integer(pattern_id),
    name = factor(name, levels = rev(source_levels))
  )
up_sets <- read_csv(
  file.path(source_dir, "figure4a_relation_source_upset_set_sizes_main.csv"),
  show_col_types = FALSE
) %>%
  mutate(name = factor(name, levels = rev(source_levels)))

plot_a_top <- ggplot(up_intersections, aes(x = pattern_id, y = intersection_edges)) +
  geom_col(width = 0.62, fill = "#3F6A9A", alpha = 0.92) +
  geom_text(
    aes(label = fmt_k(intersection_edges)),
    vjust = -0.25,
    size = 2.35,
    family = font_family,
    colour = ink
  ) +
  panel_tag("a") +
  scale_x_continuous(breaks = up_intersections$pattern_id, expand = expansion(mult = c(0.03, 0.05))) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.22)),
    labels = function(x) format(x, big.mark = ",", scientific = FALSE, trim = TRUE)
  ) +
  labs(x = NULL, y = "Intersection edges") +
  theme_fig(7.1) +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid.major.x = element_blank(),
    plot.margin = margin(6, 3, 0, 4)
  )

plot_a_bottom <- ggplot(up_matrix, aes(x = pattern_id, y = name)) +
  geom_line(
    data = up_matrix %>% filter(active) %>% group_by(pattern_id) %>% filter(n() > 1),
    aes(group = pattern_id),
    linewidth = 0.34,
    colour = "#7C8B97"
  ) +
  geom_point(
    aes(fill = active),
    shape = 21,
    size = 2.05,
    stroke = 0.28,
    colour = "#60717D"
  ) +
  geom_text(
    data = up_sets,
    aes(x = 10.85, y = name, label = fmt_k(set_edges)),
    inherit.aes = FALSE,
    hjust = 0,
    size = 2.25,
    family = font_family,
    colour = muted_ink
  ) +
  scale_fill_manual(values = c("FALSE" = "#E4E9ED", "TRUE" = "#3F6A9A"), guide = "none") +
  scale_x_continuous(
    breaks = up_intersections$pattern_id,
    labels = paste0("I", up_intersections$pattern_id),
    expand = expansion(mult = c(0.03, 0.18))
  ) +
  labs(x = NULL, y = NULL) +
  coord_cartesian(clip = "off") +
  theme_fig(7.1) +
  theme(
    axis.text.x = element_text(size = 5.8, colour = muted_ink),
    axis.text.y = element_text(size = 6.4, colour = ink),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    plot.margin = margin(0, 15, 3, 4)
  )

plot_a <- plot_a_top / plot_a_bottom + plot_layout(heights = c(0.58, 0.42))

lift_df <- read_csv(
  file.path(source_dir, "figure4b_target_transfer_lift50_main.csv"),
  show_col_types = FALSE
) %>%
  mutate(name = factor(name, levels = rev(source_levels)))

plot_b <- ggplot(lift_df, aes(y = name, x = mean_lift50)) +
  geom_vline(xintercept = 1, linewidth = 0.34, linetype = "dashed", colour = "#7C8B97") +
  geom_segment(aes(x = 1, xend = mean_lift50, yend = name), linewidth = 1.05, colour = "#2A9D8F", alpha = 0.92) +
  geom_point(size = 2.6, colour = "#138A7E") +
  geom_text(aes(label = sprintf("%.2f", mean_lift50)), hjust = -0.18, size = 2.45, family = font_family, colour = ink) +
  panel_tag("b") +
  scale_x_continuous(limits = c(0.85, max(lift_df$mean_lift50) + 0.7), breaks = c(1, 2, 3, 4), expand = expansion(mult = c(0.01, 0.08))) +
  labs(x = "Mean Lift@50", y = NULL) +
  theme_fig(7.3) +
  theme(panel.grid.major.y = element_blank(), plot.margin = margin(6, 4, 3, 5))

classifier_df <- read_csv(
  file.path(source_dir, "figure4c_target_transfer_leave_one_source_out_classifier_main.csv"),
  show_col_types = FALSE
) %>%
  mutate(name = factor(name, levels = source_levels))

classifier_long <- classifier_df %>%
  select(name, cosine_auprc, leave_one_source_out_auprc) %>%
  pivot_longer(
    cols = c(cosine_auprc, leave_one_source_out_auprc),
    names_to = "method",
    values_to = "AUPRC"
  ) %>%
  mutate(
    method = factor(
      method,
      levels = c("cosine_auprc", "leave_one_source_out_auprc"),
      labels = c("Cosine", "Frozen-feature\nclassifier")
    )
  )

classifier_mean <- classifier_long %>%
  group_by(method) %>%
  summarise(AUPRC = mean(AUPRC), .groups = "drop") %>%
  mutate(name = "Mean")

plot_c <- ggplot(classifier_long, aes(x = method, y = AUPRC, group = name)) +
  geom_line(colour = "#2A9D8F", alpha = 0.55, linewidth = 0.55) +
  geom_point(colour = "#2A9D8F", size = 1.8, alpha = 0.95) +
  geom_line(data = classifier_mean, aes(x = method, y = AUPRC, group = name), colour = "#7E5AA6", linewidth = 1.15) +
  geom_point(data = classifier_mean, aes(x = method, y = AUPRC), colour = "#7E5AA6", size = 2.6) +
  geom_text(
    data = classifier_mean,
    aes(label = sprintf("%.2f", AUPRC)),
    nudge_x = c(-0.09, 0.09),
    nudge_y = c(0.018, -0.022),
    size = 2.35,
    family = font_family,
    colour = "#7E5AA6"
  ) +
  panel_tag("c") +
  scale_y_continuous(limits = c(0.12, 0.64), breaks = c(0.2, 0.4, 0.6), expand = expansion(mult = c(0.02, 0.08))) +
  labs(x = NULL, y = "AUPRC") +
  theme_fig(7.3) +
  theme(panel.grid.major.x = element_blank(), plot.margin = margin(6, 4, 3, 5))

gain_df <- read_csv(
  file.path(source_dir, "figure4d_target_transfer_metric_gain_main.csv"),
  show_col_types = FALSE
) %>%
  mutate(
    name = factor(name, levels = rev(source_levels)),
    metric = factor(metric, levels = c("AUPRC gain", "AUROC gain")),
    label_y = y_pos + if_else(metric == "AUPRC gain", -0.02, 0.02)
  )

source_axis <- gain_df %>%
  distinct(source_index, name) %>%
  arrange(source_index)

plot_d <- ggplot(gain_df, aes(y = y_pos, x = gain, colour = metric)) +
  geom_vline(xintercept = 0, linewidth = 0.28, colour = axis_col) +
  geom_segment(aes(x = 0, xend = gain, yend = y_pos), linewidth = 0.76, alpha = 0.92) +
  geom_point(size = 2.05) +
  geom_text(
    aes(y = label_y, label = sprintf("+%.2f", gain)),
    hjust = -0.10,
    size = 2.05,
    family = font_family,
    colour = ink,
    show.legend = FALSE
  ) +
  panel_tag("d") +
  scale_colour_manual(values = metric_cols, name = NULL) +
  scale_x_continuous(limits = c(0, 0.43), breaks = c(0, 0.2, 0.4), expand = expansion(mult = c(0, 0.10))) +
  scale_y_continuous(
    breaks = source_axis$source_index,
    labels = as.character(source_axis$name),
    expand = expansion(mult = c(0.08, 0.12))
  ) +
  labs(x = "Absolute metric gain", y = NULL) +
  theme_fig(7.3) +
  theme(
    panel.grid.major.y = element_blank(),
    legend.position = c(0.60, 1.06),
    legend.direction = "horizontal",
    legend.background = element_rect(fill = "white", colour = NA),
    plot.margin = margin(6, 10, 3, 5)
  )

coords <- read_csv(
  file.path(source_dir, "figure4e_merged_database_entity_space_pca_coordinates.csv"),
  show_col_types = FALSE
) %>%
  mutate(entity_type = factor(entity_type, levels = c("Compound", "Gene")))

link_coords <- read_csv(
  file.path(source_dir, "figure4e_merged_database_entity_space_pca_links.csv"),
  show_col_types = FALSE
) %>%
  mutate(source_name = factor(source_name, levels = source_levels))

highlight_points <- bind_rows(
  link_coords %>%
    transmute(entity_type = "Compound", entity_id = compound_id, label = compound_short_name, pca_x = compound_x, pca_y = compound_y),
  link_coords %>%
    transmute(entity_type = "Gene", entity_id = gene_symbol, label = gene_symbol, pca_x = gene_x, pca_y = gene_y)
) %>%
  distinct(entity_type, entity_id, label, pca_x, pca_y) %>%
  mutate(entity_type = factor(entity_type, levels = c("Compound", "Gene")))

pc1_var <- unique(coords$pc1_var)[1]
pc2_var <- unique(coords$pc2_var)[1]

plot_e_base <- ggplot() +
  geom_hline(yintercept = 0, linewidth = 0.23, colour = grid_col) +
  geom_vline(xintercept = 0, linewidth = 0.23, colour = grid_col) +
  geom_point(
    data = coords %>% filter(entity_type == "Compound"),
    aes(x = pca_x, y = pca_y),
    shape = 21,
    fill = bg_cols["Compound"],
    size = 0.54,
    stroke = 0.035,
    colour = "white",
    alpha = 0.52,
    show.legend = FALSE
  ) +
  geom_point(
    data = coords %>% filter(entity_type == "Gene"),
    aes(x = pca_x, y = pca_y),
    shape = 24,
    fill = bg_cols["Gene"],
    size = 0.56,
    stroke = 0.035,
    colour = "white",
    alpha = 0.56,
    show.legend = FALSE
  ) +
  geom_point(
    data = highlight_points,
    aes(x = pca_x, y = pca_y, shape = entity_type, fill = entity_type),
    size = 2.70,
    stroke = 0.48,
    colour = "white"
  ) +
  scale_shape_manual(values = c(Compound = 21, Gene = 24), name = "Entity") +
  scale_fill_manual(values = entity_cols, name = "Entity") +
  scale_colour_manual(values = source_cols, name = "Database") +
  scale_x_continuous(expand = expansion(mult = c(0.04, 0.06))) +
  scale_y_continuous(expand = expansion(mult = c(0.04, 0.06))) +
  labs(
    x = paste0("PC1 (", sprintf("%.1f", 100 * pc1_var), "%)"),
    y = paste0("PC2 (", sprintf("%.1f", 100 * pc2_var), "%)")
  ) +
  coord_cartesian(clip = "off") +
  theme_fig(7.4) +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.box.spacing = unit(0.3, "mm"),
    legend.key.height = unit(3.0, "mm"),
    legend.key.width = unit(5.6, "mm"),
    plot.margin = margin(10, 8, 4, 9)
  ) +
  guides(
    fill = "none",
    shape = guide_legend(
      order = 1,
      override.aes = list(fill = unname(entity_cols), colour = "white", size = 2.25, alpha = 1)
    ),
    colour = guide_legend(order = 2, override.aes = list(linewidth = 0.75, alpha = 1))
  ) +
  panel_tag("e", size = 4.2)

for (source_i in source_levels) {
  source_data <- link_coords %>% filter(source_name == source_i)
  if (nrow(source_data) == 0) next
  curve_i <- unique(source_data$source_curve)
  curve_i <- curve_i[!is.na(curve_i)][1]
  if (length(curve_i) == 0) curve_i <- 0
  plot_e_base <- plot_e_base +
    geom_curve(
      data = source_data,
      aes(x = compound_x, y = compound_y, xend = gene_x, yend = gene_y, colour = source_name),
      curvature = curve_i,
      linewidth = 0.52,
      alpha = 0.90,
      lineend = "round"
    )
}

plot_e <- plot_e_base + ggrepel::geom_label_repel(data = highlight_points,
 aes(x=pca_x,y=pca_y,label=label), colour=ink, size=2.15, family=font_family,
 fill="white", linewidth=0.12, box.padding=0.7, point.padding=0.25,
 max.overlaps=Inf, seed=41, min.segment.length=0, segment.colour="#AAB7C1")

layout_design <- "
AABC
AADD
EEEE
EEEE
"

figure4 <- wrap_plots(
  A = plot_a,
  B = plot_b,
  C = plot_c,
  D = plot_d,
  E = plot_e,
  design = layout_design
) +
  plot_layout(widths = c(1.15, 1.15, 1.0, 1.25), heights = c(0.82, 0.82, 1.62, 1.62)) &
  theme(plot.background = element_rect(fill = "white", colour = NA))

out_base <- file.path(figure_dir, "figure4")
ggsave(paste0(out_base, ".pdf"), figure4, width = 7.45, height = 9.35, units = "in", device = cairo_pdf)
ggsave(paste0(out_base, ".svg"), figure4, width = 7.45, height = 9.35, units = "in", device = svglite::svglite)
ggsave(paste0(out_base, ".png"), figure4, width = 7.45, height = 9.35, units = "in", dpi = 600, bg = "white")
ggsave(paste0(out_base, ".tiff"), figure4, width = 7.45, height = 9.35, units = "in", dpi = 600, compression = "lzw", bg = "white")

cat("Wrote ", out_base, "\n", sep = "")
graphics.off()
invisible(gc())
quit(save = "no", status = 0, runLast = FALSE)
