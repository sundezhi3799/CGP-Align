#!/usr/bin/env Rscript

required_pkgs <- c("ggplot2", "dplyr", "tidyr", "patchwork", "svglite", "readr", "scales", "ragg")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop("Missing required R packages: ", paste(missing_pkgs, collapse = ", "), call. = FALSE)
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(readr)
  library(scales)
})

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args_all[grep("^--file=", args_all)][1])
if (is.na(file_arg)) {
  file_arg <- file.path("scripts", "make_figure5_toxicity_alert_complementarity.R")
}
out_root <- normalizePath(file.path(dirname(normalizePath(file_arg, winslash = "/", mustWork = TRUE)), ".."), winslash = "/", mustWork = TRUE)
out_dir <- file.path(out_root, "figures")
src_out <- file.path(out_root, "source_data")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(src_out, recursive = TRUE, showWarnings = FALSE)

pal <- c(
  ink = "#17212E",
  muted = "#6F7B8A",
  grid = "#E6ECF2",
  cgp = "#0B4C78",
  rdkit = "#9AA6B2",
  nyan = "#C96A3A",
  fp = "#9AA6B2",
  atom = "#128A7D",
  alert = "#B83F55",
  combo = "#673C8F",
  low = "#E9F1F5",
  high = "#0B4C78"
)

method_cols <- c(
  "CGP-Align" = pal[["cgp"]],
  "RDKit2D" = pal[["rdkit"]],
  "NYAN" = pal[["nyan"]],
  "AtomPair" = pal[["atom"]],
  "Avalon" = "#AAB3BE",
  "Morgan" = "#AAB3BE"
)

alert_cols <- c(
  "CGP + alerts" = pal[["combo"]],
  "CGP-Align latent" = pal[["cgp"]],
  "Expert alerts" = pal[["alert"]]
)

theme_bib <- function(base_size = 7.5, base_family = "Arial") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line = element_line(linewidth = 0.32, colour = pal[["ink"]]),
      axis.ticks = element_line(linewidth = 0.28, colour = pal[["ink"]]),
      axis.title = element_text(size = base_size, colour = pal[["ink"]]),
      axis.text = element_text(size = base_size - 0.35, colour = pal[["ink"]]),
      legend.title = element_text(size = base_size - 0.3, colour = pal[["ink"]]),
      legend.text = element_text(size = base_size - 0.5, colour = pal[["ink"]]),
      plot.title = element_text(size = base_size + 0.8, face = "bold", colour = pal[["ink"]], margin = margin(b = 4)),
      plot.subtitle = element_text(size = base_size - 0.2, colour = pal[["muted"]], margin = margin(b = 3)),
      plot.caption = element_text(size = base_size - 1.0, colour = pal[["muted"]], hjust = 0),
      strip.text = element_text(size = base_size - 0.2, face = "bold", colour = pal[["ink"]]),
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_line(linewidth = 0.25, colour = pal[["grid"]]),
      panel.grid.minor = element_blank(),
      legend.position = "top",
      plot.margin = margin(4, 6, 4, 6)
    )
}

theme_set(theme_bib())

save_pub_r <- function(plot, filename, width_mm = 183, height_mm = 126, dpi = 600) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4
  svglite::svglite(paste0(filename, ".svg"), width = w, height = h)
  print(plot)
  dev.off()
  grDevices::cairo_pdf(paste0(filename, ".pdf"), width = w, height = h, family = "Arial")
  print(plot)
  dev.off()
  ragg::agg_tiff(paste0(filename, ".tiff"), width = w, height = h, units = "in", res = dpi, compression = "lzw")
  print(plot)
  dev.off()
  ragg::agg_png(paste0(filename, ".png"), width = w, height = h, units = "in", res = dpi)
  print(plot)
  dev.off()
}

fmt2 <- function(x) sprintf("%.2f", as.numeric(x))
fmt3 <- function(x) sprintf("%.3f", as.numeric(x))

method_order <- c("CGP-Align", "RDKit2D", "NYAN", "AtomPair", "Avalon", "Morgan")
endpoint_order <- c("SR-ARE", "SR-MMP", "SR-HSE", "SR-p53", "SR-ATAD5")

sr5_mean <- readr::read_csv(file.path(src_out, "figure5a_sr5_representation_mean_metrics.csv"), show_col_types = FALSE) %>%
  mutate(display = factor(display, levels = rev(method_order)))

panel_a_df <- sr5_mean %>%
  select(display, auprc_mean, auroc_mean, auprc_se, auroc_se) %>%
  pivot_longer(cols = c(auprc_mean, auroc_mean), names_to = "metric", values_to = "value") %>%
  mutate(
    se = ifelse(metric == "auprc_mean", auprc_se, auroc_se),
    metric = recode(metric, auprc_mean = "AUPRC", auroc_mean = "AUROC"),
    metric = factor(metric, levels = c("AUPRC", "AUROC")),
    label_pos = ifelse(metric == "AUPRC", 0.62, 0.92)
  )

p_a <- ggplot(panel_a_df, aes(x = display, y = value, fill = display)) +
  geom_col(width = 0.62, colour = NA) +
  geom_errorbar(aes(ymin = pmax(0, value - se), ymax = pmin(1, value + se)), width = 0.18, linewidth = 0.28, colour = pal[["ink"]]) +
  geom_text(aes(y = label_pos, label = fmt3(value)), hjust = 0, size = 2.05, colour = pal[["ink"]]) +
  coord_flip(clip = "off") +
  facet_wrap(~ metric, scales = "free_x") +
  scale_fill_manual(values = method_cols, guide = "none") +
  scale_y_continuous(limits = c(0, 1.06), breaks = c(0, 0.25, 0.50, 0.75), expand = expansion(mult = c(0, 0.03))) +
  labs(title = "Cell-phenotype-related toxicity benchmark", subtitle = "Five phenotype-linked TOXRIC endpoints; fixed ExtraTrees classifier", x = NULL, y = "Mean performance") +
  theme(
    strip.background = element_blank(),
    panel.spacing.x = unit(13, "pt"),
    panel.grid.major.x = element_line(colour = pal[["grid"]])
  )

endpoint_df <- readr::read_csv(file.path(src_out, "figure5b_sr5_endpoint_auprc_heatmap.csv"), show_col_types = FALSE) %>%
  mutate(display = factor(display, levels = rev(method_order)), endpoint = factor(endpoint, levels = endpoint_order))

p_b <- ggplot(endpoint_df, aes(x = endpoint, y = display, fill = auprc)) +
  geom_tile(width = 0.94, height = 0.94, colour = "white", linewidth = 0.35) +
  geom_text(aes(label = fmt2(auprc)), size = 2.1, colour = pal[["ink"]]) +
  scale_fill_gradientn(colours = c(pal[["low"]], "#9ECAE1", pal[["high"]]), name = "AUPRC", limits = c(0.2, 0.78)) +
  labs(title = "Endpoint-level phenotype-related AUPRC", subtitle = "Performance is measured under identical folds and classifier", x = NULL, y = NULL) +
  theme(
    axis.text.x = element_text(angle = 35, hjust = 1),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    legend.position = "right",
    legend.key.height = unit(16, "pt")
  )

scaffold_df <- readr::read_csv(file.path(src_out, "figure5c_sr5_scaffold_split_mean_metrics.csv"), show_col_types = FALSE) %>%
  filter(display %in% method_order) %>%
  mutate(display = factor(display, levels = rev(method_order)))

scaffold_endpoint_df <- readr::read_csv(file.path(src_out, "figure5c_sr5_scaffold_split_endpoint_dots.csv"), show_col_types = FALSE) %>%
  filter(display %in% method_order) %>%
  mutate(
    display = factor(display, levels = rev(method_order)),
    endpoint = factor(endpoint, levels = endpoint_order)
  )

p_c <- ggplot(scaffold_df, aes(x = display, y = mean_auprc, fill = display)) +
  geom_col(width = 0.62, colour = NA) +
  geom_point(
    data = scaffold_endpoint_df,
    aes(x = display, y = auprc_mean),
    inherit.aes = FALSE,
    shape = 21, size = 1.65, stroke = 0.28,
    fill = "white", colour = pal[["ink"]], alpha = 0.88,
    position = position_jitter(width = 0.11, height = 0, seed = 11)
  ) +
  geom_text(aes(y = 0.775, label = fmt3(mean_auprc)), hjust = 1, size = 2.1, colour = pal[["ink"]]) +
  coord_flip(clip = "off") +
  scale_fill_manual(values = method_cols, guide = "none") +
  scale_y_continuous(limits = c(0, 0.78), expand = expansion(mult = c(0, 0.10))) +
  labs(
    title = "Scaffold-disjoint robustness",
    subtitle = "Bars show means; dots show five phenotype-related endpoints",
    x = NULL,
    y = "Mean AUPRC"
  ) +
  theme(plot.margin = margin(5, 10, 5, 6))

alert_method_labels <- c(
  "structure_alert_extratrees" = "Expert alerts",
  "cgp_latent_extratrees" = "CGP-Align latent",
  "cgp_plus_alert_extratrees" = "CGP + alerts"
)

alert_seed_df <- readr::read_csv(file.path(src_out, "figure5d_sr5_expert_alert_complementarity_by_run.csv"), show_col_types = FALSE) %>%
  mutate(method_label = factor(method_label, levels = c("Expert alerts", "CGP-Align latent", "CGP + alerts")))

alert_df <- readr::read_csv(file.path(src_out, "figure5d_expert_alert_complementarity_summary.csv"), show_col_types = FALSE) %>%
  mutate(method_label = factor(method_label, levels = c("Expert alerts", "CGP-Align latent", "CGP + alerts")))

panel_d_df <- alert_df %>%
  select(method_label, mean_auprc, sem_auprc, mean_lift_at_top50, sem_lift_at_top50) %>%
  pivot_longer(cols = c(mean_auprc, mean_lift_at_top50), names_to = "metric", values_to = "value") %>%
  mutate(
    sem = ifelse(metric == "mean_auprc", sem_auprc, sem_lift_at_top50),
    metric = recode(metric, mean_auprc = "AUPRC", mean_lift_at_top50 = "Lift@50"),
    metric = factor(metric, levels = c("AUPRC", "Lift@50"))
  )

p_d <- ggplot(panel_d_df, aes(x = method_label, y = value, fill = method_label)) +
  geom_col(width = 0.58, colour = NA) +
  geom_errorbar(aes(ymin = pmax(0, value - sem), ymax = value + sem), width = 0.16, linewidth = 0.28, colour = pal[["ink"]]) +
  geom_text(aes(label = ifelse(metric == "AUPRC", fmt3(value), fmt2(value))), vjust = -0.45, size = 2.1, colour = pal[["ink"]]) +
  facet_wrap(~ metric, scales = "free_y") +
  scale_fill_manual(values = alert_cols, guide = "none") +
  scale_x_discrete(labels = c(
    "Expert alerts" = "Expert\nalerts",
    "CGP-Align latent" = "CGP\nlatent",
    "CGP + alerts" = "CGP +\nalerts"
  )) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title = "Complementarity to structural alerts", subtitle = "Repeated runs across the same five phenotype-related endpoints", x = NULL, y = NULL) +
  theme(
    axis.text.x = element_text(size = 6.0, angle = 0, hjust = 0.5, vjust = 1, lineheight = 0.84),
    strip.background = element_blank(),
    panel.spacing.x = unit(11, "pt"),
    plot.margin = margin(5, 8, 5, 8)
  )

fig <- (p_a | p_b) / (p_c | p_d) +
  plot_layout(widths = c(0.92, 1.08), heights = c(1.0, 0.92), guides = "keep") +
  plot_annotation(
    tag_levels = "a",
    theme = theme(plot.tag = element_text(size = 9.2, face = "bold", colour = pal[["ink"]]))
  ) & theme(plot.margin = margin(5, 6, 5, 6), plot.title = element_blank(), plot.subtitle = element_blank())

outfile <- file.path(out_dir, "figure5_toxicity_alert_complementarity")
save_pub_r(fig, outfile, width_mm = 183, height_mm = 126, dpi = 600)
message("Wrote: ", outfile, ".svg/.pdf/.tiff/.png")
