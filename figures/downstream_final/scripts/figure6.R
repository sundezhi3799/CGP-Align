suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(patchwork)
  library(scales)
  library(grid)
  library(png)
})

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args_all[grep("^--file=", args_all)][1])
if (is.na(file_arg)) {
  file_arg <- file.path("scripts", "make_figure6_response_profile_retrieval.R")
}
build_dir <- normalizePath(
  file.path(dirname(normalizePath(file_arg, winslash = "/", mustWork = TRUE)), ".."),
  winslash = "/",
  mustWork = TRUE
)
fig_dir <- file.path(build_dir, "figures")
src_dir <- file.path(build_dir, "source_data")
mol_img_dir <- file.path(src_dir, "figure6_molecule_structures")
mol_img_journal_dir <- file.path(src_dir, "figure6_molecule_structures_journal")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(src_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(mol_img_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(mol_img_journal_dir, recursive = TRUE, showWarnings = FALSE)

fig_prefix <- file.path(fig_dir, "figure6_response_profile_retrieval")

method_levels <- c("CGP-Align", "RDKit2D", "NYAN", "Morgan", "AtomPair", "Avalon", "Random")
main_methods <- c("CGP-Align", "RDKit2D", "NYAN", "Morgan", "Random")
method_cols <- c(
  "CGP-Align" = "#154C63",
  "RDKit2D" = "#8E9AA3",
  "NYAN" = "#7B6BAF",
  "Morgan" = "#CB6F4A",
  "AtomPair" = "#2A9D8F",
  "Avalon" = "#A8AEB5",
  "Random" = "#D3D7DB"
)
method_lty <- c(
  "CGP-Align" = "solid",
  "RDKit2D" = "solid",
  "NYAN" = "solid",
  "Morgan" = "solid",
  "AtomPair" = "solid",
  "Avalon" = "solid",
  "Random" = "dotted"
)

base_theme <- theme_classic(base_size = 7.5, base_family = "Arial") +
  theme(
    axis.line = element_line(linewidth = 0.34, colour = "#111111"),
    axis.ticks = element_line(linewidth = 0.34, colour = "#111111"),
    axis.text = element_text(colour = "#111111", size = 6.5),
    axis.title = element_text(colour = "#111111", size = 7.2),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 6.2),
    legend.key.width = unit(10, "pt"),
    strip.background = element_blank(),
    strip.text = element_text(size = 6.3, face = "bold", colour = "#111111"),
    panel.grid.major.y = element_line(linewidth = 0.22, colour = "#E7EAED"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin = margin(4, 5, 4, 5)
  )
theme_set(base_theme)

read_source <- function(x) {
  read_csv(file.path(src_dir, x), show_col_types = FALSE)
}

wrap_text <- function(x, width = 24) {
  vapply(x, function(z) paste(strwrap(z, width = width), collapse = "\n"), character(1))
}

circle_df <- function(cx, cy, r, n = 160) {
  t <- seq(0, 2 * pi, length.out = n)
  data.frame(x = cx + r * cos(t), y = cy + r * sin(t))
}

safe_filename <- function(x) {
  gsub("[^A-Za-z0-9_-]+", "_", x)
}

pubchem_structure_png <- function(inchikey, name, image_size = "640x420") {
  out <- file.path(mol_img_dir, paste0(safe_filename(name), "_", substr(inchikey, 1, 14), ".png"))
  if (!file.exists(out) || is.na(file.info(out)$size) || file.info(out)$size < 500) {
    url <- sprintf(
      "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/%s/PNG?image_size=%s",
      URLencode(inchikey, reserved = TRUE),
      image_size
    )
    ok <- tryCatch({
      suppressWarnings(utils::download.file(url, out, mode = "wb", quiet = TRUE))
      TRUE
    }, error = function(e) FALSE)
    if (!ok && file.exists(out)) {
      unlink(out)
    }
  }
  if (file.exists(out) && !is.na(file.info(out)$size) && file.info(out)$size >= 500) {
    return(out)
  }
  NA_character_
}

journal_structure_png <- function(name) {
  out <- file.path(mol_img_journal_dir, paste0(safe_filename(name), "_journal.png"))
  if (file.exists(out) && !is.na(file.info(out)$size) && file.info(out)$size >= 500) {
    return(out)
  }
  NA_character_
}

compound_structure_png <- function(inchikey, name) {
  out <- journal_structure_png(name)
  if (!is.na(out)) {
    return(out)
  }
  pubchem_structure_png(inchikey, name)
}

structure_grob <- function(path) {
  if (is.na(path) || !file.exists(path)) {
    return(nullGrob())
  }
  img <- png::readPNG(path)
  rgb <- img[, , seq_len(min(3, dim(img)[3])), drop = FALSE]
  non_bg <- apply(rgb, c(1, 2), function(v) any(v < 0.97))
  if (any(non_bg)) {
    rows <- which(rowSums(non_bg) > 0)
    cols <- which(colSums(non_bg) > 0)
    pad_r <- max(2, round(diff(range(rows)) * 0.08))
    pad_c <- max(2, round(diff(range(cols)) * 0.08))
    rows <- seq(max(1, min(rows) - pad_r), min(dim(img)[1], max(rows) + pad_r))
    cols <- seq(max(1, min(cols) - pad_c), min(dim(img)[2], max(cols) + pad_c))
    img <- img[rows, cols, , drop = FALSE]
  }
  rasterGrob(img, interpolate = TRUE)
}

cluster_points <- function(cx, cy, n, spread, label, colour) {
  # Deterministic pseudo-clusters for schematic drawing only.
  i <- seq_len(n)
  angle <- (i * 2.399963) %% (2 * pi)
  radius <- spread * sqrt((i %% 31 + 1) / 31)
  data.frame(
    x = cx + radius * cos(angle) + spread * 0.15 * sin(i),
    y = cy + radius * sin(angle) + spread * 0.15 * cos(i * 0.7),
    label = label,
    colour = colour
  )
}

functional_hits <- read_source("figure6_panel_b_functional_neighbour_enrichment_source.csv") %>%
  mutate(method = factor(method, levels = method_levels))
query_pairs <- read_source("figure6_panel_c_cgp_vs_rdkit2d_source.csv")
pan <- read_source("figure6_panel_d_pancancer_source.csv") %>%
  mutate(method = factor(method, levels = method_levels))
gene_mech <- read_source("figure6_panel_e_gene_mechanism_summary_source.csv") %>%
  mutate(method = factor(method, levels = method_levels))
category_summary <- read_source("figure6_panel_e_mechanism_category_summary_source.csv") %>%
  mutate(method = factor(method, levels = method_levels))
case_hc <- read_source("figure6_panel_f_high_confidence_cases_source.csv")
case_mech <- read_source("figure6_panel_f_mechanism_cases_source.csv")

## Panel A: phenotype-driven neighbour discovery workflow.
latent_circle <- circle_df(0.43, 0.56, 0.230)
latent_pts <- bind_rows(
  cluster_points(0.335, 0.63, 26, 0.040, "query compound", "#154C63"),
  cluster_points(0.505, 0.63, 23, 0.037, "retrieved compound", "#7B6BAF"),
  cluster_points(0.420, 0.505, 18, 0.034, "profile anchor", "#9AA4AE"),
  cluster_points(0.495, 0.435, 12, 0.027, "gene neighbours", "#6E9E59")
)
evidence_boxes <- tibble::tibble(
  x0 = c(0.700, 0.700, 0.700),
  y0 = c(0.690, 0.525, 0.360),
  x1 = c(0.980, 0.980, 0.980),
  y1 = c(0.820, 0.655, 0.490),
  label = c("Low chemical\nsimilarity", "Similar response\nprofile", "Gene-program\nsupport")
)
panel_a <- ggplot() +
  geom_rect(
    aes(xmin = 0.020, xmax = 0.195, ymin = 0.465, ymax = 0.700),
    fill = "#EEF3F4", colour = "#95A3AA", linewidth = 0.35
  ) +
  geom_text(
    aes(x = 0.108, y = 0.660, label = "Query\ncompound"),
    size = 1.60, fontface = "bold", family = "Arial", lineheight = 0.84
  ) +
  geom_point(
    data = data.frame(x = c(0.065, 0.095, 0.130, 0.118, 0.086), y = c(0.555, 0.590, 0.565, 0.520, 0.505)),
    aes(x, y), size = 2.70, shape = 21, fill = "#9DD4DA", colour = "#154C63", stroke = 0.32
  ) +
  geom_segment(
    aes(x = 0.200, y = 0.585, xend = 0.258, yend = 0.585),
    linewidth = 0.42, colour = "#334E5C", arrow = arrow(length = unit(3.2, "pt"), type = "closed")
  ) +
  geom_path(data = latent_circle, aes(x, y), colour = "#8CB4BC", linewidth = 0.42) +
  geom_segment(
    data = data.frame(
      x = c(0.335, 0.500, 0.420),
      y = c(0.630, 0.630, 0.505),
      xend = c(0.420, 0.420, 0.495),
      yend = c(0.505, 0.505, 0.435)
    ),
    aes(x = x, y = y, xend = xend, yend = yend),
    linewidth = 0.28, colour = "#A4AFB8", linetype = "dashed"
  ) +
  geom_point(
    data = latent_pts,
    aes(x, y, fill = colour),
    shape = 21, colour = "white", stroke = 0.12, size = 1.25, alpha = 0.96
  ) +
  scale_fill_identity() +
  geom_text(
    aes(x = 0.43, y = 0.825, label = "Phenotype-anchored\nlatent space"),
    size = 2.00, fontface = "bold", family = "Arial", lineheight = 0.86
  ) +
  geom_text(
    aes(x = 0.325, y = 0.710, label = "query"),
    size = 1.62, fontface = "bold", family = "Arial", colour = "#154C63"
  ) +
  geom_text(
    aes(x = 0.522, y = 0.710, label = "neighbour"),
    size = 1.62, fontface = "bold", family = "Arial", colour = "#7B6BAF"
  ) +
  geom_text(
    aes(x = 0.370, y = 0.410, label = "profile\nanchor"),
    size = 1.38, family = "Arial", lineheight = 0.84, colour = "#58636E"
  ) +
  geom_text(
    aes(x = 0.555, y = 0.382, label = "gene\nsupport"),
    size = 1.38, family = "Arial", lineheight = 0.84, colour = "#4B743E"
  ) +
  geom_segment(
    aes(x = 0.575, y = 0.59, xend = 0.690, yend = 0.59),
    linewidth = 0.42, colour = "#334E5C", arrow = arrow(length = unit(3.2, "pt"), type = "closed")
  ) +
  geom_rect(
    data = evidence_boxes,
    aes(xmin = x0, xmax = x1, ymin = y0, ymax = y1),
    fill = "#E8F5F1", colour = "#A6B6B2", linewidth = 0.32
  ) +
  geom_text(
    data = evidence_boxes,
    aes(x = (x0 + x1) / 2, y = (y0 + y1) / 2, label = label),
    size = 1.70, family = "Arial", colour = "#14212B", lineheight = 0.84
  ) +
  geom_rect(
    aes(xmin = 0.710, xmax = 0.970, ymin = 0.175, ymax = 0.300),
    fill = "#F2F6F7", colour = "#A6B6B2", linewidth = 0.32
  ) +
  geom_text(
    aes(x = 0.840, y = 0.238, label = "Functional\nneighbour"),
    size = 1.82, fontface = "bold", family = "Arial", lineheight = 0.84
  ) +
  geom_segment(
    aes(x = 0.840, y = 0.350, xend = 0.840, yend = 0.308),
    linewidth = 0.42, colour = "#334E5C", arrow = arrow(length = unit(3.2, "pt"), type = "closed")
  ) +
  coord_cartesian(xlim = c(0.02, 1.00), ylim = c(0.15, 0.85), expand = FALSE) +
  theme_void(base_family = "Arial") +
  theme(plot.margin = margin(4, 5, 4, 5))

## Panel B: enrichment under increasingly strict structural filters.
panel_b_data <- functional_hits %>%
  filter(topk == 50, corr_threshold == 0.30, method %in% main_methods) %>%
  select(method, low_tanimoto, hit_yield_per_1000_ranked, unique_pair_hits, query_hit_rate) %>%
  group_by(low_tanimoto) %>%
  mutate(random_yield = hit_yield_per_1000_ranked[method == "Random"][1]) %>%
  ungroup() %>%
  mutate(
    enrichment = hit_yield_per_1000_ranked / random_yield,
    method = factor(method, levels = method_levels)
  )
panel_b_labels <- panel_b_data %>%
  filter(low_tanimoto == min(low_tanimoto), method %in% c("CGP-Align", "RDKit2D", "NYAN", "Morgan")) %>%
  mutate(label = as.character(method))
panel_b <- ggplot(panel_b_data, aes(low_tanimoto, enrichment, colour = method, linetype = method, group = method)) +
  geom_hline(yintercept = 1, linewidth = 0.28, colour = "#B7C0C7", linetype = "dotted") +
  geom_line(linewidth = 0.66, alpha = 0.98) +
  geom_point(size = 1.8, stroke = 0.2, alpha = 0.98) +
  scale_colour_manual(values = method_cols, breaks = main_methods, labels = function(x) ifelse(x == "RDKit2D", "RDKit-Morgan", x), drop = TRUE) +
  scale_linetype_manual(values = method_lty, breaks = main_methods, labels = function(x) ifelse(x == "RDKit2D", "RDKit-Morgan", x), drop = TRUE) +
  scale_x_reverse(breaks = c(0.30, 0.25, 0.20, 0.15), limits = c(0.31, 0.12)) +
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.12))) +
  labs(
    x = "Max. Morgan Tanimoto",
    y = "Enrichment over random"
  ) +
  guides(colour = guide_legend(nrow = 1, override.aes = list(linewidth = 0.8, size = 2.0)))

## Panel C: query-level paired improvement.
query_stat <- tibble::tibble(
  n_query = sum(query_pairs$n_query),
  win_n = sum(query_pairs$n_query[query_pairs$cgp_better], na.rm = TRUE),
  tie_n = sum(query_pairs$n_query[query_pairs$same], na.rm = TRUE),
  mean_delta = weighted.mean(query_pairs$`CGP-Align` - query_pairs$RDKit2D, query_pairs$n_query)
)
d_axis_max <- max(c(query_pairs$`CGP-Align`, query_pairs$RDKit2D), na.rm = TRUE) + 0.04
d_axis_max <- min(0.70, max(0.45, d_axis_max))
panel_c <- ggplot(query_pairs, aes(RDKit2D, `CGP-Align`)) +
  geom_abline(slope = 1, intercept = 0, linewidth = 0.35, colour = "#B8C0C8", linetype = "dashed") +
  geom_point(
    aes(size = n_query, fill = cgp_better),
    shape = 21, colour = "white", stroke = 0.22, alpha = 0.90, show.legend = FALSE
  ) +
  scale_fill_manual(values = c("TRUE" = "#154C63", "FALSE" = "#A8AEB5"), guide = "none") +
  scale_size_area(max_size = 4.8, guide = "none") +
  coord_cartesian(xlim = c(0, d_axis_max), ylim = c(0, d_axis_max), expand = FALSE) +
  labs(
    x = "RDKit-Morgan fraction",
    y = "CGP-Align fraction"
  ) +
  theme(
    panel.grid.major.x = element_line(linewidth = 0.22, colour = "#E7EAED"),
    panel.grid.major.y = element_line(linewidth = 0.22, colour = "#E7EAED")
  )

## Panel D: pan-cancer response-profile retention.
panel_d_data <- pan %>%
  filter(grepl("^low_morgan_lt_", setting), method %in% main_methods) %>%
  mutate(
    low_tanimoto = as.numeric(sub("low_morgan_lt_", "", setting)),
    method = factor(method, levels = method_levels)
  )
panel_d <- ggplot(panel_d_data, aes(low_tanimoto, retention50, colour = method, linetype = method, group = method)) +
  geom_hline(yintercept = 0, linewidth = 0.28, colour = "#D0D5DA") +
  geom_line(linewidth = 0.66) +
  geom_point(size = 1.6, alpha = 0.98) +
  annotate(
    "text",
    x = 0.303, y = max(panel_d_data$retention50, na.rm = TRUE) * 0.925,
    label = "578 cell lines",
    size = 1.95, hjust = 0, family = "Arial", colour = "#58636E"
  ) +
  scale_colour_manual(values = method_cols, breaks = main_methods, labels = function(x) ifelse(x == "RDKit2D", "RDKit-Morgan", x), drop = TRUE) +
  scale_linetype_manual(values = method_lty, breaks = main_methods, labels = function(x) ifelse(x == "RDKit2D", "RDKit-Morgan", x), drop = TRUE) +
  scale_x_reverse(breaks = c(0.30, 0.25, 0.20, 0.15), limits = c(0.31, 0.14)) +
  scale_y_continuous(labels = label_number(accuracy = 0.01), expand = expansion(mult = c(0.04, 0.10))) +
  labs(
    x = "Max. Morgan Tanimoto",
    y = "Oracle-normalized retention@50"
  ) +
  guides(colour = "none", linetype = "none")

## Shared case table used for Panels E and F.
case_join <- case_hc %>%
  select(
    query_name, retrieved_name, query_moa, retrieved_moa,
    query_inchikey, retrieved_inchikey,
    prism_response_corr, morgan_tanimoto
  ) %>%
  inner_join(
    case_mech %>%
      select(
        query_name, retrieved_name, shared_top_gene_count,
        dominant_categories, shared_gene_symbols_top20
      ),
    by = c("query_name", "retrieved_name")
  ) %>%
  mutate(
    pair = paste(query_name, retrieved_name, sep = " -> "),
    pair_label = wrap_text(pair, 22),
    mechanism = sub(";.*$", "", dominant_categories),
    mechanism = sub(":.*$", "", mechanism)
  )

## Panel E: compound-gene-program convergence.
network_case <- read_source("figure6_network_case.csv") %>% mutate(query_label=query_name,retrieved_label=wrap_text(retrieved_name,14))
query_structure <- compound_structure_png(network_case$query_inchikey[1],network_case$query_name[1])
retrieved_structure <- compound_structure_png(network_case$retrieved_inchikey[1],network_case$retrieved_name[1])
gene_nodes <- read_source("figure6_network_nodes.csv") %>% mutate(type="Gene",x=0.555,y=seq(0.76,0.32,length.out=n()))
program_nodes <- tibble::tibble(program=unique(gene_nodes$program),type="Program",x=0.86,y=seq(0.63,0.43,length.out=n_distinct(gene_nodes$program))) %>% mutate(name=wrap_text(program,14))
node_df <- bind_rows(gene_nodes,program_nodes)
program_cols <- setNames(c("#154C63","#CB6F4A")[seq_len(nrow(program_nodes))],program_nodes$program)
edge_compound_gene <- bind_rows(
  gene_nodes %>% transmute(xend = x, yend = y, x = 0.425, y = 0.70),
  gene_nodes %>% transmute(xend = x, yend = y, x = 0.425, y = 0.42)
)
edge_gene_program <- gene_nodes %>% left_join(program_nodes %>% select(program,yend=y),by="program") %>% transmute(x,y,xend=0.78,yend,program)
panel_e_network <- ggplot() +
  geom_rect(
    aes(xmin = 0.025, xmax = 0.420, ymin = 0.600, ymax = 0.855),
    fill = "#FFFFFF", colour = "#C5CDD3", linewidth = 0.28
  ) +
  geom_rect(
    aes(xmin = 0.025, xmax = 0.420, ymin = 0.270, ymax = 0.575),
    fill = "#FFFFFF", colour = "#C5CDD3", linewidth = 0.28
  ) +
  annotation_custom(
    structure_grob(query_structure),
    xmin = 0.040, xmax = 0.405, ymin = 0.660, ymax = 0.842
  ) +
  annotation_custom(
    structure_grob(retrieved_structure),
    xmin = 0.040, xmax = 0.405, ymin = 0.390, ymax = 0.562
  ) +
  geom_segment(
    data = edge_compound_gene,
    aes(x = x, y = y, xend = xend, yend = yend),
    linewidth = 0.25, colour = "#AEB8C2", alpha = 0.70
  ) +
  geom_segment(
    data = edge_gene_program,
    aes(x = x, y = y, xend = xend, yend = yend, colour = program),
    linewidth = 0.34, alpha = 0.85
  ) +
  geom_point(
    data = node_df,
    aes(x, y, fill = type),
    shape = 21, size = ifelse(node_df$type == "Gene", 2.25, 4.1),
    colour = "white", stroke = 0.35
  ) +
  geom_text(
    data = node_df %>% filter(type != "Gene"),
    aes(x, y - 0.090, label = wrap_text(name, 14)),
    family = "Arial", size = 1.75, lineheight = 0.82
  ) +
  annotate("text", x = 0.215, y = 0.618, label = network_case$query_label[1], size = 1.78, fontface = "bold", family = "Arial", lineheight = 0.82) +
  annotate("text", x = 0.215, y = 0.306, label = network_case$retrieved_label[1], size = 1.44, fontface = "bold", family = "Arial", lineheight = 0.74) +
  geom_label(
    data = gene_nodes,
    aes(x + 0.030, y, label = name),
    family = "Arial", size = 1.52, hjust = 0, colour = "#111111",
    fill = "#FFFFFF", alpha = 0.92, linewidth = 0,
    label.padding = unit(0.35, "pt")
  ) +
  annotate("text", x = 0.035, y = 0.90, label = "compound pair", size = 1.75, fontface = "bold", family = "Arial", hjust = 0) +
  annotate("text", x = 0.555, y = 0.90, label = "shared genes", size = 1.75, fontface = "bold", family = "Arial", hjust = 0) +
  annotate("text", x = 0.795, y = 0.90, label = "programs", size = 1.75, fontface = "bold", family = "Arial", hjust = 0) +
  scale_fill_manual(values = c("Query" = "#154C63", "Retrieved" = "#7B6BAF", "Gene" = "#C9D2D6", "Program" = "#E9F2EF"), guide = "none") +
  scale_colour_manual(values = program_cols, guide = "none") +
  coord_cartesian(xlim = c(0.02, 0.98), ylim = c(0.22, 0.94), expand = FALSE) +
  theme_void(base_family = "Arial") +
  theme(plot.margin = margin(4, 2, 4, 4))

program_data <- category_summary %>%
  filter(method == "CGP-Align") %>%
  arrange(desc(num_pairs_with_category)) %>%
  slice_head(n = 6) %>%
  mutate(
    program_label = factor(wrap_text(mechanism_category, 16), levels = rev(wrap_text(mechanism_category, 16))),
    pair_percent = 100 * as.numeric(pair_fraction),
    total_shared_gene_category_hits = as.numeric(total_shared_gene_category_hits)
  )
panel_e_program <- ggplot(program_data, aes(pair_percent, program_label)) +
  geom_segment(aes(x = 0, xend = pair_percent, y = program_label, yend = program_label), colour = "#CDD4DA", linewidth = 0.35) +
  geom_point(aes(size = total_shared_gene_category_hits, fill = pair_percent), shape = 21, colour = "white", stroke = 0.25) +
  geom_text(aes(label = sprintf("%.1f%%", pair_percent)), nudge_x = 2.4, size = 1.65, family = "Arial", hjust = 0) +
  scale_fill_gradient(low = "#DCE8EA", high = "#154C63", guide = "none") +
  scale_size_area(max_size = 4.4, guide = "none") +
  scale_x_continuous(limits = c(0, max(program_data$pair_percent) + 8), expand = c(0, 0)) +
  labs(x = "Pairs with module (%)", y = NULL) +
  theme_classic(base_size = 7.0, base_family = "Arial") +
  theme(
    axis.text.y = element_text(size = 5.7, colour = "#111111"),
    axis.text.x = element_text(size = 5.5, colour = "#111111"),
    axis.title.x = element_text(size = 6.1),
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid.major.x = element_line(linewidth = 0.20, colour = "#E7EAED"),
    plot.margin = margin(4, 4, 4, 1)
  )
panel_e <- wrap_elements(panel_e_network + panel_e_program + plot_layout(widths = c(1.76, 0.88)))

## Panel F: discovery landscape for representative cases.
case_landscape <- case_hc %>%
  left_join(
    case_mech %>%
      select(query_name, retrieved_name, shared_top_gene_count, dominant_categories, shared_gene_symbols_top20),
    by = c("query_name", "retrieved_name")
  ) %>%
  mutate(
    pair = paste(query_name, retrieved_name, sep = " -> "),
    has_mechanism = !is.na(shared_top_gene_count),
    mechanism = sub(";.*$", "", dominant_categories),
    mechanism = sub(":.*$", "", mechanism),
    mechanism_group = case_when(
      grepl("Cell cycle", mechanism) ~ "Cell cycle/mitosis",
      grepl("Lipid", mechanism) ~ "Lipid/transport state",
      TRUE ~ "Other annotated program"
    )
  )
write_csv(case_landscape, file.path(src_dir, "figure6_panel_f_discovery_landscape_source.csv"))

case_labels <- case_landscape %>% arrange(desc(shared_top_gene_count),desc(prism_response_corr)) %>% slice_head(n=3) %>% mutate(label=sprintf("%s -> %s\n%d shared genes",query_name,retrieved_name,as.integer(shared_top_gene_count)))
landscape_ymax <- max(case_landscape$prism_response_corr,na.rm=TRUE)+0.04
panel_f <- ggplot() +
  annotate("rect", xmin = 0, xmax = 0.20, ymin = 0.30, ymax = landscape_ymax, fill = "#F8FAFA", colour = NA) +
  annotate("segment", x = 0.20, xend = 0.20, y = 0.30, yend = landscape_ymax, linewidth = 0.24, linetype = "dashed", colour = "#AAB4BC") +
  annotate("segment", x = 0.04, xend = 0.20, y = 0.30, yend = 0.30, linewidth = 0.24, linetype = "dashed", colour = "#AAB4BC") +
  geom_point(
    data = case_landscape %>% filter(!has_mechanism),
    aes(x = morgan_tanimoto, y = prism_response_corr),
    colour = "#B8C1C8", size = 1.45, alpha = 0.62
  ) +
  geom_point(
    data = case_landscape %>% filter(has_mechanism),
    aes(x = morgan_tanimoto, y = prism_response_corr, fill = mechanism_group, size = shared_top_gene_count),
    shape = 21, colour = "white", stroke = 0.38, alpha = 0.96
  ) +
  ggrepel::geom_label_repel(data=case_labels,aes(x=morgan_tanimoto,y=prism_response_corr,label=label),seed=41,size=1.65,lineheight=.9,box.padding=.8,point.padding=.5,max.overlaps=Inf,min.segment.length=0,segment.colour="#8C98A3",fill="white",linewidth=0) +
  scale_fill_manual(
    values = c(
      "Cell cycle/mitosis" = "#154C63",
      "Lipid/transport state" = "#CB6F4A",
      "Other annotated program" = "#7B6BAF"
    ),
    guide = "none"
  ) +
  scale_size_area(max_size = 5.8, guide = "none") +
  scale_x_continuous(limits = c(0, 0.205), breaks = c(0.05, 0.10, 0.15, 0.20), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0.28, landscape_ymax), breaks = c(0.30, 0.40, 0.50, 0.60), expand = c(0, 0)) +
  labs(x = "Chemical similarity (Morgan Tanimoto)", y = "PRISM response correlation") +
  theme_classic(base_size = 7.2, base_family = "Arial") +
  theme(
    legend.position = "none",
    axis.title = element_text(size = 7.3),
    axis.text = element_text(size = 6.6, colour = "#111111"),
    panel.grid.major = element_line(linewidth = 0.20, colour = "#E7EAED"),
    plot.margin = margin(4, 5, 4, 5)
  )

design <- "
AABBCC
DDEEEE
FFFFFF
"
combined <- panel_a + panel_b + panel_c + panel_d + panel_e + panel_f +
  plot_layout(design = design, heights = c(1, 1.08, 0.90), guides = "collect") +
  plot_annotation(tag_levels = "a") &
  theme(
    plot.tag = element_text(size = 8.8, face = "bold", family = "Arial"),
    legend.position = "bottom"
  )

ggsave(paste0(fig_prefix, ".pdf"), combined, width = 183, height = 180, units = "mm", device = cairo_pdf)
ggsave(paste0(fig_prefix, ".svg"), combined, width = 183, height = 180, units = "mm")
ggsave(paste0(fig_prefix, ".tiff"), combined, width = 183, height = 180, units = "mm", dpi = 600, compression = "lzw")
ggsave(paste0(fig_prefix, ".png"), combined, width = 183, height = 180, units = "mm", dpi = 600)

manifest <- c(
  "{",
  sprintf('  "figure": "Figure 6",'),
  sprintf('  "title": "Phenotype-anchored representations reveal mechanism-related compound neighbours beyond chemical similarity",'),
  sprintf('  "created_at": "%s",', format(Sys.time(), "%Y-%m-%d %H:%M:%S %z")),
  sprintf('  "script": "figures/downstream_final/scripts/figure6.R",'),
  '  "panels": {',
  '    "a": "Phenotype-driven functional-neighbour discovery workflow",',
  '    "b": "Functional-neighbour enrichment under structural dissimilarity constraints",',
  '    "c": "Query-level paired improvement over RDKit-Morgan (Morgan 2048 plus 11 RDKit descriptors)",',
  '    "d": "Pan-cancer PRISM response retention under structural filters",',
  '    "e": "Compound-gene-program convergence",',
  '    "f": "Discovery landscape of mechanism-annotated neighbours"',
  '  }',
  "}"
)
writeLines(manifest, file.path(src_dir, "figure6_manifest.json"), useBytes = TRUE)

message("Saved Figure 6 to: ", fig_prefix, ".{pdf,svg,tiff,png}")
quit(save = "no", status = 0, runLast = FALSE)
