suppressPackageStartupMessages({
 library(ggplot2); library(dplyr); library(tidyr); library(readr); library(patchwork); library(scales)
})
a <- commandArgs(FALSE); f <- sub("^--file=", "", a[grep("^--file=", a)][1])
root <- dirname(normalizePath(f, winslash="/", mustWork=TRUE))
source_dir <- file.path(root, "source_data")
args <- commandArgs(TRUE)
figure_dir <- if(length(args)) args[1] else file.path(root, "../../outputs/supplementary_strict")
dir.create(figure_dir, recursive=TRUE, showWarnings=FALSE)
set.seed(1701)
sampled <- read_csv(file.path(source_dir, "figureS4_sampled.csv"), show_col_types = FALSE)
full <- read_csv(file.path(source_dir, "figureS4_full.csv"), show_col_types = FALSE)
summary_df <- read_csv(file.path(source_dir, "branch_summary.csv"), show_col_types = FALSE) %>% filter(metric %in% c("Mean Top-10", "HMean Top-10")) %>% rename(independent_run = seed)

direction_levels <- c("C-P", "P-C", "G-P", "P-G")
pal_dir <- c("C-P" = "#102A43", "P-C" = "#2F80ED", "G-P" = "#009E73", "P-G" = "#D55E00")
pal_branch <- c("compound-profile" = "#2F80ED", "gene-profile" = "#009E73")

sampled_summary <- sampled %>%
  mutate(
    short = factor(short, levels = direction_levels),
    metric = factor(metric, levels = c("Top1", "Top10")),
    value_pct = 100 * value
  ) %>%
  group_by(short, direction, branch, metric) %>%
  summarise(mean = mean(value_pct), sd = sd(value_pct), .groups = "drop")

full_summary <- full %>%
  filter(metric %in% c("Recall@1", "Recall@5", "Recall@10", "MRR")) %>%
  mutate(
    short = factor(short, levels = direction_levels),
    metric = factor(metric, levels = c("Recall@1", "Recall@5", "Recall@10", "MRR")),
    value_pct = 100 * value
  ) %>%
  group_by(short, direction, branch, metric) %>%
  summarise(mean = mean(value_pct), sd = sd(value_pct), .groups = "drop")

summary_plot <- summary_df %>%
  mutate(
    metric = factor(metric, levels = c("Mean Top-10", "HMean Top-10")),
    value_pct = 100 * value
  ) %>%
  group_by(metric) %>%
  summarise(mean = mean(value_pct), sd = sd(value_pct), .groups = "drop")

run_summary <- summary_df %>%
  mutate(metric = factor(metric, levels = c("Mean Top-10", "HMean Top-10")), value_pct = 100 * value)

random_run_hmean <- summary_df %>% filter(metric == "HMean Top-10") %>% transmute(independent_run, value = random)

observed_run_hmean <- summary_df %>%
  filter(metric == "HMean Top-10") %>%
  transmute(independent_run, value)

random_df <- tibble(
  metric = factor(c("Random-ranking HMean", "CGP-Align HMean"), levels = c("Random-ranking HMean", "CGP-Align HMean")),
  value = c(100 * mean(random_run_hmean$value), 100 * mean(observed_run_hmean$value)),
  sd = c(100 * sd(random_run_hmean$value), 100 * sd(observed_run_hmean$value))
)

theme_set(
  theme_classic(base_size = 8.4, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "#222222"),
      axis.ticks = element_line(linewidth = 0.32, colour = "#222222"),
      axis.text = element_text(colour = "#222222"),
      axis.title = element_text(colour = "#111111", size = 8.4),
      plot.title = element_text(face = "bold", size = 9.3, colour = "#111111", margin = margin(b = 4)),
      plot.subtitle = element_text(size = 7.4, colour = "#52616F", margin = margin(b = 4)),
      legend.title = element_blank(),
      legend.text = element_text(size = 7.1),
      plot.margin = margin(7, 9, 7, 9),
      panel.grid.major.y = element_line(linewidth = 0.22, colour = "#E7ECF2"),
      panel.grid.major.x = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 8.2)
    )
)

heat_theme <- theme_minimal(base_size = 8.4, base_family = "Arial") +
  theme(
    axis.text = element_text(colour = "#111111"),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    plot.title = element_text(face = "bold", size = 9.3, colour = "#111111", margin = margin(b = 4)),
    plot.subtitle = element_text(size = 7.4, colour = "#52616F", margin = margin(b = 4)),
    legend.position = "right",
    legend.title = element_text(size = 7.1, face = "bold"),
    legend.text = element_text(size = 6.8),
    plot.margin = margin(7, 9, 7, 9)
  )

p_a <- sampled_summary %>%
  mutate(text_col = if_else(mean >= 45, "white", "#111111")) %>%
  ggplot(aes(metric, short, fill = mean)) +
  geom_tile(width = 0.86, height = 0.80, colour = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.1f", mean), colour = text_col), size = 2.85, family = "Arial") +
  scale_colour_identity() +
  scale_fill_gradientn(
    colours = c("#F4F7FA", "#B9D8E8", "#5CA4CB", "#1E5A91", "#102A43"),
    limits = c(0, 90),
    breaks = c(0, 30, 60, 90),
    name = "%"
  ) +
  labs(title = "Sampled retrieval metrics", subtitle = "Figure 3-matched independent runs; candidate setting 1:100") +
  heat_theme

p_b <- full_summary %>%
  mutate(text_col = if_else(mean >= 4.5, "white", "#111111")) %>%
  ggplot(aes(metric, short, fill = mean)) +
  geom_tile(width = 0.86, height = 0.80, colour = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.1f", mean), colour = text_col), size = 2.65, family = "Arial") +
  scale_colour_identity() +
  scale_fill_gradientn(
    colours = c("#F7FBFF", "#D8ECF7", "#9BC9E2", "#4D95BF", "#102A43"),
    limits = c(0, 9.5),
    breaks = c(0, 3, 6, 9),
    name = "%"
  ) +
  labs(title = "Full-gallery retrieval metrics", subtitle = "All held-out candidates; no sampled-decoy restriction") +
  heat_theme +
  theme(axis.text.x = element_text(angle = 22, hjust = 1))

p_c <- ggplot(sampled_summary, aes(short, mean, colour = short)) +
  geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.16, linewidth = 0.42) +
  geom_point(size = 2.7) +
  geom_point(
    data = sampled %>% mutate(short = factor(short, levels = direction_levels), value_pct = 100 * value),
    aes(short, value_pct, colour = short),
    inherit.aes = FALSE,
    size = 1.35,
    alpha = 0.48,
    position = position_jitter(width = 0.05, height = 0, seed = 1701)
  ) +
  facet_wrap(~metric, nrow = 1, scales = "free_y") +
  scale_colour_manual(values = pal_dir) +
  scale_y_continuous(labels = label_number(suffix = "%"), expand = expansion(mult = c(0.08, 0.14))) +
  labs(title = "Run-level sampled retrieval", subtitle = "Large points show three-run mean", x = NULL, y = "Retrieval") +
  theme(legend.position = "none")

p_d <- ggplot() +
  geom_col(data = random_df, aes(metric, value, fill = metric), width = 0.58) +
  geom_errorbar(
    data = random_df %>% filter(metric == "CGP-Align HMean"),
    aes(metric, ymin = value - sd, ymax = value + sd),
    width = 0.16,
    linewidth = 0.42
  ) +
  geom_point(
    data = run_summary %>% filter(metric == "HMean Top-10"),
    aes(x = factor("CGP-Align HMean", levels = levels(random_df$metric)), y = value_pct),
    size = 1.45,
    alpha = 0.55,
    position = position_jitter(width = 0.05, height = 0, seed = 1701),
    colour = "#111111"
  ) +
  geom_text(data = random_df, aes(metric, value + sd + 1, label = sprintf("%.1f%%", value)), vjust = -0.45, size = 2.75, family = "Arial") +
  scale_fill_manual(values = c("Random-ranking HMean" = "#D9E1E8", "CGP-Align HMean" = "#102A43")) +
  scale_y_continuous(limits = c(0, 62), breaks = seq(0, 60, 15), labels = label_number(suffix = "%")) +
  labs(
    title = "Retrieval exceeds random-ranking expectation",
    subtitle = "Null adjusts for multiple positive profiles per entity",
    x = NULL,
    y = "Top-10 retrieval"
  ) +
  theme(legend.position = "none", axis.text.x = element_text(size = 7.4))

fig <- (p_a | p_b) / (p_c | p_d) +
  plot_layout(heights = c(1.05, 1), widths = c(1, 1)) +
  plot_annotation(tag_levels = "a") &
  theme(
    plot.margin = margin(7, 9, 7, 9),
    plot.tag = element_text(size = 13, face = "bold", colour = "#111111"),
    plot.tag.position = c(0, 1),
    plot.title = element_blank(),
    plot.subtitle = element_blank()
  )

out_base <- file.path(figure_dir, "figureS4")
width_mm <- 183
height_mm <- 132
w <- width_mm / 25.4
h <- height_mm / 25.4

ggsave(paste0(out_base, ".png"), fig, width = w, height = h, dpi = 600, bg = "white", device = ragg::agg_png)
ggsave(paste0(out_base, ".tiff"), fig, width = w, height = h, dpi = 600, bg = "white", device = ragg::agg_tiff, compression = "lzw")
ggsave(paste0(out_base, ".pdf"), fig, width = w, height = h, bg = "white", device = cairo_pdf, family = "Arial")
ggsave(paste0(out_base, ".svg"), fig, width = w, height = h, bg = "white", device = svglite::svglite)

message("Wrote: ", out_base)

writeLines(capture.output(sessionInfo()), file.path(figure_dir, "sessionInfo_S4.txt"))
if(length(warnings())) print(warnings())
