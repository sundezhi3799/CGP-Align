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
df <- read_csv(file.path(source_dir, "figureS5_controls.csv"), show_col_types = FALSE)

direction_levels <- c("C-P", "P-C", "G-P", "P-G")
pal <- c("Observed" = "#102A43", "Random-ranking" = "#D8E0E8")
pal_dir <- c("C-P" = "#102A43", "P-C" = "#2F80ED", "G-P" = "#009E73", "P-G" = "#D55E00")

theme_set(
  theme_classic(base_size = 8.3, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "#222222"),
      axis.ticks = element_line(linewidth = 0.32, colour = "#222222"),
      axis.text = element_text(colour = "#222222"),
      axis.title = element_text(colour = "#111111", size = 8.4),
      plot.title = element_text(face = "bold", size = 9.4, colour = "#111111", margin = margin(b = 4)),
      plot.subtitle = element_text(size = 7.3, colour = "#52616F", margin = margin(b = 4)),
      legend.title = element_blank(),
      legend.text = element_text(size = 7.1),
      plot.margin = margin(7, 9, 7, 9),
      panel.grid.major.y = element_line(linewidth = 0.22, colour = "#E7ECF2"),
      panel.grid.major.x = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 8.1)
    )
)

mean_ci <- function(x) {
  tibble(
    mean = mean(x, na.rm = TRUE),
    sd = sd(x, na.rm = TRUE)
  )
}

sampled_top10 <- df %>%
  filter(candidate_setting == "1:100 sampled", metric == "Top10", short %in% direction_levels) %>%
  mutate(short = factor(short, levels = direction_levels)) %>%
  pivot_longer(c(observed, random_ranking), names_to = "series", values_to = "value") %>%
  mutate(series = recode(series, observed = "Observed", random_ranking = "Random-ranking"))

sampled_top10_summary <- sampled_top10 %>%
  group_by(short, series) %>%
  summarise(mean = mean(value), sd = sd(value), .groups = "drop")

full_r10 <- df %>%
  filter(candidate_setting == "full gallery", metric == "Recall@10", short %in% direction_levels) %>%
  mutate(short = factor(short, levels = direction_levels)) %>%
  pivot_longer(c(observed, random_ranking), names_to = "series", values_to = "value") %>%
  mutate(series = recode(series, observed = "Observed", random_ranking = "Random-ranking"))

full_r10_summary <- full_r10 %>%
  group_by(short, series) %>%
  summarise(mean = mean(value), sd = sd(value), .groups = "drop")

hmean <- read_csv(file.path(source_dir, "branch_summary.csv"), show_col_types=FALSE) %>%
 filter(metric == "HMean Top-10") %>% transmute(independent_run=seed, observed=value, random_ranking=random) %>%
 pivot_longer(c(random_ranking, observed), names_to="series", values_to="value") %>%
 mutate(series=factor(recode(series, random_ranking="Random-ranking", observed="Observed"), levels=c("Random-ranking","Observed")))

fold_df <- df %>%
  filter(
    (candidate_setting == "1:100 sampled" & metric == "Top10" & short %in% direction_levels) |
      (candidate_setting == "full gallery" & metric == "Recall@10" & short %in% direction_levels)
  ) %>%
  mutate(
    short = factor(short, levels = direction_levels),
    setting = factor(if_else(candidate_setting == "1:100 sampled", "Sampled Top-10", "Full-gallery R@10"),
                     levels = c("Sampled Top-10", "Full-gallery R@10"))
  ) %>%
  group_by(setting, short) %>%
  summarise(mean = mean(fold_enrichment), sd = sd(fold_enrichment), .groups = "drop")

p_a <- ggplot(sampled_top10_summary, aes(short, 100 * mean, fill = series)) +
  geom_col(position = position_dodge(width = 0.68), width = 0.58) +
  geom_errorbar(aes(ymin = 100 * (mean - sd), ymax = 100 * (mean + sd)),
                position = position_dodge(width = 0.68), width = 0.16, linewidth = 0.35) +
  geom_point(
    data = sampled_top10,
    aes(short, 100 * value, fill = series),
    position = position_jitterdodge(jitter.width = 0.04, dodge.width = 0.68, seed = 1701),
    shape = 21,
    size = 1.15,
    alpha = 0.55,
    stroke = 0.15,
    colour = "#222222"
  ) +
  scale_fill_manual(values = pal) +
  scale_y_continuous(labels = label_number(suffix = "%"), expand = expansion(mult = c(0.02, 0.10))) +
  labs(title = "Sampled retrieval against a random-ranking control", subtitle = "Top-10 with 100 sampled negatives", x = NULL, y = "Hit rate")

p_b <- ggplot(full_r10_summary, aes(short, 100 * mean, fill = series)) +
  geom_col(position = position_dodge(width = 0.68), width = 0.58) +
  geom_errorbar(aes(ymin = 100 * (mean - sd), ymax = 100 * (mean + sd)),
                position = position_dodge(width = 0.68), width = 0.16, linewidth = 0.35) +
  geom_point(
    data = full_r10,
    aes(short, 100 * value, fill = series),
    position = position_jitterdodge(jitter.width = 0.04, dodge.width = 0.68, seed = 1701),
    shape = 21,
    size = 1.15,
    alpha = 0.55,
    stroke = 0.15,
    colour = "#222222"
  ) +
  scale_fill_manual(values = pal) +
  scale_y_continuous(labels = label_number(suffix = "%"), expand = expansion(mult = c(0.02, 0.12))) +
  labs(title = "Full-gallery retrieval remains above null", subtitle = "Recall@10 over all held-out candidates", x = NULL, y = "Recall@10")

p_c <- ggplot(hmean, aes(series, 100 * value, group = independent_run)) +
  geom_line(colour = "#9AA8B5", linewidth = 0.45) +
  geom_point(aes(fill = series), shape = 21, size = 2.3, colour = "#222222", stroke = 0.25) +
  scale_fill_manual(values = pal) +
  scale_y_continuous(labels = label_number(suffix = "%"), expand = expansion(mult = c(0.06, 0.10))) +
  labs(title = "Paired-run HMean control", subtitle = "Each line is one independent run", x = NULL, y = "HMean Top-10") +
  theme(legend.position = "none")

p_d <- ggplot(fold_df, aes(short, mean, fill = short)) +
  geom_hline(yintercept = 1, linetype = "dashed", linewidth = 0.35, colour = "#8795A1") +
  geom_point(aes(colour = short), size = 2.1) +
  scale_colour_manual(values = pal_dir) +
  geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.14, linewidth = 0.35) +
  facet_wrap(~setting, nrow = 1, scales = "free_y") +
  scale_fill_manual(values = pal_dir) +
  scale_y_continuous(trans = "log10", breaks = c(1, 2, 5, 10, 20, 50, 100), labels = label_number(suffix = "x")) +
  labs(title = "Enrichment over random-ranking", subtitle = "Dashed line marks no enrichment", x = NULL, y = "Fold enrichment") +
  theme(legend.position = "none", strip.text = element_text(size = 7.5, face = "bold"))

fig <- (p_a | p_b) / (p_c | p_d) +
  plot_layout(heights = c(1, 1), widths = c(1, 1.05)) +
  plot_annotation(tag_levels = "a") &
  theme(
    plot.margin = margin(7, 11, 7, 9),
    plot.tag = element_text(size = 13, face = "bold", colour = "#111111"),
    plot.tag.position = c(0, 1),
    plot.title = element_blank(),
    plot.subtitle = element_blank()
  )

out_base <- file.path(figure_dir, "figureS5")
width_mm <- 183
height_mm <- 132
w <- width_mm / 25.4
h <- height_mm / 25.4

ggsave(paste0(out_base, ".png"), fig, width = w, height = h, dpi = 600, bg = "white", device = ragg::agg_png)
ggsave(paste0(out_base, ".tiff"), fig, width = w, height = h, dpi = 600, bg = "white", device = ragg::agg_tiff, compression = "lzw")
ggsave(paste0(out_base, ".pdf"), fig, width = w, height = h, bg = "white", device = cairo_pdf, family = "Arial")
ggsave(paste0(out_base, ".svg"), fig, width = w, height = h, bg = "white", device = svglite::svglite)

message("Wrote: ", out_base)

writeLines(capture.output(sessionInfo()), file.path(figure_dir, "sessionInfo_S5.txt"))
if(length(warnings())) print(warnings())
