packages <- c("ggplot2", "dplyr", "tidyr", "readr", "patchwork", "scales", "ragg")
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
if (!all(vapply(packages, requireNamespace, logical(1), quietly = TRUE))) stop("Missing figure dependencies")
