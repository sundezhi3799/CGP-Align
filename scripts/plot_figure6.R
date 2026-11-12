# Reproduce the manuscript figure from the supplied source tables.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE)[1])
root <- dirname(dirname(normalizePath(script, winslash="/")))
target <- file.path(root, "reproduction", "figures", "figure6/reproduce.R")
args <- commandArgs(trailingOnly=TRUE)
if(length(args)==1) args <- c(file.path(dirname(target), "u2os_similarity"), args)
status <- system2(file.path(R.home("bin"), "Rscript"), c(shQuote(target), vapply(args, shQuote, character(1))))
quit(status=status)
