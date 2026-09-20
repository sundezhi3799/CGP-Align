args <- commandArgs(trailingOnly=FALSE)
p <- sub("^--file=", "", args[grepl("^--file=",args)][1])
base <- dirname(normalizePath(p))
for (f in c("figure4.R","figure5.R","figure6.R","figureS2.R","figureS3.R")) {
 status <- system2(file.path(R.home("bin"),"Rscript"), shQuote(file.path(base,"scripts",f)))
 if (status != 0) warning(paste(f,"returned",status,"; inspect exports and logs"))
}
