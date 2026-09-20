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

loss <- read_csv(file.path(source_dir,"figureS1_losses.csv"), show_col_types=FALSE)
val <- read_csv(file.path(source_dir,"figureS1_validation.csv"), show_col_types=FALSE)
selected <- read_csv(file.path(source_dir,"figureS1_selected.csv"), show_col_types=FALSE)
pal <- c("31"="#165A7C", "37"="#148A7B", "41"="#8C6BB1")
loss$seed <- factor(loss$seed); val$seed <- factor(val$seed); selected$seed <- factor(selected$seed)
loss$component <- factor(loss$component,levels=c("Total objective","Compound","ORF","CRISPR"))
selected$direction <- factor(selected$direction,levels=c("C-P","P-C","G-P","P-G"))
theme_set(theme_classic(base_size=8.3,base_family="Arial") + theme(
 axis.line=element_line(linewidth=.35,colour="#222222"),
 axis.text=element_text(colour="#222222"), axis.title=element_text(size=8.4),
 panel.grid.major.y=element_line(linewidth=.22,colour="#E7ECF2"),
 strip.background=element_blank(),strip.text=element_text(face="bold",size=8.1),
 legend.title=element_blank(),legend.text=element_text(size=7.1),
 plot.title=element_text(face="bold",size=9),plot.margin=margin(7,9,7,9)))
p_a <- ggplot(loss,aes(epoch,loss,colour=seed)) + geom_line(linewidth=.35) +
 facet_wrap(~component,ncol=2,scales="free_y") + scale_colour_manual(values=pal) +
 labs(x="Epoch",y="Training loss",title="Training trajectories") + theme(legend.position="none")
sel <- val %>% filter(selected==1)
p_b <- ggplot(val,aes(epoch,100*value,colour=seed)) +
 annotate("rect",xmin=40,xmax=300,ymin=-Inf,ymax=Inf,fill="#EAF1F4",alpha=.5) +
 geom_line(linewidth=.5) + geom_point(data=sel,size=2.3,shape=18) +
 scale_colour_manual(values=pal,labels=paste("Seed",names(pal))) +
 scale_y_continuous(labels=label_number(suffix="%")) +
 labs(x="Epoch",y="Validation HMean Top-10",title="Validation and model selection") +
 theme(legend.position="bottom")
candidates <- val %>% filter(selection_active==1)
p_c <- ggplot(candidates,aes(epoch,100*value,colour=seed)) + geom_point(size=.8,alpha=.55) +
 geom_point(data=sel,size=2,shape=18) +
 geom_text(data=sel,aes(label=paste0("epoch ",epoch),hjust=ifelse(epoch<100,0,.5)),nudge_y=.7,size=2.4,colour="#222222") +
 facet_wrap(~seed,nrow=1,labeller=label_both) + scale_colour_manual(values=pal) +
 scale_x_continuous(breaks=c(40,150,300)) +
 scale_y_continuous(labels=label_number(suffix="%"),expand=expansion(mult=c(.08,.23))) +
 labs(x="Eligible epoch",y="Validation HMean Top-10",title="Selected checkpoints") + theme(legend.position="none",panel.spacing.x=grid::unit(3,"mm"),axis.text.x=element_text(size=7))
p_d <- ggplot(selected,aes(direction,100*value,colour=seed,shape=seed)) +
 geom_point(size=2,position=position_dodge(width=.32)) +
 scale_colour_manual(values=pal,labels=paste("Seed",names(pal))) +
 scale_shape_manual(values=c(16,17,15),labels=paste("Seed",names(pal))) +
 scale_y_continuous(labels=label_number(suffix="%"),limits=c(0,100)) +
 labs(x=NULL,y="Validation Top-10",title="Selected-model retrieval") + theme(legend.position="none")
fig <- (p_a | p_b) / (p_c | p_d) + plot_annotation(tag_levels="a") &
 theme(plot.tag=element_text(size=13,face="bold"),plot.tag.position=c(0,1))
w<-183/25.4; h<-132/25.4; out_base<-file.path(figure_dir,"figureS1")
ggsave(paste0(out_base,".png"),fig,width=w,height=h,dpi=600,bg="white",device=ragg::agg_png)
ggsave(paste0(out_base,".tiff"),fig,width=w,height=h,dpi=600,bg="white",device=ragg::agg_tiff,compression="lzw")
ggsave(paste0(out_base,".pdf"),fig,width=w,height=h,bg="white",device=cairo_pdf,family="Arial")
ggsave(paste0(out_base,".svg"),fig,width=w,height=h,bg="white",device=svglite::svglite)
writeLines(capture.output(sessionInfo()),file.path(figure_dir,"sessionInfo_S1.txt"))
if(length(warnings())) print(warnings())
message("Wrote: ",out_base)
