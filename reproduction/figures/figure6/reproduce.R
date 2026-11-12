all_args <- commandArgs(trailingOnly=FALSE)
script_path <- sub('^--file=', '', all_args[grepl('^--file=', all_args)][1])
base_dir <- dirname(normalizePath(script_path,winslash='/',mustWork=TRUE))
# Figure 6. Run from the repository root.
# Rscript scripts/plot_figure6.R <u2os_similarity directory> <output directory>
suppressPackageStartupMessages({
 library(ggplot2); library(dplyr); library(tidyr); library(utils)
 library(patchwork); library(scales); library(grid); library(png)
})
options(device=function(...) grDevices::cairo_pdf(file=tempfile(fileext=".pdf"),...))
args <- commandArgs(trailingOnly=TRUE)
if(length(args)==0) args <- c(file.path(base_dir,"u2os_similarity"),file.path(base_dir,"exports"))
if(length(args)!=2) stop("Supply input and output directories")
comparison_root <- normalizePath(args[1],winslash="/",mustWork=TRUE)
fig_dir <- args[2]
dir.create(fig_dir,recursive=TRUE,showWarnings=FALSE)
src_dir <- normalizePath(file.path(base_dir,"source_data"),winslash="/",mustWork=TRUE)
read_csv <- function(file,show_col_types=FALSE,...){d<-read.csv(file,check.names=FALSE,stringsAsFactors=FALSE);for(k in names(d)){if(is.character(d[[k]]) && all(na.omit(d[[k]]) %in% c('True','False')))d[[k]]<-d[[k]]=='True'};d}
method_levels <- c("ChemBERTa", "MoLFormer", "CGP-Align", "RDKit2D", "NYAN", "Morgan", "AtomPair", "Avalon", "Random")
main_methods <- c("CGP-Align", "RDKit2D", "NYAN", "ChemBERTa", "MoLFormer", "Morgan", "Random")
method_cols <- c("ChemBERTa"="#9B83BF", "MoLFormer"="#7353A2",
  "CGP-Align" = "#0B4C78",
  "RDKit2D" = "#8E9AA3",
  "NYAN" = "#128A7D",
  "Morgan" = "#CB6F4A",
  "AtomPair" = "#2A9D8F",
  "Avalon" = "#A8AEB5",
  "Random" = "#D3D7DB"
)
method_lty <- c("ChemBERTa"="solid", "MoLFormer"="solid",
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
  analysis_dir <- dirname(comparison_root)
  generated <- c(
    figure6_panel_b_functional_neighbour_enrichment_source.csv="prism_high_confidence_hit_summary.csv",
    figure6_panel_d_pancancer_source.csv="prism_continuous_metric_summary.csv"
  )
  if(x %in% names(generated) && file.exists(file.path(analysis_dir,generated[[x]]))) {
    return(read_csv(file.path(analysis_dir,generated[[x]]),show_col_types=FALSE))
  }
  if(x=="figure6_panel_c_cgp_vs_rdkit2d_source.csv" && file.exists(file.path(analysis_dir,"prism_u2os_single_cell_query_metrics.csv"))) {
    return(read_csv(file.path(analysis_dir,"prism_u2os_single_cell_query_metrics.csv"),show_col_types=FALSE) %>%
      filter(active_threshold == -1,low_tanimoto == .20,topk == 50,method %in% c("CGP-Align","RDKit2D")) %>%
      select(query_idx,method,top_active_fraction) %>%
      pivot_wider(names_from=method,values_from=top_active_fraction) %>%
      mutate(cgp_better=`CGP-Align`>RDKit2D,same=`CGP-Align`==RDKit2D,n_query=1))
  }
  read_csv(file.path(src_dir, x), show_col_types = FALSE)
}

wrap_text <- function(x, width = 24) {
  vapply(x, function(z) paste(strwrap(z, width = width), collapse = "\n"), character(1))
}

structure_grob <- function(path) {
 name <- if(grepl("BIIB021",basename(path))) "BIIB021" else "R547"
 atoms <- read.csv(file.path(src_dir,"chemical_vectors",paste0(name,"_atoms.csv")),na.strings="")
 bonds <- read.csv(file.path(src_dir,"chemical_vectors",paste0(name,"_bonds.csv")))
 atoms$label[is.na(atoms$label)] <- ""
 if(name=="R547") { tmp<-atoms$x; atoms$x<-atoms$y; atoms$y<- -tmp }
 span<-max(diff(range(atoms$x)),diff(range(atoms$y)))
 atoms$x<-(atoms$x-mean(range(atoms$x)))/span*.82+.5
 atoms$y<-(atoms$y-mean(range(atoms$y)))/span*.82+.5
 g<-list()
 for(j in seq_len(nrow(bonds))) {
  a<-atoms[bonds$a[j]+1,]; b<-atoms[bonds$b[j]+1,]
  dx<-b$x-a$x;dy<-b$y-a$y;l<-sqrt(dx*dx+dy*dy)
  ux<-dx/l;uy<-dy/l
  cut1<-if(nchar(a$label)>0) .032 else 0
  cut2<-if(nchar(b$label)>0) .032 else 0
  shifts<-if(bonds$order[j]==2) c(-.011,.011) else if(bonds$order[j]==3) c(-.02,0,.02) else 0
  for(off in shifts) g[[length(g)+1]]<-segmentsGrob(a$x+ux*cut1-uy*off,a$y+uy*cut1+ux*off,b$x-ux*cut2-uy*off,b$y-uy*cut2+ux*off,gp=gpar(col="#151515",lwd=.85,lineend="round"))
 }
 for(i in which(nchar(atoms$label)>0)) {
  label<-atoms$label[i]
  # Atom labels remain editable text; hydrogens are chemically derived from SDF.
  if(grepl('H2$',label)) label<-parse(text=paste0(substr(label,1,nchar(label)-2),'H[2]'))[[1]]
  tg<-textGrob(label,x=atoms$x[i],y=atoms$y[i],gp=gpar(fontfamily="Arial",fontsize=6.1,col="#151515"))
  g[[length(g)+1]]<-rectGrob(x=atoms$x[i],y=atoms$y[i],width=grobWidth(tg)+unit(.65,"pt"),height=grobHeight(tg)+unit(.35,"pt"),gp=gpar(fill="white",col=NA))
  g[[length(g)+1]]<-tg
 }
 gTree(children=do.call(gList,g),vp=viewport(width=unit(1,"snpc"),height=unit(1,"snpc")))
}

functional_hits <- read_source("figure6_panel_b_functional_neighbour_enrichment_source.csv") %>%
  mutate(method = factor(method, levels = method_levels))
query_pairs <- read_source("figure6_panel_c_cgp_vs_rdkit2d_source.csv")
pan <- read_source("figure6_panel_d_pancancer_source.csv") %>%
  mutate(method = factor(method, levels = method_levels))
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
  scale_colour_manual(values = method_cols, breaks = main_methods, drop = TRUE) +
  scale_linetype_manual(values = method_lty, breaks = main_methods, drop = TRUE) +
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
    x = "RDKit2D fraction",
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
  scale_colour_manual(values = method_cols, breaks = main_methods, drop = TRUE) +
  scale_linetype_manual(values = method_lty, breaks = main_methods, drop = TRUE) +
  scale_x_reverse(breaks = c(0.30, 0.25, 0.20, 0.15), limits = c(0.31, 0.14)) +
  scale_y_continuous(labels = label_number(accuracy = 0.01), expand = expansion(mult = c(0.04, 0.10))) +
  labs(
    x = "Max. Morgan Tanimoto",
    y = "Oracle-normalized retention@50"
  ) +
  guides(colour = "none", linetype = "none")

panel_c <- panel_c + scale_size_area(max_size=2.8,guide="none") + labs(x="RDKit-Morgan fraction")
panel_b <- panel_b + scale_colour_manual(values=method_cols,breaks=main_methods,labels=c("CGP-Align","RDKit-Morgan","NYAN","ChemBERTa","MoLFormer","Morgan","Random")) + scale_linetype_manual(values=method_lty,breaks=main_methods,labels=c("CGP-Align","RDKit-Morgan","NYAN","ChemBERTa","MoLFormer","Morgan","Random"))


old_a <- png::readPNG(file.path(src_dir,"workflow.png"))
panel_a <- wrap_elements(full=rasterGrob(old_a,width=unit(.98,"snpc"),height=unit(.98*dim(old_a)[1]/dim(old_a)[2],"snpc"),interpolate=TRUE))
query_structure <- "BIIB021"
retrieved_structure <- "R547"
# Compact functional summary; each shared gene has one display group.
func <- read_source("figure6e_function_groups.csv") %>% arrange(desc(n)) %>%
 mutate(group=factor(group,levels=rev(group)))

pair_metrics <- read_source("figure6_network_case.csv") %>% filter(query_name=="BIIB021",retrieved_name=="R547") %>% slice(1)
metric_labels <- c("Morgan Tanimoto", "Pan-cancer PRISM r", "Shared top-50 genes")
metric_values <- c(sprintf("%.3f",pair_metrics$morgan_tanimoto),sprintf("%.3f",pair_metrics$prism_response_corr),sprintf("%d / %d",pair_metrics$shared_top_gene_count,pair_metrics$top_gene_k))
pair_panel <- ggplot()+
 annotate("text",x=.5,y=1.01,label="BIIB021 and R547",size=2.5,fontface="bold",colour="#222222")+
 annotation_custom(structure_grob(query_structure),xmin=.01,xmax=.99,ymin=.65,ymax=.97)+
 annotate("text",x=.5,y=.645,label="BIIB021",size=2.6,fontface="bold",colour="#154C63")+
 annotation_custom(structure_grob(retrieved_structure),xmin=.01,xmax=.99,ymin=.31,ymax=.62)+
 annotate("text",x=.5,y=.315,label="R547",size=2.6,fontface="bold",colour="#7B6BAF")+
 annotate("text",x=.045,y=c(.215,.14,.065),label=metric_labels,hjust=0,size=2.05,colour="#344D59")+
 annotate("text",x=.95,y=c(.215,.14,.065),label=metric_values,hjust=1,size=2.55,fontface="bold",colour="#154C63")+
 coord_cartesian(xlim=c(0,1),ylim=c(0,1.04),expand=FALSE)+theme_void()+theme(plot.margin=margin(6,8,6,5))
# UniProt-based functional categories; each shared gene is counted once.
function_panel <- ggplot(func,aes(n,group))+
 geom_col(width=.58,fill="#729CA9")+
 geom_text(aes(label=n),hjust=-.4,size=2.3,colour="#222222")+
 scale_x_continuous(limits=c(0,10.3),breaks=c(0,3,6,9),expand=c(0,0))+
 labs(title="Shared-gene functions",x="Number of shared genes",y=NULL)+
 theme_classic(base_size=7,base_family="Arial")+
 theme(plot.title=element_text(size=7.5,face="bold",colour="#222222"),plot.subtitle=element_text(size=6.2,colour="#526774"),axis.text.y=element_text(size=6.1,colour="#243E49"),axis.text.x=element_text(size=6.3),axis.title.x=element_text(size=6.8),axis.line.y=element_blank(),axis.ticks.y=element_blank(),panel.grid.major.x=element_line(colour="#EDF1F3",linewidth=.2),plot.margin=margin(9,8,9,3))
panel_e <- wrap_elements(wrap_elements(full=ggplotGrob(pair_panel))+function_panel+plot_layout(widths=c(1.6,1.35)))

paired_q <- read_csv(file.path(comparison_root,"paired_queries.csv"),show_col_types=FALSE) %>% filter(active_query)
comparison_stats <- read_csv(file.path(comparison_root,"summary.csv"),show_col_types=FALSE) %>% filter(scope=="active_queries")
upper <- ceiling(max(paired_q$`CGP-Align`,paired_q$NYAN,paired_q$`RDKit-Morgan`)*2)/2
make_comparison <- function(base) {
 st <- comparison_stats %>% filter(baseline==base)
 ggplot(paired_q,aes(x=.data[[base]],y=.data[["CGP-Align"]]))+
 geom_abline(slope=1,intercept=0,linetype="dashed",colour="#A6B2BA",linewidth=.3)+
 geom_point(shape=21,fill="#154C63",colour="white",stroke=.16,size=1.3,alpha=.75)+
 scale_x_continuous(limits=c(0,upper),breaks=0:5,expand=expansion(mult=c(.01,.025)))+
 scale_y_continuous(limits=c(0,upper),breaks=0:5,expand=expansion(mult=c(.01,.025)))+
 coord_equal()+
 labs(title=base,x=bquote(.(base)~mean~"|"*Delta*LFC*"|"),y=expression("CGP-Align mean "*"|"*Delta*LFC*"|"))+
 theme_classic(base_size=7,base_family="Arial")+
 theme(plot.title=element_text(size=7.5,face="bold",colour="#222222"),plot.subtitle=element_text(size=5.8),axis.title=element_text(size=6.5),axis.text=element_text(size=6.3,colour="#243E49"),plot.margin=margin(4,8,3,5))
}
gain_q <- read_csv(file.path(comparison_root,"query_comparison.csv"),show_col_types=FALSE) %>% filter(query_active)
stopifnot(identical(sort(gain_q$query_idx),sort(paired_q$query_idx)))
gain_panel <- ggplot(gain_q,aes(x=1,y=gain))+
 geom_hline(yintercept=0,colour="#8798A2",linetype="dashed",linewidth=.3)+
 geom_violin(fill="#EAF1F4",colour="#7296A6",width=.68,linewidth=.35,trim=TRUE)+
 geom_point(position=position_jitter(width=.13,height=0,seed=20260923),size=.55,alpha=.38,colour="#35647A")+
 annotate("segment",x=.82,xend=1.18,y=mean(gain_q$gain),yend=mean(gain_q$gain),colour="#A9684A",linewidth=.4)+
 annotate("point",x=1,y=mean(gain_q$gain),shape=23,size=2.0,fill="#A9684A",colour="white",stroke=.3)+
 annotate("text",x=1,y=max(gain_q$gain)+.16,label=sprintf("Mean = %.2f",mean(gain_q$gain)),hjust=.5,size=2.1,family="Arial",colour="#A9684A")+
 scale_y_continuous(limits=c(min(gain_q$gain)-.08,max(gain_q$gain)+.30),breaks=c(0,.5,1,1.5),expand=c(0,0))+
 scale_x_continuous(breaks=1,labels="Matched control\nminus CGP-Align",limits=c(.55,1.45))+
 labs(title="Matched control",x=NULL,y=expression("Reduction in mean "*"|"*Delta*LFC*"|"))+
 theme_classic(base_size=7,base_family="Arial")+
 theme(plot.title=element_text(size=7.5,face="bold",colour="#222222"),plot.subtitle=element_text(size=5.8),axis.title=element_text(size=6.5),axis.text=element_text(size=6.3,colour="#243E49"),plot.margin=margin(4,8,3,5))
# Common typography and axis geometry at final publication size.
axis_style <- theme(axis.line=element_line(linewidth=.28,colour="#333333"),axis.ticks=element_line(linewidth=.28,colour="#333333"),axis.ticks.length=unit(2,"pt"),axis.text=element_text(size=6.4,colour="#333333"),axis.title=element_text(size=7,colour="#222222"),panel.grid=element_blank(),panel.grid.major=element_blank(),panel.grid.major.x=element_blank(),panel.grid.major.y=element_blank(),panel.grid.minor=element_blank(),plot.margin=margin(5,7,5,5))
panel_b <- panel_b + axis_style
panel_c <- panel_c + axis_style + scale_size_area(max_size=2.15,guide="none") + labs(x="RDKit-Morgan active fraction",y="CGP-Align active fraction")
panel_d <- panel_d + axis_style + scale_y_continuous(limits=c(-.005,.105),breaks=c(0,.025,.05,.075,.1),labels=c("0","0.025","0.050","0.075","0.100"))
for(nm in c("panel_b","panel_d")) {
 pp <- get(nm)
 for(i in seq_along(pp$layers)) {
  if(inherits(pp$layers[[i]]$geom,"GeomPoint")) pp$layers[[i]]$aes_params$size <- 1.9
  if(inherits(pp$layers[[i]]$geom,"GeomLine")) pp$layers[[i]]$aes_params$linewidth <- .55
 }
 assign(nm,pp)
}
gain_panel <- gain_panel + axis_style + theme(aspect.ratio=NULL)
group_heading <- wrap_elements(full=textGrob("U2OS response similarity (n = 187 queries)",x=unit(5,"pt"),hjust=0,gp=gpar(fontfamily="Arial",fontsize=7.5,fontface="bold",col="#222222")),ignore_tag=TRUE)

newdir<-file.path(base_dir,'..','summary')
new_b<-read_csv(file.path(newdir,'prism_enrichment.csv')) %>% filter(method %in% c('ChemBERTa','MoLFormer')) %>% mutate(low_tanimoto=cutoff)
panel_b_data<-bind_rows(panel_b_data %>% mutate(method=as.character(method)),new_b) %>% mutate(method=factor(method,levels=main_methods))
new_d<-read_csv(file.path(newdir,'prism_continuous_summary.csv')) %>% filter(method %in% c('ChemBERTa','MoLFormer')) %>% mutate(low_tanimoto=cutoff)
panel_d_data<-bind_rows(panel_d_data %>% mutate(method=as.character(method)),new_d) %>% mutate(method=factor(method,levels=main_methods))
display<-setNames(main_methods,main_methods);display['RDKit2D']<-'RDKit-Morgan'
lineplot<-function(dd,metric,ylab){
 ggplot(dd,aes(low_tanimoto,.data[[metric]],colour=method,linetype=method,group=method))+
 geom_line(linewidth=.55)+geom_point(size=1.6)+
 scale_colour_manual(values=method_cols,breaks=main_methods,labels=display)+scale_linetype_manual(values=method_lty,breaks=main_methods,labels=display)+
 scale_x_reverse(breaks=c(.30,.25,.20,.15),limits=c(.31,.14))+
 labs(x='Max. Morgan Tanimoto',y=ylab,colour=NULL,linetype=NULL)+axis_style+
 guides(colour=guide_legend(nrow=2,byrow=TRUE),linetype=guide_legend(nrow=2,byrow=TRUE))
}
panel_b<-lineplot(panel_b_data,'enrichment','Enrichment over random')+geom_hline(yintercept=1,colour='#B7C0C7',linetype='dotted',linewidth=.28)
panel_d<-lineplot(panel_d_data,'retention50','Oracle-normalized retention@50')
q<-read_csv(file.path(newdir,'prism_u2os_queries.csv')) %>% filter(active==TRUE)
qmethods<-c('CGP-Align','RDKit-Morgan','NYAN','ChemBERTa','MoLFormer')
qcols<-c('CGP-Align'='#0B4C78','RDKit-Morgan'='#8E9AA3','NYAN'='#128A7D','ChemBERTa'='#9B83BF','MoLFormer'='#7353A2')
q<-q %>% mutate(method=factor(method,levels=rev(qmethods)))
qm<-q %>% group_by(method) %>% summarise(value=mean(top_active_fraction),.groups='drop')
panel_c<-ggplot(q,aes(top_active_fraction,method,fill=method))+
 geom_boxplot(width=.48,outlier.shape=NA,alpha=.4,linewidth=.3)+
 geom_point(aes(colour=method),position=position_jitter(width=0,height=.12,seed=41),size=.45,alpha=.25)+
 geom_point(data=qm,aes(value,method),inherit.aes=FALSE,shape=23,fill='white',size=1.5,stroke=.3)+
 scale_fill_manual(values=qcols,guide='none')+scale_colour_manual(values=qcols,guide='none')+
 scale_x_continuous(limits=c(0,1),breaks=c(0,.5,1))+
 labs(x='Top-50 active fraction',y=NULL)+axis_style+theme(axis.text.y=element_text(size=6))
qq<-q %>% select(query_idx,method,mean_absdiff) %>% pivot_wider(names_from=method,values_from=mean_absdiff)
pair_panel_new<-function(bases){
 dd<-qq %>% pivot_longer(all_of(bases),names_to='comparator',values_to='value') %>% mutate(comparator=factor(comparator,levels=bases))
 ggplot(dd,aes(value,`CGP-Align`))+geom_abline(slope=1,intercept=0,colour='#A6B2BA',linetype='dashed',linewidth=.3)+
 geom_point(shape=21,fill='#0B4C78',colour='white',stroke=.10,size=.8,alpha=.75)+
 facet_wrap(~comparator,nrow=1)+coord_fixed(xlim=c(0,5.5),ylim=c(0,5.5))+
 scale_x_continuous(breaks=c(0,2,4))+scale_y_continuous(breaks=c(0,2,4))+
 labs(x=expression('Comparator mean '*'|'*Delta*LFC*'|'),y=expression('CGP-Align mean '*'|'*Delta*LFC*'|'))+
 axis_style+theme(strip.text=element_text(size=6),axis.title=element_text(size=6),axis.text=element_text(size=5.8),panel.spacing.x=unit(2,'mm'))
}
gain_panel<-gain_panel+theme(axis.title.y=element_text(size=6),axis.text=element_text(size=6),plot.title=element_text(size=6.5))
design<-'AAAAABBBBBCCCCC
DDDDDEEEEEEEEEE
FFFFFFFFFFFFGGG'
combined<-panel_a+panel_b+panel_d+panel_c+panel_e+pair_panel_new(c('NYAN','RDKit-Morgan','ChemBERTa','MoLFormer'))+gain_panel+
 plot_layout(design=design,heights=c(1,1.1,.70),guides='collect')+
 plot_annotation(tag_levels=list(c('a','b','c','d','e','f','g'))) &
 theme(plot.tag=element_text(size=9,face='bold',family='Arial'),legend.position='top',legend.text=element_text(size=6.3),legend.key.width=unit(10,'pt'),legend.margin=margin(0,0,0,0))
prefix<-file.path(fig_dir,'figure6')
svglite::svglite(paste0(prefix,'.svg'),width=183/25.4,height=180/25.4);print(combined);dev.off()
converter<-Sys.getenv('RSVG_CONVERT', Sys.which('rsvg-convert'))
for(fmt in c('pdf','png')){
 opts<-c(paste0('--format=',fmt),'--output',shQuote(paste0(prefix,'.',fmt)))
 if(fmt=='png')opts<-c(opts,'--dpi-x','300','--dpi-y','300')
 stopifnot(system2(converter,c(opts,shQuote(paste0(prefix,'.svg'))),env='LD_LIBRARY_PATH=')==0)
}
stopifnot(nrow(q)==187*5,nrow(qq)==187,nrow(panel_b_data)==28,nrow(panel_d_data)==28)
writeLines(capture.output(sessionInfo()),file.path(fig_dir,'sessionInfo.txt'))
