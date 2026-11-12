read_csv <- function(file,show_col_types=FALSE,...) read.csv(file,check.names=FALSE,...)
write_csv <- function(x,file) write.csv(x,file,row.names=FALSE,na='')
all_args <- commandArgs(trailingOnly=FALSE)
script_path <- sub('^--file=', '', all_args[grepl('^--file=', all_args)][1])
base_dir <- dirname(normalizePath(script_path,winslash='/',mustWork=TRUE))
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)

  library(grid)
  library(patchwork)
})

options(device=function(...) grDevices::cairo_pdf(file=tempfile(fileext=".pdf"),...))
source_dir <- file.path(base_dir,"source_data")
args <- commandArgs(trailingOnly=TRUE)
if(length(args)==0) args <- file.path(base_dir,"exports")
if(length(args)!=1) stop("Supply one output directory")
figure_dir <- args[1]
dir.create(figure_dir,recursive=TRUE,showWarnings=FALSE)
ink <- "#1F2D3A"
muted_ink <- "#5D6B78"
grid_col <- "#E9EEF3"
axis_col <- "#263542"
font_family <- "Arial"

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
        size = 8.5,
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
    pattern_id = as.integer(pattern_id), active = tolower(active)=="true",
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
  geom_segment(aes(x = 1, xend = mean_lift50, yend = name), linewidth = 0.65, colour = "#2A9D8F", alpha = 0.92) +
  geom_point(size = 2.1, colour = "#138A7E") +
  geom_text(aes(label = sprintf("%.2f", mean_lift50)), hjust = -0.18, size = 2.45, family = font_family, colour = ink) +
  panel_tag("b") +
  scale_x_continuous(limits = c(0.85, max(lift_df$mean_lift50) + 0.7), breaks = c(1, 2, 3, 4), expand = expansion(mult = c(0.01, 0.08))) +
  labs(x = "Mean Lift@50", y = NULL) +
  theme_fig(7.3) +
  theme(panel.grid.major.y = element_blank(), plot.margin = margin(6, 4, 3, 5))

coords <- read_csv(
  file.path(source_dir, "figureS9_pca_coordinates.csv"),
  show_col_types = FALSE
) %>%
  mutate(entity_type = factor(entity_type, levels = c("Compound", "Gene")))

link_coords <- read_csv(
  file.path(source_dir, "figureS9_pca_links.csv"),
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
    alpha = 0.62,
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
    alpha = 0.64,
    show.legend = FALSE
  ) +
  geom_point(
    data = highlight_points,
    aes(x = pca_x, y = pca_y, shape = entity_type, fill = entity_type),
    size = 2.4,
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
  coord_fixed(ratio=1, clip="on") +
  theme_fig(7.4) +
  theme(

    legend.position = "bottom",
    legend.box = "vertical", legend.direction="horizontal",
    legend.box.spacing = unit(0.3, "mm"),
    legend.key.height = unit(3.0, "mm"),
    legend.key.width = unit(5.6, "mm"),
    plot.margin = margin(3, 3, 2, 3)
  ) +
  guides(
    fill = "none",
    shape = guide_legend(
      order = 1,
      override.aes = list(fill = unname(entity_cols), colour = "white", size = 2.25, alpha = 1)
    ),
    colour = guide_legend(order = 2, nrow=2, byrow=TRUE, override.aes = list(linewidth = 0.75, alpha = 1))
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
      linewidth = 0.40,
      alpha = 0.90,
      lineend = "round"
    )
}

plot_e <- plot_e_base + ggrepel::geom_label_repel(data = highlight_points,
 aes(x=pca_x,y=pca_y,label=label), colour=ink, size=2.2, family=font_family,
 fill="white", linewidth=0.08, label.padding=unit(0.06,"lines"), label.r=unit(0.03,"lines"), box.padding=0.35, point.padding=0.15,
 max.overlaps=Inf, seed=41, min.segment.length=0, segment.colour="#AAB7C1")


# Matched representation benchmark. Summary statistics are computed on gpu80.
raw32 <- read_csv(file.path(source_dir,if(file.exists(file.path(source_dir,'metrics32_internal.csv'))) 'metrics32_internal.csv' else 'relation_metrics32.csv'),show_col_types=FALSE)
raw64 <- read_csv(file.path(source_dir,if(file.exists(file.path(source_dir,'metrics64_internal.csv'))) 'metrics64_internal.csv' else 'relation_metrics64.csv'),show_col_types=FALSE)
keep <- c('RDKit','RDKit-Morgan','Morgan2048','AtomPair2048','Avalon1024','NYAN','ChemBERTa','MoLFormer','NYAN+RDKit','ChemBERTa+RDKit','MoLFormer+RDKit','CGP-Align_ESM2','CGP-Align+RDKit_ESM2','CGP-Align_aligned','CGP-Align+RDKit_aligned')
raw32 <- raw32 %>% filter(method %in% keep) %>% mutate(dimension=32)
raw64 <- raw64 %>% filter(method %in% keep) %>% mutate(dimension=64)
all <- bind_rows(raw32,raw64)
sm <- all %>% group_by(dimension,graph,source,method) %>% summarise(ap_sd=sd(auprc),auc_sd=sd(auroc),auprc=mean(auprc),auroc=mean(auroc),.groups='drop')
overall <- sm %>% group_by(dimension,graph,method) %>% summarise(auprc=mean(auprc),auroc=mean(auroc),.groups='drop')
order <- c('CGP-Align+RDKit_ESM2','CGP-Align_ESM2','MoLFormer+RDKit','ChemBERTa+RDKit','NYAN+RDKit','MoLFormer','ChemBERTa','NYAN','RDKit-Morgan','RDKit','Morgan2048','AtomPair2048','Avalon1024')
labels <- setNames(c('CGP-Align + RDKit','CGP-Align','MoLFormer + RDKit','ChemBERTa + RDKit','NYAN + RDKit','MoLFormer','ChemBERTa','NYAN','RDKit-Morgan','RDKit','Morgan','AtomPair','Avalon'),order)
src <- setNames(source_levels,tolower(source_levels))
heat <- function(dim,metric,tag){
 dat <- sm %>% filter(dimension==dim,method %in% order) %>% mutate(value=.data[[metric]],method=factor(method,levels=rev(order)),source=factor(src[source],levels=source_levels),graph=factor(graph,levels=c('broad','strict'),labels=c('Broad relations','Target-related relations'))) %>% group_by(graph,source) %>% mutate(best=value==max(value)) %>% ungroup()
 ggplot(dat,aes(source,method,fill=value))+geom_tile(colour='white',linewidth=.3)+
 geom_tile(data=filter(dat,best),fill=NA,colour='#263542',linewidth=.45)+
 geom_text(aes(label=sprintf('%.3f',value),fontface=ifelse(best,'bold','plain')),size=2.15,family=font_family,colour='#172A36')+
 facet_grid(~graph)+scale_y_discrete(labels=labels)+scale_x_discrete(position='top',labels=c('DGIdb','DrugRep','Hetionet','OpenBioLink','PharMeBINet'))+
 scale_fill_gradient(low='#F5F8F9',high='#76B7B2',limits=c(.48,.82),name=ifelse(metric=='auprc','AUPRC','AUROC'))+
 labs(x=NULL,y=NULL,tag=tag)+theme_fig(7.2)+theme(axis.line=element_blank(),axis.ticks=element_blank(),panel.grid=element_blank(),axis.text.x=element_text(size=5.7,angle=20,hjust=0),axis.text.y=element_text(size=6.7),strip.background=element_blank(),strip.text=element_text(size=7.4,face='bold'),legend.position='right',legend.key.height=unit(4,'mm'),legend.key.width=unit(2.5,'mm'),legend.text=element_text(size=6),legend.title=element_text(size=6.5),panel.spacing=unit(3,'mm'),plot.margin=margin(4,3,4,4))
}

# Same source-held-out AP values; bars summarize three repeats, errors are sample SD.
bar_order <- c('RDKit','RDKit-Morgan','Morgan2048','AtomPair2048','Avalon1024',
 'NYAN','ChemBERTa','MoLFormer','CGP-Align_ESM2',
 'NYAN+RDKit','ChemBERTa+RDKit','MoLFormer+RDKit','CGP-Align+RDKit_ESM2')
bar_cols <- setNames(c('#9AA6B2','#B7936B','#D2B99D','#667883','#A5A777',
 '#128A7D','#9B83BF','#7353A2','#0B4C78','#78B8A9','#C4B1D5','#A697BF','#673C8F'),bar_order)
bc <- sm %>% filter(dimension==32,method %in% bar_order) %>%
 group_by(graph,source) %>% mutate(best=auprc==max(auprc)) %>% ungroup() %>%
 mutate(method=factor(method,levels=bar_order),source=factor(src[source],levels=source_levels),
 graph=factor(graph,levels=c('broad','strict'),labels=c('Broad relations','Target-related relations')))
stopifnot(nrow(bc)==130,all(is.finite(bc$ap_sd)),all(bc$ap_sd>=0))
bp <- position_dodge(width=.88)
plot_c <- ggplot(bc,aes(source,auprc,fill=method,group=method))+
 geom_col(position=bp,width=.84)+
 geom_errorbar(aes(ymin=auprc-ap_sd,ymax=auprc+ap_sd),position=bp,width=.22,linewidth=.18,colour='#39434B')+
 geom_point(aes(y=auprc+ap_sd+.025,alpha=best),position=bp,shape=25,size=1.1,fill='#17212E',colour='#17212E',stroke=.15,show.legend=FALSE)+
 scale_alpha_manual(values=c('TRUE'=1,'FALSE'=0),guide='none')+
 scale_fill_manual(values=bar_cols,breaks=bar_order,labels=labels[bar_order],drop=FALSE)+
 facet_grid(~graph)+
 scale_y_continuous(limits=c(0,.84),breaks=c(0,.2,.4,.6,.8),expand=c(0,0))+
 scale_x_discrete(expand=expansion(add=.5))+
 labs(x=NULL,y='AUPRC',fill=NULL,tag='c')+
 theme_fig(7.2)+theme(panel.grid.major.x=element_blank(),panel.grid.major.y=element_line(colour='#E6ECF2',linewidth=.15),
 axis.text.x=element_text(angle=20,hjust=1,size=6),axis.text.y=element_text(size=6),
 strip.background=element_blank(),strip.text=element_text(size=7.2,face='bold'),panel.spacing=unit(4,'mm'),
 legend.position='top',legend.text=element_text(size=6),legend.key.width=unit(3,'mm'),legend.key.height=unit(2,'mm'),
 legend.spacing.x=unit(2,'mm'),legend.margin=margin(0,0,3,0))+
 guides(fill=guide_legend(nrow=3,byrow=TRUE))

# Preserve the original data geometry; provide sufficient physical panel area.
legend_plot <- plot_e + guides(colour=guide_legend(order=2,nrow=1,override.aes=list(linewidth=.75,alpha=1))) +
 theme(legend.direction="horizontal",legend.position="bottom",legend.box="vertical",legend.spacing.y=unit(1,"pt"),legend.margin=margin(0,0,0,0),legend.box.spacing=unit(0,"pt"))
lg <- ggplotGrob(legend_plot)
ids <- grep("guide-box",lg$layout$name)
ids <- ids[!vapply(lg$grobs[ids],inherits,logical(1),"zeroGrob")]
legend_panel <- wrap_elements(full=lg$grobs[[ids[1]]])
plot_e <- plot_e + theme(legend.position="none")
plot_a_top <- plot_a_top + scale_x_continuous(limits=c(.5,11.9),breaks=1:10,expand=c(0,0))
plot_a_bottom <- plot_a_bottom + scale_x_continuous(limits=c(.5,11.9),breaks=1:10,labels=paste0('I',1:10),expand=c(0,0))
plot_a <- plot_a_top / plot_a_bottom + plot_layout(heights=c(.58,.42))
plot_c <- plot_c + facet_wrap(~graph,ncol=1) + theme(legend.margin=margin(0,0,2,0))
figure4 <- ((plot_a|plot_b)+plot_layout(widths=c(1.1,1))) / plot_c + plot_layout(heights=c(79,101))
supp9 <- (plot_e+labs(tag=NULL)) / legend_panel + plot_layout(heights=c(168,12))
supp <- heat(32,'auroc','a')/heat(64,'auprc','b')/heat(64,'auroc','c')+plot_layout(heights=c(1,1,1))
export <- function(p,name,h){
 prefix <- file.path(figure_dir,name)
 svglite::svglite(paste0(prefix,'.svg'),width=183/25.4,height=h/25.4,bg='white')
 print(p);invisible(dev.off())
 converter <- Sys.getenv('RSVG_CONVERT',Sys.which('rsvg-convert'))
 stopifnot(file.exists(converter))
 convert <- function(ext,dpi=NULL,suffix=''){
  args <- c(paste0('--format=',ext),'--output',shQuote(paste0(prefix,suffix,'.',ext)))
  if(!is.null(dpi)) args <- c(args,'--dpi-x',dpi,'--dpi-y',dpi)
  status <- system2(converter,c(args,shQuote(paste0(prefix,'.svg'))),env='LD_LIBRARY_PATH=')
  stopifnot(status==0)
 }
 convert('pdf');convert('png',600);convert('png',180,'_preview')
 raster <- png::readPNG(paste0(prefix,'.png'))
 ragg::agg_tiff(paste0(prefix,'.tiff'),width=ncol(raster),height=nrow(raster),units='px',res=600,compression='lzw',background='white')
 grid::grid.raster(raster,interpolate=FALSE);invisible(dev.off())
}
export(figure4,'Figure_4',180)
export(supp,'Supplementary_Figure_S8',185)
export(supp9,'Supplementary_Figure_S9',180)
writeLines(capture.output(sessionInfo()),file.path(figure_dir,'sessionInfo.txt'))
