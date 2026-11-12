source(file.path(dirname(sub('^--file=','',grep('^--file=',commandArgs(),value=TRUE)[1])),'base.R'))
summary_dir <- file.path(out_root,'..','summary')
ep <- read_csv(file.path(summary_dir,'tox30_all_endpoint_metrics.csv'))
methods <- c('RDKit','NYAN_mean','ChemBERTa','MoLFormer','CGP','NYAN_mean_RDKit','ChemBERTa_RDKit','MoLFormer_RDKit','CGP_RDKit')
labels <- c('RDKit','NYAN','ChemBERTa','MoLFormer','CGP-Align','NYAN + RDKit','ChemBERTa + RDKit','MoLFormer + RDKit','CGP-Align + RDKit')
names(labels)<-methods
cols <- setNames(c('#9AA6B2','#128A7D','#9B83BF','#7353A2','#0B4C78','#78B8A9','#C4B1D5','#A697BF','#673C8F'),methods)
ids <- unique(read_csv(file.path(src_out,'Supplementary_Table_9.csv'))$target)
ids <- c(ids[16:30],ids[1:15])
heat30 <- function(endpoints,metric='AP'){
 dd<-ep %>% filter(target %in% endpoints) %>% group_by(target) %>% mutate(best=.data[[metric]]==max(.data[[metric]])) %>% ungroup() %>% mutate(target=factor(target,levels=endpoints),method=factor(method,levels=rev(methods)))
 ggplot(dd,aes(target,method,fill=.data[[metric]]))+geom_tile(width=.96,height=.94)+
 geom_tile(data=filter(dd,best),fill=NA,width=.96,height=.94,colour='#202020',linewidth=.45)+
 geom_text(aes(label=sprintf('%.2f',.data[[metric]]),fontface=ifelse(best,'bold','plain'),colour=.data[[metric]]>.65),size=1.8)+
 scale_colour_manual(values=c('FALSE'='#17212E','TRUE'='white'),guide='none')+
 scale_fill_gradientn(colours=c('#F2F6FA','#9ECAE1','#0B4C78'),limits=c(0,1),name=if(metric=='AP') 'AUPRC' else 'AUROC',breaks=c(0,.5,1))+
 scale_x_discrete(labels=setNames(sub('^[^_]+_','',endpoints),endpoints),expand=c(0,0))+
 scale_y_discrete(labels=labels,expand=c(0,0))+labs(x=NULL,y=NULL)+
 theme(axis.text.x=element_text(angle=40,hjust=1,size=6),axis.text.y=element_text(size=6),axis.ticks=element_blank(),axis.line=element_blank(),panel.grid=element_blank(),legend.position='right',legend.key.height=unit(4,'mm'),legend.key.width=unit(2,'mm'),legend.text=element_text(size=5.8),legend.title=element_text(size=6))
}
bars30 <- function(endpoints,show_legend=FALSE){
 dd<-ep %>% filter(target %in% endpoints) %>% group_by(target) %>% mutate(best=AP==max(AP)) %>% ungroup() %>% mutate(target=factor(target,levels=endpoints),method=factor(method,levels=methods))
 pos<-position_dodge(width=.88)
 ggplot(dd,aes(target,AP,fill=method,group=method))+
 geom_col(position=pos,width=.84,show.legend=show_legend)+
 geom_point(aes(y=AP+.025,alpha=best),position=pos,shape=25,size=1.1,fill='#17212E',stroke=.15,colour='#17212E',show.legend=FALSE)+
 scale_alpha_manual(values=c('TRUE'=1,'FALSE'=0),guide='none')+
 scale_fill_manual(values=c(cols,spacer='transparent'),limits=c(methods,'spacer'),breaks=c(methods[1:5],'spacer',methods[6:9]),labels=c(labels,spacer=''),drop=FALSE)+
 scale_x_discrete(labels=setNames(sub('^[^_]+_','',endpoints),endpoints),expand=expansion(add=.5))+
 scale_y_continuous(limits=c(0,1.08),breaks=c(0,.5,1),expand=c(0,0))+
 labs(x=NULL,y='AUPRC',fill=NULL)+guides(fill=guide_legend(nrow=2,byrow=TRUE))+
 theme(axis.text.x=element_text(angle=40,hjust=1,size=5.8),axis.text.y=element_text(size=6),panel.grid.major.x=element_blank(),panel.grid.major.y=element_line(colour='#E6ECF2',linewidth=.2),legend.position='top',legend.text=element_text(size=6),legend.key.size=unit(2,'mm'),legend.key.width=unit(3,'mm'),legend.spacing.x=unit(4,'mm'),legend.key.spacing.x=unit(3,'mm'),legend.margin=margin(0,0,3,0))
}

distribution <- function(metric='AP'){
 dd<-ep %>% mutate(method=factor(method,levels=methods));mm<-dd %>% group_by(method) %>% summarise(value=mean(.data[[metric]]),.groups='drop')
 ggplot(dd,aes(method,.data[[metric]],fill=method))+geom_boxplot(width=.46,outlier.shape=NA,alpha=.35,linewidth=.28,staplewidth=.45)+
 geom_point(aes(colour=method),position=position_jitter(width=.12,height=0,seed=41),size=.6,alpha=.5,show.legend=FALSE)+
 geom_point(data=mm,aes(method,value),inherit.aes=FALSE,shape=23,fill='white',size=1.6,stroke=.3)+
 geom_text(data=mm,aes(method,1.05,label=sprintf('%.3f',value)),inherit.aes=FALSE,size=2.05)+
 scale_fill_manual(values=cols,guide='none')+scale_colour_manual(values=cols,guide='none')+
 scale_x_discrete(labels=setNames(gsub(' + ',' +\n',labels,fixed=TRUE),methods))+
 scale_y_continuous(limits=c(0,1.1),breaks=c(0,.25,.5,.75,1),expand=c(0,0))+
 labs(x=NULL,y=paste('Endpoint',if(metric=='AP') 'AUPRC' else 'AUROC'))+
 theme(axis.text.x=element_text(size=6,lineheight=.9),panel.grid.major.x=element_blank(),panel.grid.major.y=element_line(colour='#E6ECF2',linewidth=.2))
}
paired <- function(metric='AP'){
 wide<-ep %>% select(target,method,all_of(metric)) %>% pivot_wider(names_from=method,values_from=all_of(metric))
 dd<-wide %>% pivot_longer(c(NYAN_mean_RDKit,ChemBERTa_RDKit,MoLFormer_RDKit),names_to='comparator',values_to='value') %>% mutate(comparator=factor(comparator,levels=c('NYAN_mean_RDKit','ChemBERTa_RDKit','MoLFormer_RDKit')),group=ifelse(grepl('Endocrine Disruption_SR-',target),'Stress response','Other endpoints'))
 ggplot(dd,aes(value,CGP_RDKit))+geom_abline(slope=1,intercept=0,linetype='dashed',colour='#9AA6B2',linewidth=.3)+
 geom_point(aes(shape=group,colour=group),size=1.25,alpha=.85)+facet_wrap(~comparator,nrow=1,labeller=as_labeller(labels))+
 scale_shape_manual(values=c('Stress response'=17,'Other endpoints'=16))+scale_colour_manual(values=c('Stress response'='#0B4C78','Other endpoints'='#9AA6B2'))+
 scale_x_continuous(limits=c(0,1),breaks=c(0,.5,1))+scale_y_continuous(limits=c(0,1),breaks=c(0,.5,1))+coord_fixed()+
 labs(x=paste('Comparator',if(metric=='AP') 'AUPRC' else 'AUROC'),y=paste('CGP-Align + RDKit',if(metric=='AP') 'AUPRC' else 'AUROC'),colour=NULL,shape=NULL)+
 theme(panel.grid=element_blank(),legend.position='bottom',legend.text=element_text(size=6),legend.key.height=unit(2,'mm'),legend.key.width=unit(3,'mm'),legend.margin=margin(0,0,0,0),legend.box.spacing=unit(0,'mm'),axis.title=element_text(size=6.5),axis.text=element_text(size=6),strip.text=element_text(size=6.5),panel.spacing.x=unit(12,'mm'))
}
five<-read_csv(file.path(summary_dir,'tox5_all_endpoint_metrics.csv')) %>% mutate(endpoint=sub('Endocrine Disruption_','',task),endpoint=factor(endpoint,levels=endpoint_order),method=factor(method,levels=rev(c('CGP-Align','RDKit2D','NYAN','ChemBERTa','MoLFormer','AtomPair','Avalon','Morgan')))) %>% group_by(endpoint) %>% mutate(best=auprc==max(auprc)) %>% ungroup()
focus<-ggplot(five,aes(endpoint,method,fill=auprc))+geom_tile(width=.94,height=.94)+geom_tile(data=filter(five,best),fill=NA,width=.94,height=.94,colour='black',linewidth=.5)+
 geom_text(aes(label=sprintf('%.3f',auprc),fontface=ifelse(best,'bold','plain'),colour=auprc>.55),size=1.95)+scale_colour_manual(values=c('FALSE'='#17212E','TRUE'='white'),guide='none')+
 scale_fill_gradientn(colours=c('#F2F6FA','#9ECAE1','#0B4C78'),limits=c(0,.8),name='AUPRC')+labs(x=NULL,y=NULL)+
 theme(axis.text.x=element_text(angle=35,hjust=1,size=6),axis.text.y=element_text(size=6.3),axis.ticks=element_blank(),axis.line=element_blank(),panel.grid=element_blank(),legend.position='right',legend.key.height=unit(4,'mm'),legend.key.width=unit(2,'mm'),legend.text=element_text(size=6),legend.title=element_text(size=6))
mean_bars <- function(){
 mm<-ep %>% group_by(method) %>% summarise(value=mean(AP),sd=sd(AP),n=n(),.groups='drop') %>% mutate(method=factor(method,levels=rev(methods)))
 write.csv(mm,file.path(out_dir,'Figure5c_mean_sd.csv'),row.names=FALSE)
 ggplot(mm,aes(value,method,fill=method))+geom_col(width=.62,show.legend=FALSE)+
 geom_errorbar(aes(xmin=value-sd,xmax=value+sd),orientation='y',width=.22,linewidth=.3,colour='#333333')+
 geom_text(aes(x=value+sd+.025,label=sprintf('%.3f',value)),hjust=0,size=2.0)+
 scale_fill_manual(values=cols,guide='none')+scale_y_discrete(labels=labels)+
 scale_x_continuous(limits=c(0,1.08),breaks=c(0,.5,1),expand=c(0,0))+
 labs(x='Mean AUPRC',y=NULL)+theme(axis.text.y=element_text(size=6),axis.text.x=element_text(size=6),axis.title.x=element_text(size=6.5),panel.grid.major.y=element_blank(),panel.grid.major.x=element_line(colour='#E6ECF2',linewidth=.2))
}
tagged<-function(p,i)p+labs(tag=letters[i],title=NULL,subtitle=NULL)+theme(plot.tag=element_text(size=10,face='bold'),plot.margin=margin(4,6,4,6))
compact_pairs<-paired()+theme(panel.spacing.x=unit(2.5,'mm'),strip.text=element_text(size=5.5),axis.title=element_text(size=6),axis.text=element_text(size=5.5),legend.text=element_text(size=5.5))
panels<-list(bars30(ids[1:15],TRUE),bars30(ids[16:30]),mean_bars(),compact_pairs,focus,p_d)
fig<-wrap_plots(lapply(seq_along(panels),function(i)tagged(panels[[i]],i)),design='AAAAAAAAAA\nBBBBBBBBBB\nCCCCDDDDDD\nEEEEEFFFFF',heights=c(1.3,1.05,1.15,1.05))
export<-function(p,name,w=183,h=215){
 prefix<-file.path(out_dir,name);svglite::svglite(paste0(prefix,'.svg'),width=w/25.4,height=h/25.4);print(p);dev.off()
 converter<-Sys.getenv('RSVG_CONVERT', Sys.which('rsvg-convert'))
 for(fmt in c('pdf','png')){opts<-c(paste0('--format=',fmt),'--output',shQuote(paste0(prefix,'.',fmt)));if(fmt=='png')opts<-c(opts,'--dpi-x','300','--dpi-y','300');stopifnot(system2(converter,c(opts,shQuote(paste0(prefix,'.svg'))),env='LD_LIBRARY_PATH=')==0)}
}
export(fig,'Figure_5')
export(tagged(distribution('AUROC'),1)/tagged(paired('AUROC'),2),'Supplementary_Figure_S6',183,130)
export(tagged(heat30(ids[1:15],'AUROC'),1)/tagged(heat30(ids[16:30],'AUROC'),2),'Supplementary_Figure_S7',183,135)
stopifnot(nrow(ep)==270,nrow(five)==40,all(table(ep$method)==30))
write.csv(ep,file.path(out_dir,'Figure5_30_endpoint_metrics.csv'),row.names=FALSE)
write.csv(five,file.path(out_dir,'Figure5_five_endpoint_metrics.csv'),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out_dir,'sessionInfo.txt'))
