import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.decomposition import PCA
R=Path('/data3/sdz/cgp_align_downstream_20260920');D=R/'embeddings/seed41';O=R/'figure4';I=R/'inputs'
c=pd.read_parquet(D/'compound_entities.parquet');g=pd.read_parquet(D/'gene_entities.parquet');zc=np.load(D/'compound.npy');zg=np.load(D/'gene.npy')
ci={}
for i,row in c.iterrows():
 for k in ['compound_id','inchikey','entity_id']:
  if k in row:ci[str(row[k])]=i
groups={}
for i,s in enumerate(g.gene_symbol.fillna('').str.upper()):
 if s:groups.setdefault(s,[]).append(i)
gv={s:zg[ix].mean(0) for s,ix in groups.items()};gv={s:v/max(np.linalg.norm(v),1e-8) for s,v in gv.items()}
coords=pd.read_csv(I/'figure4e_merged_database_entity_space_pca_coordinates.csv');valid=[r.entity_id in (ci if r.entity_type=='Compound' else gv) for r in coords.itertuples()];missing=coords.loc[np.logical_not(valid),['entity_type','entity_id']].to_dict('records');coords=coords.loc[valid].copy();z=np.vstack([zc[ci[r.entity_id]] if r.entity_type=='Compound' else gv[r.entity_id] for r in coords.itertuples()]);pca=PCA(n_components=2,svd_solver='full');xy=pca.fit_transform(z.astype(np.float64));coords['pca_x']=xy[:,0];coords['pca_y']=xy[:,1];coords['pc1_var']=pca.explained_variance_ratio_[0];coords['pc2_var']=pca.explained_variance_ratio_[1]
links=pd.read_csv(I/'figure4e_merged_database_entity_space_pca_links.csv');edges=pd.read_csv(I/'compound_target_edges.csv.gz');edges=edges[edges.compound_id.isin(ci)&edges.gene_id.str.upper().isin(gv)].copy();edges['score']=[float(zc[ci[r.compound_id]]@gv[r.gene_id.upper()]) for r in edges.itertuples()]
for i,r in links.iterrows():
 a=coords[(coords.entity_type=='Compound')&coords.entity_id.eq(r.compound_id)].iloc[0];b=coords[(coords.entity_type=='Gene')&coords.entity_id.eq(r.gene_symbol)].iloc[0];score=float(zc[ci[r.compound_id]]@gv[r.gene_symbol]);sub=edges[edges.source_database.fillna('').str.split('|').apply(lambda x:r.source in x)];rank=int((sub.score>score).sum()+1)
 links.loc[i,['compound_x','compound_y','gene_x','gene_y','compound_gene_cosine','rank','n_edges','similarity_percentile','rank_label']]=[a.pca_x,a.pca_y,b.pca_x,b.pca_y,score,rank,len(sub),100*(1-(rank-.5)/len(sub)),f'#{rank}/{len(sub)}']
 links.loc[i,'pair_id']='Reference '+str(r.pair_rank)
coords.to_csv(O/'figure4e_merged_database_entity_space_pca_coordinates.csv',index=False);links.to_csv(O/'figure4e_merged_database_entity_space_pca_links.csv',index=False)
(O/'pca_provenance.json').write_text(json.dumps({'seed':41,'selection':'same reference pairs and entity universe as preceding figure, restricted to current available entities','missing_entities':missing,'entity_counts':coords.entity_type.value_counts().to_dict(),'pc_variance':pca.explained_variance_ratio_.tolist()},indent=2));print('PCA complete missing',len(missing))

from collections import Counter
names={'dgidb':'DGIdb','drugrep':'DrugRep','hetionet':'Hetionet','openbiolink':'OpenBioLink','pharmebinet':'PharMeBINet'}
sets=edges.source_database.fillna('').map(lambda s:tuple(sorted(names[k] for k in set(s.split('|'))&set(names))))
counts=Counter(x for x in sets if x);top=sorted(counts.items(),key=lambda x:(-x[1],x[0]))[:10]
inter=[];matrix=[]
for i,(members,n) in enumerate(top,1):
 pattern='&'.join(members);inter.append(dict(pattern_id=i,pattern_label=f'I{i}',pattern=pattern,degree=len(members),intersection_edges=n))
 for name in names.values():matrix.append(dict(pattern_id=i,pattern=pattern,name=name,active=name in members))
pd.DataFrame(inter).to_csv(O/'figure4a_relation_source_upset_intersections_main.csv',index=False)
pd.DataFrame(matrix).to_csv(O/'figure4a_relation_source_upset_matrix_main.csv',index=False)
pd.DataFrame([dict(name=name,set_edges=sum(n for members,n in counts.items() if name in members)) for name in names.values()]).to_csv(O/'figure4a_relation_source_upset_set_sizes_main.csv',index=False)
pd.DataFrame([dict(source_file='compound_target_edges.csv.gz',relation_universe='mapped compound-gene edge universe',selected_sources='; '.join(names.values()),n_edges_in_selected_sources=sum(counts.values()),n_top_intersections_plotted=10)]).to_csv(O/'figure4a_relation_source_upset_metadata_main.csv',index=False)
print('A mapped edges',sum(counts.values()))
