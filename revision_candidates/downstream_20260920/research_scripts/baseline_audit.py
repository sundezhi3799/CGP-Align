from pathlib import Path
import pandas as pd,json,hashlib,shutil
R=Path(__file__).resolve().parent;P=Path('public_repository/CGP-Align');raw=Path('output/cgp_align/paper/raw_results/tox_np_mechanism')
current=pd.read_csv(R/'figure5/random_cgp_folds.csv');checks=[]
for folder in ['toxric22_fixed_extratrees_representation_benchmark','check_sr_atad5_extratrees_representation_benchmark','sr5_fixed_extratrees_with_nyan_latent']:
 d=raw/folder;meta=json.loads((d/'toxric_representation_model_metadata.json').read_text());assert meta['n_estimators']==160 and meta['seed']==41 and meta['n_splits']==5
 old=pd.read_csv(d/'toxric_representation_model_folds.csv');old=old[old.task.isin(current.task)&old.model.eq('extratrees')]
 for t in old.itertuples():
  q=current[(current.task==t.task)&(current.fold==t.fold)].iloc[0]
  for k in ['n_train','n_test','positive_train','positive_test']:assert q[k]==getattr(t,k)
 checks.append({'source':str(d),'matching_fold_rows':len(old),'n_estimators':160,'seed':41,'fold_count':5})
sc=Path('output/cgp_align/paper/manuscript_cgp_align_profile/supplementary_experiments/sr5_scaffold_split_v1');old=pd.read_csv(sc/'sr5_scaffold_fold_metrics.csv');new=pd.read_csv(R/'figure5/scaffold_cgp_folds.csv');common=[k for k in ['n_train','n_test','positive_train','positive_test'] if k in old and k in new]
for t in old.itertuples():
 q=new[(new.task==t.task)&(new.fold==t.fold)]
 if len(q):
  for k in common:assert q.iloc[0][k]==getattr(t,k)
out={'random_folds':checks,'scaffold_rows':len(old),'scaffold_count_columns_verified':common,'scope':'same loader, task cohorts and seed; archived fold train/test and positive counts verified; historical per-row fold hashes were not stored','baseline_tables_reused':True}
(R/'figure5/baseline_cohort_audit.json').write_text(json.dumps(out,indent=2));shutil.copy2(R/'figure5/baseline_cohort_audit.json',P/'revision_candidates/downstream_20260920/figure5/baseline_cohort_audit.json');print(json.dumps(out,indent=2))
