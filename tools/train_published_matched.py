"""Retrain a comparator with its archived recipe and released prepared inputs."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'scripts'),str(ROOT)]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--artifacts',type=Path,required=True)
 p.add_argument('--matched-artifacts',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--seed',type=int,choices=[31,37,41],required=True)
 p.add_argument('--method',choices=['morgan','molformer_xl','chemberta'],required=True)
 p.add_argument('--device',default='cuda:0')
 p.add_argument('--dry-run',action='store_true')
 a=p.parse_args()
 if not (ROOT/'RUNTIME_VERSION.json').exists():p.error('Stage the final runtime first')
 base=a.matched_artifacts.resolve()/'matched';out=a.output.resolve()
 for inputs in [a.artifacts.resolve(),a.matched_artifacts.resolve()]:
  if out==inputs or inputs in out.parents or out in inputs.parents:p.error('Use a separate output tree')
 out.mkdir(parents=True,exist_ok=False)
 from run_strict_seed import command_from_config
 from evaluate_published_matched import link_or_copy
 prepared=out/'data'
 shutil.copytree(base/f'seed{a.seed}/data'/a.method,prepared,copy_function=link_or_copy)
 link_or_copy(base/f'features/{a.method}.npy',prepared/'compound_structure_features.npy')
 link_or_copy(base/f'seed{a.seed}/data/profile_features.npy',prepared/'profile_features.npy')
 c=json.loads((base/f'models/seed{a.seed}'/a.method/'recipe.json').read_text(encoding='utf-8'))
 assert c['seed']==a.seed and c['lambda_protein_profile']==0 and c['epochs']==160
 c.update(data_dir=str(prepared),intrinsic_split_path=str(prepared/'splits_intrinsic_entity.json'),protein_embedding_dir=str(a.artifacts.resolve()/'protein'),checkpoint_dir=str(out/'checkpoints'),log_dir=str(out/'logs'),device=a.device)
 command=command_from_config('train_cgp_align_base_intrinsic',c)
 (out/'portable_recipe.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
 (out/'command.json').write_text(json.dumps(command,indent=2),encoding='utf-8')
 if not a.dry_run:
  with (out/'training.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
 print('Prepared' if a.dry_run else 'Completed',a.seed,a.method)

if __name__=='__main__':main()
