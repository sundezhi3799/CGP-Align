"""Reevaluate Figure 3F/G from released final and matched artifacts."""
import argparse,os,shutil,subprocess,sys
from pathlib import Path

def link_or_copy(src,dst):
 try:os.link(src,dst)
 except OSError:shutil.copy2(src,dst)
 return dst

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--artifacts',type=Path,required=True)
 p.add_argument('--matched-artifacts',type=Path,required=True)
 p.add_argument('--seed',type=int,choices=[31,37,41],required=True)
 p.add_argument('--methods',nargs='+',choices=['cgp','morgan','molformer_xl','chemberta'],default=['cgp','morgan','molformer_xl','chemberta'])
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--device',default='cuda:0')
 a=p.parse_args();root=Path(__file__).resolve().parents[1]
 if not (root/'RUNTIME_VERSION.json').exists():p.error('Stage the final runtime before running this command')
 a.output.mkdir(parents=True,exist_ok=False)
 base=a.matched_artifacts.resolve()/'matched';source=base/f'seed{a.seed}/data';prepared=a.output.resolve()/'prepared'
 shutil.copytree(source,prepared,copy_function=link_or_copy)
 for method in ['morgan','molformer_xl','chemberta']:
  for src,dst in [(base/f'features/{method}.npy',prepared/method/'compound_structure_features.npy'),(prepared/'profile_features.npy',prepared/method/'profile_features.npy')]:
   try:os.link(src,dst)
   except OSError:shutil.copy2(src,dst)
 env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 for method in a.methods:
  ck=a.artifacts.resolve()/f'models/seed{a.seed}/joint/best_model.pt' if method=='cgp' else base/f'models/seed{a.seed}/{method}/best_model.pt'
  cmd=[sys.executable,str(root/'tools/eval_matched_benchmark.py'),'--prepared',str(prepared),'--checkpoint',str(ck),'--method',method,'--output',str(a.output.resolve()/method),'--aggregation','mean_encoded','--replicate-data',str(a.artifacts.resolve()/f'data/seed{a.seed}/compound'),'--artifacts',str(a.artifacts.resolve()),'--device',a.device]
  with (a.output/f'{method}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
  print('Evaluated',a.seed,method,flush=True)
if __name__=='__main__':main()
