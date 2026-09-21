"""Recompute downstream analyses from versioned inputs and final CGP checkpoints."""
import argparse,json,os,sys,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--task',choices=['embeddings','relations','toxicity','alerts','prism'],required=True)
 p.add_argument('--inputs',type=Path,required=True)
 p.add_argument('--artifacts',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--seeds',type=int,nargs='+',default=[31,37,41])
 p.add_argument('--device',default='cuda:0')
 p.add_argument('--embeddings',type=Path,help='Optional released embedding cache, or a previously generated embedding directory')
 a=p.parse_args()
 if not (ROOT/'cgp_align/models/version.json').exists():p.error('Model configuration is missing; run from a complete CGP-Align checkout.')
 a.inputs=a.inputs.resolve();a.artifacts=a.artifacts.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
 a.embeddings=a.embeddings.resolve() if a.embeddings else a.output/'embeddings'
 os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
 sys.path.insert(0,str(ROOT/'scripts/downstream'))
 import portable_analysis
 getattr(portable_analysis,a.task)(a)
 (a.output/(a.task+'_COMPLETE.json')).write_text(json.dumps({'task':a.task,'seeds':a.seeds,'inputs':str(a.inputs),'artifacts':str(a.artifacts),'embeddings':str(a.embeddings)},indent=2))
if __name__=='__main__':main()
