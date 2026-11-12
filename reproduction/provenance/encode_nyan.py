import os
os.environ.update(CUDA_VISIBLE_DEVICES='',TF_CPP_MIN_LOG_LEVEL='3',OMP_NUM_THREADS='2')
import sys,json,time,logging,warnings,hashlib,argparse
ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);args=ap.parse_args()
from pathlib import Path
ROOT=Path('/data3/sdz/projects/CGP-Align/revisions_R1_4_20261110');OUT=ROOT/'nyan';OUT.mkdir(exist_ok=True)
OFF=Path('/home/sdz/projects/TMDiffusion/third_party/nyan_official');sys.path.insert(0,str(OFF))
import numpy as np,tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
tf.random.set_seed(41)
from spektral.data import Dataset,Graph,BatchLoader
import loading
from data import BondType
from model import NyanEncoder,NyanDecoder,VAE
warnings.filterwarnings('ignore');logging.getLogger('pysmiles').setLevel(logging.CRITICAL)
class Data(Dataset):
 def __init__(self,graphs):self.graphs_source=graphs;super().__init__()
 def read(self):return self.graphs_source
smiles=json.loads((ROOT/'features/fig3_smiles.json').read_text())
encoder=NyanEncoder(latent_dim=64,batched=True);model=VAE(encoder,NyanDecoder(fingerprint_bits=679,regression=1613))
status=model.load_weights(str(OFF/'saves/ZINC-extmodel5hk-3M')).expect_partial();encoder.output_names=['z_mean','z_log_var','z']
def encode(seq):
 graphs=[]
 for s in seq:
  raw=loading.get_data(str(s),only_biggest=True,unknown_atom_is_dummy=True)
  x,a,e=loading.convert(*raw,bonds=[BondType.SINGLE,BondType.DOUBLE,BondType.TRIPLE,BondType.AROMATIC,BondType.NOT_CONNECTED])
  e=np.asarray(e,np.float32)
  if e.size==0:e=np.zeros((len(x),len(x),5),dtype=np.float32)
  graphs.append(Graph(x=np.asarray(x,np.float32),a=np.asarray(a,np.float32),e=e,y=np.array(0)))
 loader=BatchLoader(Data(graphs),batch_size=1,epochs=1,mask=True,shuffle=False,node_level=False)
 mu,_,_=encoder.predict(loader.load(),steps=len(seq),verbose=0)
 status.assert_existing_objects_matched();assert np.isfinite(mu).all();return mu
# Verify posterior mean against the existing toxicity cache before new inference.
prior=Path('/data3/sdz/projects/TCM-Safety/toxric30_nyan_20261013/results/nyan_features.npz')
with np.load(prior) as z:probe=encode(z['smiles'][:16]);err=float(abs(probe-z['NYAN_mean'][:16]).max())
assert err<1e-5,err
(OUT/'verification.json').write_text(json.dumps({'existing_cache_max_abs':err,'encoder':'official frozen 64D posterior mean','batch_size':1}))
parts=[]
for start in range(0,len(smiles),512):
 if (start//512)%args.shards!=args.shard:continue
 p=OUT/f'chunk_{start:06d}.npy';seq=smiles[start:start+512]
 if p.exists():x=np.load(p)
 else:
  x=encode(seq)
  with p.with_suffix('.tmp').open('wb') as stream:np.save(stream,x)
  p.with_suffix('.tmp').replace(p)
 assert x.shape==(len(seq),64) and np.isfinite(x).all();parts.append(x)
 (OUT/f'status_{args.shard}.json').write_text(json.dumps({'last_completed_start':start,'total':len(smiles),'time':time.time()}))
 print('ENCODED',start+len(seq),len(smiles),flush=True)
(OUT/f'shard_{args.shard}_complete.json').write_text(json.dumps({'shard':args.shard,'shards':args.shards}))
