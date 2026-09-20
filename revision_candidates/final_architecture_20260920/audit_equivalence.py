import sys, json, hashlib
from pathlib import Path
ROOT=Path('/data3/sdz/cgp_align_final_arch_code_20260920')
sys.path[:0]=[str(ROOT/'submission_model'),str(ROOT/'scripts')]
import torch
import torch.nn.functional as F
from cgp_align.model import CGPAlign
from cgp_align.losses import cgp_align_objective
from cgp_align import graph as sg
import train_cgp_align_replicate as tr
import cgp_gnn as tg
torch.set_num_threads(4)
torch.manual_seed(123)
a=CGPAlign()
b=tr.ReplicateCGPAlign(profile_dim=3180,protein_dim=2560,embed_dim=256,gnn_hidden_dim=256,gnn_layers=6,gene_hidden_dims=(512,),profile_hidden_dims=(512,),modality_context_dim=32,dropout=.2,profile_source_adapters=False,profile_source_embedding=True,profile_source_dropout=0,num_profile_sources=3,profile_input_layernorm=False,profile_feature_dropout=0,profile_mlp_norm=False,split_gene_modality_branches=True)
def rename(k):return k.replace('orf_gene_encoder.','gene_encoder.')
b.load_state_dict({rename(k):v for k,v in a.state_dict().items()},strict=True)
result={'state_dict_strict_match':True,'checks':[]}
def check(label,x,y):
    torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6)
    result['checks'].append({'name':label,'max_abs_error':float((x-y).abs().max())})
smiles=['CCO','c1ccccc1','[Na+]','C[C@H](N)C(=O)O']
sa=sg.batch_graphs([sg.graph_from_smiles(s) for s in smiles],torch.device('cpu'))
sb=tg.batch_graphs([tg.graph_from_smiles(s) for s in smiles],torch.device('cpu'))
for i,(x,y) in enumerate(zip(sa,sb)): check('graph_tensor_'+str(i),x,y)
px=torch.randn(8,3180); gx=torch.randn(8,2560)
mods=torch.tensor([0,1,0,1,0,1,0,1]); sources=torch.tensor([0,1,2,0,1,2,0,1])
for training in [False,True]:
 a.train(training);b.train(training)
 for name,fa,fb in [('compound',lambda:a.compound_encoder(*sa),lambda:b.compound_encoder(*sb)),('gene',lambda:a.encode_gene(gx,mods+1),lambda:b.encode_gene(gx,mods)),('profile',lambda:a.encode_profile(px,sources),lambda:b.encode_profile(px,sources))]:
  torch.manual_seed(91); x=fa();torch.manual_seed(91);y=fb();check(name+('_train' if training else '_eval'),x,y)
 # Distinct parameter names are mapped explicitly; compare end-to-end gradients.
 a.zero_grad();b.zero_grad()
 torch.manual_seed(92);loss_a=a.encode_gene(gx,mods+1).square().sum()+a.encode_profile(px,sources).sum()+a.compound_encoder(*sa).sum()
 torch.manual_seed(92);loss_b=b.encode_gene(gx,mods).square().sum()+b.encode_profile(px,sources).sum()+b.compound_encoder(*sb).sum()
 loss_a.backward();loss_b.backward()
 grads=dict(b.named_parameters())
 maxerr=0.
 for k,p in a.named_parameters():
  q=grads[rename(k)];assert p.grad is not None and q.grad is not None
  torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6)
  maxerr=max(maxerr,float((p.grad-q.grad).abs().max()))
 result['checks'].append({'name':'all_parameter_gradients_'+str(training),'max_abs_error':maxerr})
codes=torch.tensor([0,0,1,2,2,3,4,4])
z=[F.normalize(torch.randn(8,256),dim=1).requires_grad_() for _ in range(6)]
la=cgp_align_objective(z[0],z[1],codes,codes,z[2],z[3],codes,codes,z[4],z[5],codes,codes)['total']
lb=tr.multipositive_contrastive_loss(z[0],z[1],codes,codes,.07)
gl,_,_=tr.modality_balanced_contrastive_loss_parts(torch.cat([z[2],z[4]]),torch.cat([z[3],z[5]]),codes.repeat(2),codes.repeat(2),torch.tensor([0]*8+[1]*8),torch.tensor([0]*8+[1]*8),.07,reduction='sum')
lb=lb+gl;check('three_term_loss_repeated_entities',la,lb)
for i,(ga,gb) in enumerate(zip(torch.autograd.grad(la,z,retain_graph=True),torch.autograd.grad(lb,z))):check('loss_gradient_'+str(i),ga,gb)
result['linear_layers']={name:[[m.in_features,m.out_features] for m in module.modules() if isinstance(m,torch.nn.Linear)] for name,module in [('orf',b.gene_encoder),('crispr',b.crispr_gene_encoder),('profile',b.profile_encoder),('compound_projection',b.compound_encoder.projection)]}
result['compound_initializers']='Not reused: legacy projection contains LayerNorm absent from submission model'
result['data_audits']=[]
for seed in (31,37,41):
 for kind in ('compound','gene'):
  p=Path(f'/data3/sdz/cgp_align_strict_20260918/seed{seed}/data/{kind}/profile_correction_audit.json')
  c=json.loads(p.read_text());assert c['seed']==seed and c['fit_val_overlap']==c['fit_test_overlap']==0
  result['data_audits'].append(c)
result['status']='PASS'
(ROOT/'equivalence_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'status':'PASS','checks':len(result['checks']),'linear_layers':result['linear_layers']},indent=2))
