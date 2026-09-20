"""Train final-architecture branches/joint models from released prepared inputs."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'scripts'),str(ROOT)]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='New directory, separate from released artifacts')
    p.add_argument('--seed',type=int,choices=[31,37,41],default=41)
    p.add_argument('--stage',choices=['all','joint','compound','orf','crispr'],default='joint')
    p.add_argument('--variant',choices=['full','no_source_indicator','no_pretraining'],default='full')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();base=a.artifacts.resolve();out=a.output.resolve()
    marker=ROOT/'RUNTIME_VERSION.json'
    if not marker.exists() or json.loads(marker.read_text())['version']!='final_architecture_20260920':
        p.error('Run this tool from a staged final-architecture runtime.')
    if out==base or base in out.parents or out in base.parents:p.error('Output must be separate from the artifacts tree.')
    if out.exists():p.error('Use a new output directory to preserve existing runs.')
    out.mkdir(parents=True)
    from run_strict_seed import command_from_config
    stages=['compound','orf','crispr','joint'] if a.stage=='all' else [a.stage]
    records=[]
    for branch in stages:
        c=json.loads((base/f'models/seed{a.seed}'/branch/'configuration.json').read_text())
        folder=out/branch
        c.update(run_name=branch,checkpoint_dir=str(folder/'checkpoints'),log_dir=str(folder/'logs'),output_dir=str(folder/'eval'),device=a.device,eval_only=False,train_only=False,smoke_test=False)
        if branch=='joint':
            module='train_cgp_align_replicate'
            c.update(compound_data_dir=str(base/f'data/seed{a.seed}/compound'),gene_data_dir=str(base/f'data/seed{a.seed}/gene'),gene_protein_embedding_dir=str(base/'protein'))
            c['init_gene_checkpoint']=None
            for name in ['compound','orf','crispr']:
                key='init_compound_checkpoint' if name=='compound' else f'init_{name}_gene_checkpoint'
                c[key]=str(out/name/'checkpoints'/name/'best_model.pt') if a.stage=='all' else str(base/f'models/seed{a.seed}'/name/'best_model.pt')
                if a.variant=='no_pretraining':c[key]=None
            c['disable_profile_source_embedding']=a.variant=='no_source_indicator'
        elif branch=='compound':
            module='train_compound_mocop_replicate';c['data_dir']=str(base/f'data/seed{a.seed}/compound')
        else:
            module='train_gene_mocop';c.update(data_dir=str(base/f'data/seed{a.seed}/gene'),protein_embedding_dir=str(base/'protein'))
        command=command_from_config(module,c);folder.mkdir(parents=True)
        (folder/'portable_configuration.json').write_text(json.dumps(c,indent=2))
        records.append({'stage':branch,'command':command})
        if not a.dry_run:
            with (folder/'run.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
    (out/'commands.json').write_text(json.dumps(records,indent=2))
    print('Prepared' if a.dry_run else 'Completed',len(records),'stage(s):',out)

if __name__=='__main__':main()
