"""Rebuild Figure 3 and S1/S4/S5 in a fresh directory after equal-objective export."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rscript',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    f=a.output/'figure3';s=a.output/'supplement';f.mkdir();s.mkdir()
    shutil.copytree(a.artifacts/'figure3_sources',f/'source_data')
    shutil.copyfile(ROOT/'figures/figure3/reproduce.R',f/'reproduce.R')
    shutil.copytree(a.artifacts/'source_logs',s/'source_logs')
    for name in ('figureS1.R','figureS4.R','figureS5.R'): shutil.copyfile(ROOT/'figures/supplementary_strict'/name,s/name)
    def run(args): subprocess.run(args,cwd=ROOT,check=True)
    run([sys.executable,str(ROOT/'tools/build_figure3_matched_sources.py'),'--results',str(a.artifacts/'encoded'),'--output',str(f/'source_data')])
    run([sys.executable,str(ROOT/'tools/build_strict_supplement_sources.py'),'--source',str(a.artifacts/'figure3_sources'),'--figure-root',str(s),'--gene-reduction','sum'])
    renders=[]
    for script,folder,stem in [(f/'reproduce.R',f,'figure3')]+[(s/(x+'.R'),s,x) for x in ('figureS1','figureS4','figureS5')]:
        output=folder/'exports';log=folder/(stem+'_render.log')
        with log.open('w') as handle:
            result=subprocess.run([a.rscript,str(script),str(output)],stdout=handle,stderr=subprocess.STDOUT)
        files=[output/(stem+ext) for ext in ('.png','.pdf','.svg','.tiff')]
        text=log.read_text(errors='replace')
        if not all(x.exists() and x.stat().st_size>1000 for x in files) or 'Wrote' not in text:
            raise RuntimeError(f'Incomplete rendering: {log}')
        if result.returncode and ('Error' in text or 'Execution halted' in text):
            raise RuntimeError(f'Render error: {log}')
        renders.append(dict(figure=stem,exit_code=result.returncode,log=str(log),files=list(map(str,files))))
    (a.output/'status.json').write_text(json.dumps(dict(status='rendered_pending_visual_review',objective='compound + ORF + CRISPR',renders=renders),indent=2))
    print('Rendered new figures; visual review required before replacing manuscript figures.')
if __name__=='__main__':main()
