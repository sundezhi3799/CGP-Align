"""Change only scheduling to forked workers; keep scientific code/cache protocol intact."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
source=(ROOT/'run_tox30.py').read_text(encoding='utf-8-sig')
old='''    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending=[pool.submit(train,job) for job in jobs]
        for future in as_completed(pending):
            parts.append(future.result());completed+=1'''
new='''    import multiprocessing
    global TRAIN
    TRAIN=train
    with multiprocessing.get_context('fork').Pool(a.workers) as pool:
        for result in pool.imap_unordered(dispatch, jobs, chunksize=1):
            parts.append(result);completed+=1'''
assert source.count(old)==1
runtime=source.replace(old,new).replace('def main():','def dispatch(job):\n    return TRAIN(job)\n\ndef main():',1)
(ROOT/'execution_processes.json').write_text(json.dumps({'original_code_sha256':hashlib.sha256(source.encode()).hexdigest(),'executed_code_sha256':hashlib.sha256(runtime.encode()).hexdigest(),'scheduling_only_change':True,'replacement_before':old,'replacement_after':new,'workers':sys.argv[sys.argv.index('--workers')+1] if '--workers' in sys.argv else 8},indent=2))
(ROOT/'run_tox30_executed_source.txt').write_text(runtime)
# Make dispatch importable by fork workers in this __main__ module.
__file__=str(ROOT/'run_tox30.py')
exec(compile(runtime,__file__,'exec'),globals())
