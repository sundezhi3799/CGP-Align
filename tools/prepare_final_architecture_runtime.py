"""Stage the versioned submission-architecture implementation without altering historical scripts."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = ROOT / 'revision_candidates/final_architecture_20260920'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError('Use a new directory outside this repository')
    hashes = json.loads((VERSION / 'code_hashes.json').read_text())
    for src in (VERSION / 'training_code').glob('*.py'):
        assert hashlib.sha256(src.read_bytes()).hexdigest() == hashes['scripts/' + src.name], src
    out.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    for folder in ('scripts', 'tools', 'cgp_align', 'manifests'):
        shutil.copytree(ROOT / folder, out / folder, ignore=ignore)
    shutil.copytree(VERSION / 'submission_model', out / 'submission_model', ignore=ignore)
    for src in (VERSION / 'training_code').glob('*.py'):
        shutil.copyfile(src, out / 'scripts' / src.name)
    (out / 'RUNTIME_VERSION.json').write_text(json.dumps({
        'version': 'final_architecture_20260920',
        'source': str(VERSION),
        'scope': 'model training and evaluation runtime; prepared data and weights must be supplied',
    }, indent=2))
    print('Staged submission-architecture runtime:', out)

if __name__ == '__main__':
    main()
