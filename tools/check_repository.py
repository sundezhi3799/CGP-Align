"""Check syntax, imported-file provenance and tracked-file checksums (standard library)."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'manifests/release_checksums.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def files():
    for p in sorted(ROOT.rglob('*')):
        rel = p.relative_to(ROOT)
        if not p.is_file() or p == MANIFEST:
            continue
        if any(x in {'.git', '.venv', '__pycache__', '.pytest_cache'} for x in rel.parts):
            continue
        if rel.parts[:2] == ('paper_snapshot', 'results') or rel.parts[0] in {'output', 'outputs', 'data', 'checkpoints'}:
            continue
        if p.suffix in {'.pyc', '.pyo'}:
            continue
        yield p

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-checksums', action='store_true', help='Regenerate after intentional edits.')
    args = parser.parse_args()
    inventory = list(files())
    for path in inventory:
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path.relative_to(ROOT)))
    with (ROOT / 'manifests/source_provenance.csv').open(encoding='utf-8', newline='') as f:
        provenance = list(csv.DictReader(f))
    for row in provenance:
        path = ROOT / row['destination']
        if sha(path) != row['copied_sha256']:
            raise SystemExit('Imported file changed; record intentional modifications: ' + row['destination'])
    current = {p.relative_to(ROOT).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in inventory}
    if args.write_checksums:
        MANIFEST.write_text(json.dumps(current, indent=2) + '\n', encoding='utf-8')
    elif not MANIFEST.exists() or json.loads(MANIFEST.read_text(encoding='utf-8')) != current:
        raise SystemExit('Checksum inventory missing or changed. Review edits before regenerating.')
    print(json.dumps({'status': 'PASS', 'files': len(current), 'unchanged_imports': len(provenance),
                      'scope': 'syntax, file integrity and provenance; not full scientific reproduction'}, indent=2))

if __name__ == '__main__':
    main()
