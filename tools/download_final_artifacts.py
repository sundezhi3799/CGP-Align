"""Download, SHA256-verify and safely extract the final-architecture release."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://github.com/sundezhi3799/CGP-Align/releases/download/final-architecture-20260920/'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

class Parts(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None
    def readable(self): return True
    def read(self, size=-1):
        if size < 0: raise ValueError('Streaming reader requires a bounded read')
        out = bytearray()
        while len(out) < size:
            if self.current is None:
                try: self.current = next(self.paths).open('rb')
                except StopIteration: break
            block = self.current.read(size - len(out))
            if block: out.extend(block)
            else: self.current.close(); self.current = None
        return bytes(out)
    def close(self):
        if self.current: self.current.close()
        super().close()

def extract_group(group, cache, output):
    expected = {x['path']: x for x in group['files']}
    seen = set()
    with Parts([cache / p['name'] for p in group['parts']]) as stream:
        with tarfile.open(fileobj=stream, mode='r|gz') as archive:
            for member in archive:
                if not member.isfile() or member.name not in expected:
                    raise ValueError('Unexpected archive member: ' + member.name)
                target = (output / member.name).resolve()
                if output.resolve() not in target.parents: raise ValueError('Unsafe archive path')
                meta = expected[member.name]
                if member.size != meta['bytes'] or member.name in seen: raise ValueError('Invalid archive member')
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.partial')
                h = hashlib.sha256()
                with archive.extractfile(member) as src, temporary.open('wb') as dst:
                    for b in iter(lambda: src.read(8 * 1024 * 1024), b''):
                        h.update(b); dst.write(b)
                if h.hexdigest() != meta['sha256']: raise ValueError('Content checksum mismatch: ' + member.name)
                temporary.replace(target); seen.add(member.name)
    if seen != set(expected): raise ValueError('Archive incomplete')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, help='Defaults to OUTPUT/downloads; retains verified parts')
    p.add_argument('--groups', nargs='+', default=['final-models', 'protein-features', 'prepared-seed41'])
    p.add_argument('--manifest', type=Path, default=ROOT / 'manifests/final_artifacts_20260920.json')
    p.add_argument('--offline', action='store_true', help='Verify and extract already downloaded parts only')
    p.add_argument('--verify-only', action='store_true', help='Verify extracted files without downloading')
    a = p.parse_args(); cache = a.cache or a.output / 'downloads'
    manifest = json.loads(a.manifest.read_text(encoding='utf-8-sig'))
    groups = {g['name']: g for g in manifest['groups']}
    for name in a.groups:
        if name not in groups: p.error('Unknown group: ' + name)
    a.output.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    for name in a.groups:
        group = groups[name]
        if not a.verify_only:
            for part in group['parts']:
                path = cache / part['name']
                if not path.exists() or path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
                    if a.offline: raise ValueError('Missing or invalid cached part: ' + str(path))
                    tmp = path.with_name(path.name + '.partial')
                    print('Downloading', part['name'], flush=True)
                    with urllib.request.urlopen(URL + part['name'], timeout=180) as src, tmp.open('wb') as dst:
                        for b in iter(lambda: src.read(8 * 1024 * 1024), b''): dst.write(b)
                    if tmp.stat().st_size != part['bytes'] or digest(tmp) != part['sha256']: raise ValueError('Download checksum mismatch')
                    tmp.replace(path)
            extract_group(group, cache, a.output)
        for f in group['files']:
            path = a.output / f['path']
            if not path.exists() or path.stat().st_size != f['bytes'] or digest(path) != f['sha256']:
                raise ValueError('Extracted file checksum mismatch: ' + f['path'])
        print('Verified', name, len(group['files']), 'files', flush=True)

if __name__ == '__main__': main()
