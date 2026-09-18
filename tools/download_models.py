"""Download exact historical checkpoints and verify their published SHA256 hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('checkpoints/primary'))
    parser.add_argument('--seed', type=int, choices=(31, 37, 41), action='append', help='Repeat to select seeds; default all three.')
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[1] / 'manifests/models.json'
    models = json.loads(manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        if args.seed and model['seed'] not in args.seed: continue
        path = args.output_dir / ('primary_seed' + str(model['seed']) + '.pt')
        if path.exists():
            if path.stat().st_size != model['bytes'] or digest(path) != model['sha256']:
                raise SystemExit('Existing file does not match the manifest; preserve or rename it: ' + str(path))
            print('Verified existing ' + str(path))
            continue
        url = model.get('public_url')
        if not url or not url.startswith('https://'):
            raise SystemExit('Public HTTPS artifact URL unavailable for seed ' + str(model['seed']))
        partial = path.with_suffix('.pt.download')
        request = urllib.request.Request(url, headers={'User-Agent': 'CGP-Align-model-downloader'})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as output:
            while block := response.read(1024 * 1024): output.write(block)
        if partial.stat().st_size != model['bytes'] or digest(partial) != model['sha256']:
            raise SystemExit('Downloaded artifact failed integrity checks: ' + str(partial))
        partial.replace(path)
        print('Downloaded and verified ' + str(path))
    print('Historical weights only: see docs/TRAINING_ARCHIVE_AUDIT.md before interpreting results.')


if __name__ == '__main__':
    main()
