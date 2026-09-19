"""Reevaluate one seed's four frozen models, preserving the historical run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from prepare_matched_benchmark import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-run', type=Path, required=True)
    p.add_argument('--strict-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=(31, 37, 41), required=True)
    a = p.parse_args()
    output = a.output_root / f'seed{a.seed}'
    output.mkdir(parents=True, exist_ok=False)
    previous = a.previous_run / f'seed{a.seed}'
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    status = dict(seed=a.seed, status='running', protocol='strict_entity_latent_mean_v1',
                  protocol_selected_after_test_sensitivity_analysis=True,
                  previous_run=str(previous), weights_unchanged=True)
    try:
        for method in ('cgp', 'morgan', 'molformer_xl', 'chemberta'):
            status['method'] = method
            (output / 'status.json').write_text(json.dumps(status, indent=2))
            metrics = json.loads((previous / 'eval' / method / 'metrics.json').read_text())
            assert sha(Path(metrics['checkpoint'])) == metrics['checkpoint_sha256']
            command = [sys.executable, str(Path(__file__).with_name('eval_matched_benchmark.py')),
                '--prepared', str(previous / 'data'), '--checkpoint', metrics['checkpoint'],
                '--method', method, '--output', str(output / method),
                '--aggregation', 'mean_encoded', '--replicate-data',
                str(a.strict_root / f'seed{a.seed}' / 'data/compound')]
            with (output / f'{method}.log').open('x') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        status['status'] = 'complete'
    except Exception as exc:
        status.update(status='failed', error=str(exc))
        raise
    finally:
        (output / 'status.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
