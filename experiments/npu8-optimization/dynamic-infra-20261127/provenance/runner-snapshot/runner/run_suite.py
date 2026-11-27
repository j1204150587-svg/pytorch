"""Run the dynamic-shape ZeRO-2 controls serially on host88 logical NPU 8."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
LOCK = Path('/run/lock/recsys-gr-ds019-npu8.lock')
VARIANTS = ('PARTITION', 'NORM', 'BOTH')


def file_arg(option, name, hash_option=None):
    path = HERE / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return [f'--{option}', str(path), f'--{hash_option or option}-sha256', digest]


def wait_for_card_lock():
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)


def run_pair(variant, mode, order, audit=False):
    label = f'{variant.lower()}-{mode}-{order}' + ('-audit' if audit else '')
    command = [sys.executable, str(HERE / 'run_npu8.py'), 'run', '--stage', order, '--mode', mode,
               '--approval-ref', f'dynamic-shapes-20261127-{label}',
               '--extension-switch', f'RECSYS_DYNAMIC_{variant}']
    for option, name in [('candidate-patch', 'candidate.patch'), ('candidate-module', 'overflow.py'),
                         ('extension-patch', 'extension.patch'), ('extension-module', 'graphed_hstu.py'),
                         ('shape-schedule', 'dynamic_shapes_npu8.json')]:
        command += file_arg(option, name)
    command += file_arg('shared-norm-module', 'zero2_batch_norm.py', 'shared-norm')
    if audit:
        command += ['--warmup', '2', '--measure', '6', '--audit-state']
    print(f'WAIT {label}', flush=True)
    wait_for_card_lock()
    while True:
        print(f'RUN {label}', flush=True)
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        print(result.stdout, end='', flush=True)
        if result.stderr:
            print(result.stderr, end='', file=sys.stderr, flush=True)
        if result.returncode != 10:
            break
        wait_for_card_lock()
    lines = result.stdout.splitlines()
    run = Path(lines[0]) if lines and lines[0].startswith('/data/') else None
    entry = {'label': label, 'variant': variant.lower(), 'mode': mode, 'order': order, 'audit': audit,
             'exit': result.returncode, 'directory': str(run) if run else None}
    if result.returncode == 0 and audit:
        left, right = run / '01-A/rank000', run / '02-B/rank000'
        a = json.loads((left / 'state_audit.json').read_text())
        b = json.loads((right / 'state_audit.json').read_text())
        a_loss = [step['loss'] for step in json.loads((left / 'metrics.json').read_text())['steps']]
        b_loss = [step['loss'] for step in json.loads((right / 'metrics.json').read_text())['steps']]
        entry.update(audit_equal=a == b and a_loss == b_loss,
                     audited_steps=len(a['gradients_and_norms']),
                     different_state_keys=[key for key in a if a[key] != b.get(key)],
                     loss_max_delta=max(abs(x-y) for x, y in zip(a_loss, b_loss, strict=True)))
    print(json.dumps(entry), flush=True)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    entries = []

    def record(entry):
        entries.append(entry)
        pending = args.out.with_suffix('.tmp')
        pending.write_text(json.dumps(entries, indent=2) + '\n')
        pending.replace(args.out)

    accepted = []
    for variant in VARIANTS:
        entry = run_pair(variant, 'no-profiler', 'ab', audit=True)
        record(entry)
        if entry.get('audit_equal') and entry['audited_steps'] == 8:
            accepted.append(variant)
    if not args.audit_only:
        for variant in accepted:
            for mode, order in (('no-profiler', 'ab'), ('no-profiler', 'ba'), ('l0', 'ab')):
                record(run_pair(variant, mode, order))
    expected = len(VARIANTS) * (1 if args.audit_only else 4)
    return int(len(entries) != expected or any(entry['exit'] for entry in entries) or len(accepted) != len(VARIANTS))


if __name__ == '__main__':
    sys.exit(main())
