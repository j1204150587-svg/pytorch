"""Run the dynamic-shape ZeRO-2 controls serially on host88 logical NPU 8."""
import argparse
import fcntl
import hashlib
import json
import math
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
    deadline = time.monotonic() + 900
    with LOCK.open('a') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'shared card lock stayed busy: {LOCK}')
                time.sleep(4)


def evaluate_audit(entry):
    run = Path(entry['directory'])
    left, right = run / '01-A/rank000', run / '02-B/rank000'
    a = json.loads((left / 'state_audit.json').read_text())
    b = json.loads((right / 'state_audit.json').read_text())
    a_loss = [step['loss'] for step in json.loads((left / 'metrics.json').read_text())['steps']]
    b_loss = [step['loss'] for step in json.loads((right / 'metrics.json').read_text())['steps']]
    states = list(zip(a['gradients_and_norms'], b['gradients_and_norms'], strict=True))
    loss_pairs = list(zip(a_loss, b_loss, strict=True))
    def fp32_close(x, y):
        return math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-6)

    gradient_equal = all(x['gradients'] == y['gradients'] for x, y in states)
    weights_and_adamw_equal = all(a[key] == b.get(key) for key in a if key != 'gradients_and_norms')
    norm_close = all(fp32_close(x['norm'], y['norm']) for x, y in states)
    loss_close = all(fp32_close(x, y) for x, y in loss_pairs)
    entry.update(audit_equal=a == b and a_loss == b_loss, audited_steps=len(states),
                 different_state_keys=[key for key in a if a[key] != b.get(key)],
                 loss_max_delta=max(abs(x-y) for x, y in loss_pairs),
                 norm_max_delta=max(abs(x['norm']-y['norm']) for x, y in states),
                 gradient_equal=gradient_equal, weights_and_adamw_equal=weights_and_adamw_equal,
                 audit_passed=gradient_equal and weights_and_adamw_equal and norm_close and loss_close)


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
        evaluate_audit(entry)
    print(json.dumps(entry), flush=True)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='reuse successful pairs and recheck saved audit evidence')
    args = parser.parse_args()
    entries = json.loads(args.out.read_text()) if args.resume and args.out.is_file() else []

    def record(entry):
        for index, previous in enumerate(entries):
            if previous['label'] == entry['label']:
                entries[index] = entry
                break
        else:
            entries.append(entry)
        pending = args.out.with_suffix('.tmp')
        pending.write_text(json.dumps(entries, indent=2) + '\n')
        pending.replace(args.out)

    def run_or_reuse(variant, mode, order, audit=False):
        label = f'{variant.lower()}-{mode}-{order}' + ('-audit' if audit else '')
        entry = next((old for old in entries if old['label'] == label and old['exit'] == 0), None)
        if entry is None:
            entry = run_pair(variant, mode, order, audit)
        elif audit:
            evaluate_audit(entry)
            print(json.dumps(entry), flush=True)
        record(entry)
        return entry

    accepted = []
    for variant in VARIANTS:
        entry = run_or_reuse(variant, 'no-profiler', 'ab', audit=True)
        if entry.get('audit_passed') and entry['audited_steps'] == 8:
            accepted.append(variant)
    if not args.audit_only:
        for variant in accepted:
            for mode, order in (('no-profiler', 'ab'), ('no-profiler', 'ba'), ('l0', 'ab')):
                run_or_reuse(variant, mode, order)
    expected = len(VARIANTS) * (1 if args.audit_only else 4)
    return int(len(entries) != expected or any(entry['exit'] for entry in entries) or len(accepted) != len(VARIANTS))


if __name__ == '__main__':
    sys.exit(main())
