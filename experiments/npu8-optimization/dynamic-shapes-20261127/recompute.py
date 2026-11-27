"""Recompute paired dynamic-shape results from the copied raw metrics and CSVs."""
import argparse
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def shape_sequence(metrics):
    return [(row['batch_size'], row['seq_len']) for row in metrics['steps']]


def steady(metrics, shapes):
    record = metrics['summary']['steady_state']
    measured = shapes[record['start_step_inclusive']:record['end_step_exclusive']]
    samples = sum(batch for batch, _ in measured)
    tokens = sum(batch * length for batch, length in measured)
    assert (len(measured), samples, tokens) == (record['steps'], record['samples'], record['tokens'])
    elapsed = record['elapsed_s_synchronized']
    return dict(steps=len(measured), samples=samples, tokens=tokens, elapsed_s=elapsed,
                ms_per_step=elapsed * 1000 / len(measured), samples_per_s=samples / elapsed,
                tokens_per_s=tokens / elapsed)


def level0(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        rows = list(csv.DictReader(source))
    assert [int(row['Step']) for row in rows] == [4, 5, 6]
    totals = {key: sum(float(row[column]) for row in rows) for key, column in (
        ('stage', 'Stage'), ('free', 'Free'), ('computing', 'Computing'),
        ('unoverlapped_communication', 'Communication(Not Overlapped)'))}
    return {**{f'{key}_ms': total / 3000 for key, total in totals.items()},
            'free_ratio': totals['free'] / totals['stage']}


def recompute(evidence):
    output = []
    for row in read(evidence / 'runs.json'):
        item = dict(row)
        output.append(item)
        if row['exit'] != 0:
            continue
        pair = evidence / row['label']
        assert read(pair / 'status.json')['exit'] == 0
        a, = pair.glob('*-A')
        b, = pair.glob('*-B')
        a_metrics, b_metrics = (read(leg / 'rank000/metrics.json') for leg in (a, b))
        shapes = shape_sequence(a_metrics)
        assert shapes == shape_sequence(b_metrics) and len(shapes) == len(set(shapes))
        a_command, b_command = (read(leg / 'command.json') for leg in (a, b))
        assert set(a_command['env']) == set(b_command['env'])
        changed = {key for key, value in a_command['env'].items() if b_command['env'][key] != value}
        assert changed == {f'RECSYS_DYNAMIC_{row["variant"].upper()}'}
        if row['audit']:
            a_state, b_state = (read(leg / 'rank000/state_audit.json') for leg in (a, b))
            distinct_loss = any(x['loss'] != y['loss'] for x, y in zip(a_metrics['steps'], b_metrics['steps'], strict=True))
            assert (a_state == b_state and not distinct_loss) == row['audit_equal']
        elif row['mode'] == 'no-profiler':
            old, new = steady(a_metrics, shapes), steady(b_metrics, shapes)
            item.update(a=old, b=new, saved_ms_per_step=old['ms_per_step']-new['ms_per_step'],
                        step_gain=1-new['ms_per_step']/old['ms_per_step'])
        else:
            old, new = (level0(leg / 'step_trace_time.csv') for leg in (a, b))
            item.update(a=old, b=new, saved_free_ms=old['free_ms']-new['free_ms'],
                        free_ratio_gain=1-new['free_ratio']/old['free_ratio'])
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    args = parser.parse_args()
    print(json.dumps(recompute(args.evidence), indent=2))
