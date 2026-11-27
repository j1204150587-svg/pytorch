"""Add step markers only in a staged copy for a diagnostic (timings invalid)."""
import argparse
from pathlib import Path


def mark(source):
    original = source.read_text(encoding="utf-8")
    anchor = "            for step, (batch_size, sequence_length) in enumerate(shapes):\n"
    if original.count(anchor) != 1:
        raise ValueError("expected one dynamic eager loop")
    prefix = """            import os as _diag_os, time as _diag_time, json as _diag_json
            from pathlib import Path as _DiagPath
            _diag_dir = _DiagPath(_diag_os.environ['NPU_CACHE_DIAG_DIR'])
            _diag_file = _diag_dir / f'markers-{_diag_os.getpid()}.tsv'
            _diag_libs = {row.split()[-1] for row in _DiagPath('/proc/self/maps').read_text().splitlines()
                          if any(x in row for x in ('/libnnopbase.so', '/libopapi.so', '/libtorch_npu.so', '/cache_audit.so'))}
            (_diag_dir / f'fingerprint-{_diag_os.getpid()}.json').write_text(_diag_json.dumps({
                'worker_pid': _diag_os.getpid(), 'mapped_libraries': sorted(_diag_libs),
                'environment': {k: _diag_os.environ.get(k) for k in ('ACLNN_CACHE_LIMIT', 'LD_AUDIT', 'TASK_QUEUE_ENABLE')},
            }, indent=2))
            def _cache_mark(step_id, batch, seq):
                with open('/proc/self/status') as f:
                    rss = {k: int(v.split()[0]) for k, v in (line.split(':', 1) for line in f) if k in ('VmRSS', 'VmHWM')}
                with _diag_file.open('a') as f:
                    print(_diag_time.monotonic_ns(), step_id, batch, seq, rss['VmRSS'], rss['VmHWM'], sep='\\t', file=f)
"""
    staged = original.replace(anchor, prefix + anchor + "                _cache_mark(step, batch_size, sequence_length)\n")
    end = "        end = time.perf_counter()\n"
    if staged.count(end) != 1:
        raise ValueError("expected one synchronized window end")
    staged = staged.replace(end, end + "        _cache_mark(len(shapes), 0, 0)\n")
    compile(staged, str(source), "exec")
    source.write_text(staged, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staged_train", type=Path)
    mark(parser.parse_args().staged_train)
