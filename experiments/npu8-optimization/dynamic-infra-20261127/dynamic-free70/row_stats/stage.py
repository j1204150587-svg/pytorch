"""Attach the guarded row-moments eager operator to the flat-stats control."""
from pathlib import Path
import shutil
import sys


def replace(source, before, after):
    if source.count(before) != 1:
        raise ValueError(before)
    return source.replace(before, after)


def stage(workspace):
    path = workspace / "flat_stats.py"
    text = path.read_text()
    text = replace(text, '    total = sum(sizes)\n', '''    total = sum(sizes)
    moments = None
    if os.getenv("NPU_DYNAMIC_ROW_STATS") == "1":
        if mode != "both":
            raise ValueError("row moments requires both flat stats enabled")
        from rows import PartitionMoments
        moments = PartitionMoments(sizes, device)
''')
    text = replace(text, '            invalid = nonfinite(flat)', '''            if moments is None:
                invalid = nonfinite(flat)
            else:
                invalid = moments.scan(flat)
                records["row_scans"] = records.get("row_scans", 0) + 1''')
    text = replace(text, '            result=row_norm_square(flat,plan)', '''            result = moments.square(flat) if moments is not None else None
            if result is None:
                if moments is not None:
                    records["row_cache_fallback"] = records.get("row_cache_fallback", 0) + 1
                result = row_norm_square(flat,plan)
            else:
                records["row_norm_hits"] = records.get("row_norm_hits", 0) + 1''')
    compile(text, str(path), "exec")
    path.write_text(text)
    shutil.copyfile(Path(__file__).with_name("rows.py"), workspace / "rows.py")


if __name__ == "__main__":
    stage(Path(sys.argv[1]))
