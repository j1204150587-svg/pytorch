"""Apply the observed NPULauncher physical block cap without changing logical work."""
from pathlib import Path
import sys


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"unexpected source near {old}")
    return source.replace(old, new)


def stage(ws):
    path = ws / "fold_probe.py"
    src = path.read_text()
    mark = '    return plan, backend_call\n'
    src = replace(src, mark, '''    enabled = os.getenv("NPU_DYNAMIC_FOLD_GRID_CAP") == "1"
    known_driver = (witness["launcher_source_cache_hash"] ==
                    "911eed289b4d5b9773e4aaaef768c08cb28b28acfc8a493ca9e338bdc9a14e05"
                    and witness["auto_map_physical_cap"] == 40
                    and witness["driver_py_sha256"] ==
                    "b39118053062f3fc6d7f1047f8c9eb229de47a2b6106d8119cc7087f5b38818e")
    capped = getattr(extension, "_npu_hstu_fast_launch_with_physical_cap", None)
    report["verified_physical_cap"] = bool(enabled and known_driver and callable(capped))
    if report["verified_physical_cap"]:
        def launch(plan, stream, x, y, z, tensors):
            if x * y * z > 40:
                return capped(plan, stream, x, y, z, tensors, 40)
            return backend_call(plan, stream, x, y, z, tensors)
        return plan, launch
''' + mark)
    mark = '    physical_cap = report["driver_witness"]["auto_map_physical_cap"]\n'
    src = replace(src, mark, mark + '    if report.get("verified_physical_cap"):\n        physical_cap = None\n')
    compile(src, str(path), "exec")
    path.write_text(src)
    path = ws / "eager_fold_plan.py"
    if not path.is_file():
        return
    src = path.read_text()
    mark = '    elif INFO["driver_witness"]["auto_map_physical_cap"] is not None and grid > INFO["driver_witness"]["auto_map_physical_cap"]:'
    src = replace(src, mark, mark.replace('elif INFO[', 'elif not INFO.get("verified_physical_cap") and INFO[', 1))
    mark = '    row["route"] = "cpp_returned"\n'
    src = replace(src, mark, mark + '''    cap = INFO["driver_witness"]["auto_map_physical_cap"]
    row["physical_blocks"] = min(grid, cap) if cap else grid
    if cap and grid > cap:
        TOTAL["cpp_capped_returned"] += 1
''')
    compile(src, str(path), "exec")
    path.write_text(src)


if __name__ == "__main__":
    stage(Path(sys.argv[1]))
