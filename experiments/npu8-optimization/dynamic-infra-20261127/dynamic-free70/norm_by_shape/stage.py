"""Compare redundant wide conversions and shape batching in separate training pairs."""
from pathlib import Path
import shutil
import sys


def stage(workspace):
    module = workspace / "wide_norm.py"
    source = module.read_text()
    init = '    if mode=="off":\n        return\n'
    call = "            result = original(full, groups)"
    if source.count(init) != 1 or source.count(call) != 1:
        raise ValueError("expected the existing wide norm stage")
    source = source.replace(init, init + '''    from norm_by_shape import ParamNormGroups, norm_without_casts
    selected = os.environ.get("NPU_DYNAMIC_NORM_LAYOUT", "original")
    if selected not in ("original","nocast","groups") or mode != "join":
        raise ValueError("requires the wide join prerequisite")
    seen.update(layout=selected, original_calls=0, native_width_calls=0,
                gathered_calls=0, layout_fallback=0, groups_per_call=0, wide_dtype=None)
    cached = None
''')
    source = source.replace('    def wide(flat, plan):\n', '    def wide(flat, plan):\n        nonlocal cached\n')
    source = source.replace(call, '''            seen["wide_dtype"] = str(full.dtype)
            if selected == "original":
                result = original(full, groups)
                seen["original_calls"] += 1
                seen["groups_per_call"] = len(groups)
            elif full.device.type != "npu" or full.dtype != torch.float32:
                seen["layout_fallback"] += 1
                result = original(full, groups)
            elif selected == "nocast":
                result = norm_without_casts(full, groups)
                seen["native_width_calls"] += 1
                seen["groups_per_call"] = len(groups)
            else:
                if cached is None or cached[0] is not plan or cached[1] != full.device:
                    cached = (plan, full.device, ParamNormGroups(plan, full.device))
                result = cached[2](full)
                seen["gathered_calls"] += 1
                seen["groups_per_call"] = len(cached[2].layout)
''')
    compile(source, str(module), "exec")
    module.write_text(source)
    shutil.copyfile(Path(__file__).with_name("norm_by_shape.py"), workspace / "norm_by_shape.py")


if __name__ == "__main__":
    stage(Path(sys.argv[1]))
