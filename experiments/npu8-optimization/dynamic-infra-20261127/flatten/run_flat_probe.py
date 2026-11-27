"""Run the 8-shape native #41 applicability diagnostic under the existing host card lock."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


def insert_once(code, anchor, value):
    if code.count(anchor) != 1:
        raise ValueError(f"unexpected source at {anchor}")
    return code.replace(anchor, value + anchor)


def prepare(source, stage, probe_directory="flatten", probe_module="inspect_flat"):
    for item in source.iterdir():
        if item.is_file() and item.suffix in {".py", ".json", ".patch"}:
            shutil.copyfile(item, stage / item.name)
    p = stage / "run_npu8.py"
    code = p.read_text()
    if "DYNAMIC_INFRA_RECIPE" not in code:
        raise ValueError("expected adapted runner")
    hook = '''    if getattr(args, "mode", None) == "no-profiler":
        shutil.copyfile(HERE.parent / "flatten" / "inspect_flat.py", workspace / "inspect_flat.py")
        entry = workspace / "train_deepspeed.py"
        code = entry.read_text()
        loop = "            for step, (batch_size, sequence_length) in enumerate(shapes):\\n"
        require(code.count(loop) == 1, "missing dynamic eager training loop")
        code = code.replace(loop, "            from inspect_flat import install as _install_flat_probe\\n"
                            "            _flat_probe = _install_flat_probe(engine.optimizer, rank_dir)\\n" + loop +
                            "                _flat_probe.update(step=step, shape=[batch_size, sequence_length])\\n")
        entry.write_text(code)
'''
    hook = hook.replace('"flatten"', repr(probe_directory)).replace("inspect_flat", probe_module)
    code = insert_once(code, '    for source_file in workspace.glob("*.py"):', hook)
    compile(code, str(p), "exec")
    p.write_text(code)
    p = stage / "run_suite.py"
    code = insert_once(p.read_text(), "    print(f'WAIT {label}', flush=True)\n",
                       '    command += ["--warmup", "2", "--measure", "6"]\n')
    compile(code, str(p), "exec")
    p.write_text(code)


def main():
    study = Path(__file__).resolve().parents[1]
    stage = Path(tempfile.mkdtemp(prefix="flat41-probe-runner-", dir=study))
    prepare(study / "runner", stage)
    recipe = stage / "flat41-probe.json"
    # The views package retains the separately switchable native #41 and #43.
    common = json.loads((study / "unflatten.json").read_text())["common"]
    common.update(NPU_EXPERIMENT_FLAT_ND="1", NPU_EXPERIMENT_UNFLATTEN_VIEWS="0")
    recipe.write_text(json.dumps(dict(common=common, A={}, B={}), indent=2))
    os.environ["DYNAMIC_INFRA_RECIPE"] = str(recipe)
    sys.path.insert(0, str(stage))
    from run_suite import run_pair
    record = run_pair("NATIVE", "no-profiler", "baseline")
    (stage / "result.json").write_text(json.dumps(record, indent=2))
    if record["exit"]:
        return record["exit"]
    leg = Path(record["directory"]) / "01-A" / "rank000"
    rows = [json.loads(line) for line in (leg / "flatten-probe.jsonl").read_text().splitlines()]
    info, calls = rows[0], rows[1:]
    if {c["step"] for c in calls} != set(range(8)) or not all(c["native_flatten_scopes"] for c in calls):
        raise ValueError(f"incomplete flatten coverage in {leg}")
    print(json.dumps(dict(directory=str(leg), marker_available=info["marker_available"], calls=len(calls),
                          hits=sum(c["native_fast_path_scopes"] for c in calls))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
