"""Observe #43 on actual ZeRO unflatten arguments in an eight-shape diagnostic."""
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    study = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(study / "flatten"))
    from run_flat_probe import prepare
    stage = Path(tempfile.mkdtemp(prefix="unflatten-probe-runner-", dir=study))
    prepare(study / "runner", stage, probe_directory="unflatten_probe", probe_module="inspect_unflatten")
    config = json.loads((study / "unflatten.json").read_text())
    recipe = stage / "unflatten-probe.json"
    recipe.write_text(json.dumps(dict(common=config["common"], A=config["B"], B={}), indent=2))
    os.environ["DYNAMIC_INFRA_RECIPE"] = str(recipe)
    sys.path.insert(0, str(stage))
    from run_suite import run_pair
    record = run_pair("NATIVE", "no-profiler", "baseline")
    (stage / "result.json").write_text(json.dumps(record, indent=2))
    if record["exit"]:
        return record["exit"]
    output = Path(record["directory"]) / "01-A/rank000/unflatten-probe.jsonl"
    info, *rows = map(json.loads, output.read_text().splitlines())
    if {c["step"] for c in rows} != set(range(8)):
        raise ValueError(f"incomplete unflatten coverage: {output}")
    print(json.dumps(dict(output=str(output), registration=info["native_npu_registration"],
                          calls=len(rows), hits=sum(c["events"]["native_marker"] for c in rows))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
