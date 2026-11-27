"""Run one true-shape eager baseline or one-switch A/B family on the existing NPU 8 runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
FREE70 = HERE.parent
RUNNER_FILES = ("candidate.patch", "overflow.py", "extension.patch", "graphed_hstu.py", "zero2_batch_norm.py",
                "steady_window.patch", "dynamic_shapes_npu8.json", "evidence.py", "run_npu8.py", "run_suite.py")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(path)


def check(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validation_source(src):
    old = '''        require(norm == (variant == "B" and args.extension_switch in ("RECSYS_DYNAMIC_NORM", "RECSYS_DYNAMIC_BOTH")),
                "dynamic norm activation mismatch")
        require(partition == (variant == "B" and args.extension_switch in ("RECSYS_DYNAMIC_PARTITION", "RECSYS_DYNAMIC_BOTH")),
                "dynamic partition activation mismatch")'''
    new = '''        env = read_json(out / "command.json")["env"]
        for active, name in ((norm, "NORM"), (partition, "PARTITION")):
            expected = any(env.get(k) == "1" for k in ("RECSYS_DYNAMIC_" + name, "RECSYS_DYNAMIC_BOTH"))
            require(active == expected, "unexpected dynamic " + name + " activation")
        if norm:
            record = read_json(rank / "norm_stats.json")
            require(record["valid"] and record["calls"] == record["batched_calls"] == flags["steps"]
                    and not record["fallback_reasons"], "norm activation counts differ from steps")
        if partition:
            require(read_json(rank / "fast_partition.json")["partition"] == flags["steps"],
                    "partition activation counts differ from steps")'''
    check(src.count(old) == 1, "unrecognized dynamic activation validator")
    old_study = 'STUDY = "dynamic-infra-20261127"'
    check(src.count(old_study) == 1, "unrecognized stock runner destination")
    return src.replace(old, new).replace(old_study, 'STUDY = "dynamic-infra-20261127/dynamic-free70"')


def pipeline(work, stages):
    if not stages:
        return None
    if isinstance(stages,str):
        return (FREE70/stages).resolve(strict=True)
    check(isinstance(stages,list) and stages and len(set(stages))==len(stages),"expected ordered unique stages")
    bundle=work/"stage-pipeline"
    bundle.mkdir(exist_ok=True)
    for number,name in enumerate(stages):
        source=(FREE70/name).resolve(strict=True)
        target=bundle/f"stage_{number}"
        target.mkdir(exist_ok=True)
        for f in sorted(source.glob("*.py")):
            dest=target/f.name
            if dest.exists():
                check(dest.read_bytes()==f.read_bytes(),f"implementation changed: {f}; use a new experiment name")
            else:
                shutil.copyfile(f,dest)
    (bundle/"stage.py").write_text("""from pathlib import Path
import subprocess
import sys

if __name__=="__main__":
    root=Path(__file__).resolve().parent
    for stage in sorted(root.glob("stage_*"),key=lambda p:int(p.name.split("_")[1])):
        subprocess.run([sys.executable,"-B",str(stage/"stage.py"),sys.argv[1]],check=True)
""",encoding="utf-8")
    return bundle


def materialize_runner(source, dest, stage_source=None, endpoint=False):
    root = dest / "runner"
    root.mkdir(exist_ok=True)
    hashes = {}
    for name in RUNNER_FILES:
        origin = source / name
        data = validation_source(origin.read_text(encoding="utf-8")).encode() if name == "run_npu8.py" else origin.read_bytes()
        if stage_source and name=="run_npu8.py":
            src = data.decode()
            anchor = '            workspace, hashes = prepare(root / "workspace", session, args)\n'
            check(src.count(anchor)==1, "unrecognized workspace staging section")
            src = src.replace(anchor,anchor + '''            package = HERE / "stage_code"
            shutil.copytree(package, session / "inputs" / "stage_code")
            subprocess.run([sys.executable, "-B", str(package / "stage.py"), str(workspace)], check=True)
            hashes = files_sha256(workspace)
            write_sha256(session / "workspace.sha256", hashes)
''')
            anchor = '    write_json(out / "validated.json", result)\n'
            check(src.count(anchor)==1, "unrecognized result writer")
            src = src.replace(anchor,'''    from free70_validation import verify_activation
    verify_activation(read_json(out / "recipe.json"), variant, out / "rank000", flags["steps"])
''' + anchor)
            if endpoint:
                mark = '            package = HERE / "stage_code"\n'
                check(src.count(mark) == 1, "expected one candidate workspace")
                src = src.replace(mark, '''            plain = session / "workspace-01"
            shutil.copytree(workspace, plain)
            plain_hashes = files_sha256(plain)
            write_sha256(session / "workspace-01.sha256", plain_hashes)
''' + mark)
                mark = 'artifacts, workspace, hashes,\n                                container, init_pid, args, variant)'
                check(src.count(mark) == 1, "expected one measured pair dispatch")
                src = src.replace(mark, 'artifacts, plain if variant == "A" else workspace,\n'
                    '                                plain_hashes if variant == "A" else hashes,\n'
                    '                                container, init_pid, args, variant)')
                src = src.replace('    verify_activation(read_json(out / "recipe.json"), variant,',
                                  '    if variant == "B":\n        verify_activation(read_json(out / "recipe.json"), variant,')
            data = src.encode()
        target = root / name
        if target.exists():
            check(target.read_bytes() == data, f"measured runner or source changed: {target}")
        else:
            target.write_bytes(data)
        hashes[name] = digest(data)
    if stage_source:
        stage_copy = root / "stage_code"
        stage_copy.mkdir(exist_ok=True)
        for origin in sorted(stage_source.rglob("*.py")):
            relative = origin.relative_to(stage_source)
            target = stage_copy / relative
            target.parent.mkdir(exist_ok=True,parents=True)
            data = origin.read_bytes()
            if target.exists():
                check(target.read_bytes()==data,f"staged implementation changed: {origin}; use a new family")
            else:
                target.write_bytes(data)
            hashes["stage_code/"+relative.as_posix()] = digest(data)
        check((stage_copy/"stage.py").is_file(),"stage.py is missing")
        helper = HERE/"verify.py"
        target = root/"free70_validation.py"
        if target.exists():
            check(target.read_bytes()==helper.read_bytes(),"activation validator changed; use a new family")
        else:
            shutil.copyfile(helper,target)
        hashes[target.name] = digest(target.read_bytes())
    schedule = read(root / "dynamic_shapes_npu8.json")
    check(len(schedule) == len(set(map(tuple,schedule))) == 60
          and sum(b for b,n in schedule[10:]) == 136 and sum(b*n for b,n in schedule[10:]) == 2994,
          "expected the configured 60 unique dynamic shapes")
    save(dest / "runner-sha256.json", hashes)
    return root


def recipe_info(path):
    recipe = read(path)
    for k in ("common","A","B"):
        check(isinstance(recipe.get(k),dict) and all(isinstance(a,str) and isinstance(b,str)
              for a,b in recipe[k].items()), f"expected string environment map: {k}")
    check(not (recipe["common"].keys() & (recipe["A"].keys() | recipe["B"].keys())), "common keys overlap treatment")
    for env in (recipe["common"], recipe["A"], recipe["B"]):
        for k,v in env.items():
            check(k not in ("RECSYS_ZERO2_BATCH_OVERFLOW","TASK_QUEUE_ENABLE","RECSYS_DYNAMIC_STATE_AUDIT")
                  and "REPLAY" not in k and "GRAPH" not in k, f"fixed eager training setting: {k}")
    changed = [k for k in recipe["A"].keys() | recipe["B"].keys() if recipe["A"].get(k)!=recipe["B"].get(k)]
    if recipe.get("comparison") == "stock01-to-combination":
        check(recipe["common"] == {} and recipe["A"] == {"NPU_DYNAMIC_ENDPOINT": "0"}
              and recipe["B"].get("NPU_DYNAMIC_ENDPOINT") == "1",
              "endpoint A must be the original chunk-overflow-only baseline")
        changed = ["NPU_DYNAMIC_ENDPOINT"]
    check(len(changed) <= 1, "only one experimental switch may change")
    check(re.fullmatch("[a-zA-Z0-9_-]+", path.stem), "unsafe recipe name")
    return recipe, (changed[0] if changed else None)


def gate(work, runner, recipe, workspace):
    passed = work / "gate"
    if (passed / "comparison.json").is_file() and read(passed / "comparison.json")["passed"]:
        return
    check(workspace and workspace.is_dir(), "semantic gate needs --workspace or a prior baseline-mode run")
    attempt = Path(tempfile.mkdtemp(prefix="gate-attempt-", dir=work)) / "gate"
    original_workspace = workspace
    if recipe.get("stage"):
        code = runner/"stage_code"
        prior = workspace.parent/"inputs/stage_code"
        if prior.is_dir():
            before = {p.relative_to(prior):digest(p.read_bytes()) for p in prior.rglob("*.py")}
            after = {p.relative_to(code):digest(p.read_bytes()) for p in code.rglob("*.py")}
            check(before==after,"gate workspace was built with different candidate code")
        else:
            clean = attempt.parent/"staged-workspace"
            shutil.copytree(workspace,clean,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
            subprocess.run([sys.executable,"-B",str(code/"stage.py"),str(clean)],check=True)
            workspace = clean
    validator = STUDY/"semantic/run_semantic_pair.py"
    if recipe.get("comparison") == "stock01-to-combination":
        check(original_workspace != workspace, "endpoint needs a clean base workspace")
        script = validator.read_text()
        mark = "    stage(source, work)\n"
        check(script.count(mark) == 1 and script.count('inner(work),') == 1, "unexpected semantic runner")
        script = script.replace(mark, mark + '    plain = output / "workspace-01"\n'
            + '    stage(Path(' + repr(str(original_workspace)) + '), plain)\n'
            + '    write(output / "workspace-01.sha256.json", {str(p.relative_to(plain)): sha(p) '
            + 'for p in sorted(plain.rglob("*")) if p.is_file()})\n')
        script = script.replace("inner(work),", 'inner(plain if variant == "A" else work),')
        validator = attempt.parent/"endpoint_semantic.py"
        validator.write_text(script)
        shutil.copyfile(STUDY/"semantic/dynamic_semantic.py", attempt.parent/"dynamic_semantic.py")
    subprocess.run([sys.executable, str(validator),
                    "--workspace",str(workspace), "--runner-dir",str(runner),"--output",str(attempt),
                    "--common-env",json.dumps(recipe["common"]),"--env-a",json.dumps(recipe["A"]),
                    "--env-b",json.dumps(recipe["B"])], check=True)
    check(read(attempt/"comparison.json")["passed"],"semantic comparison failed")
    for leg in sorted(attempt.glob("0*-*")):
        env = read(leg/"command.json")["env"]
        rank = leg/"rank000"
        if recipe.get("activation") and (recipe.get("comparison") != "stock01-to-combination" or leg.name[-1] == "B"):
            from verify import verify_activation
            verify_activation(recipe,leg.name[-1],rank,8)
        for name, file in (("PARTITION","fast_partition.json"),("NORM","norm_stats.json")):
            active = env.get("RECSYS_DYNAMIC_"+name)=="1" or env.get("RECSYS_DYNAMIC_BOTH")=="1"
            check((rank/file).is_file()==active,f"{leg}: {name} activation")
            if active:
                rec = read(rank/file)
                if name=="PARTITION":
                    check(rec["partition"]==8,"semantic partition count")
                else:
                    check(rec["valid"] and rec["batched_calls"]==rec["calls"]==5 and not rec["fallback_reasons"],
                          "semantic norm update count")
    passed.symlink_to(attempt.relative_to(work),target_is_directory=True)


def main():
    import fcntl
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("recipe",type=Path)
    cli.add_argument("jobs",nargs="+", help="gate, l1/baseline, l0/baseline, no-profiler/baseline, or MODE/ab,ba,aa")
    cli.add_argument("--runner",type=Path,default=STUDY/"runner")
    cli.add_argument("--workspace",type=Path,help="existing staged dynamic eager workspace for the semantic gate")
    args = cli.parse_args()
    path = args.recipe.resolve(strict=True)
    recipe,switch = recipe_info(path)
    work = FREE70/"runs"/path.stem
    work.mkdir(parents=True,exist_ok=True)
    with (work/"family.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        spec = work/"recipe.json"
        if spec.exists():
            check(read(spec)==recipe, "recipe changed; use a new recipe name for a new implementation")
        else:
            save(spec,recipe)
        stage = pipeline(work,recipe.get("stage"))
        runner = materialize_runner(args.runner.resolve(strict=True),work,stage,
                                    recipe.get("comparison") == "stock01-to-combination")
        os.environ["DYNAMIC_INFRA_RECIPE"] = str(spec)
        sys.path.insert(0,str(runner))
        from run_suite import run_pair
        index = work/"runs.json"
        runs = read(index) if index.is_file() else []
        for job in args.jobs:
            if job=="gate":
                workspace = args.workspace
                if not workspace:
                    reference = next((r for r in runs if r["exit"]==0),None)
                    workspace = Path(reference["directory"])/"workspace" if reference else None
                gate(work,runner,recipe,workspace)
                continue
            mode,order = job.split("/")
            check(mode in ("l1","l0","no-profiler") and order in ("baseline","ab","ba","aa"), f"unknown job {job}")
            if order!="baseline":
                check(order=="aa" or switch is not None, "differential pair requires one changed switch")
                check((work/"gate/comparison.json").is_file() and read(work/"gate/comparison.json")["passed"],
                      "run gate before candidate pairs")
            if any(r["exit"]==0 and r["mode"]==mode and r["order"]==order for r in runs):
                continue
            result = run_pair("NATIVE",mode,order)
            runs.append(result)
            save(index,runs)
            if result["exit"]:
                raise SystemExit(result["exit"])


if __name__=="__main__":
    main()
