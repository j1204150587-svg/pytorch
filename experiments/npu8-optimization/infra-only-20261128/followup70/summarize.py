"""Recompute a completed experiment and export the inputs used for its report."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

cli = argparse.ArgumentParser()
cli.add_argument("name")
cli.add_argument("--dest", type=Path, required=True)
args = cli.parse_args()
study = Path(__file__).resolve().parent.parent
out = args.dest / args.name
out.mkdir(exist_ok=True, parents=True)
def pair(mode,order):
    return study / "pairs" / f"{args.name}-{mode}-{order}"
def raw(mode, order):
    result = json.loads((pair(mode,order)/"result.json").read_bytes())
    assert result["exit"] == 0
    return Path(result["directory"])

for mode,order in (("audit","ab"),("l0","ab"),("l0","ba"),("no-profiler","ab"),("no-profiler","ba"),("l1","ab")):
    raw(mode,order)
cmd = [sys.executable,"-B",str(study/"runner/paired_metrics.py"),"--schedule",str(study/"archived-runner/dynamic_shapes_npu8.json")]
for order in ("ab","ba"):
    cmd += ["--round",str(pair("no-profiler",order)),str(pair("l0",order))]
with (out/"metrics.json").open("w") as fp:
    subprocess.run(cmd, stdout=fp, check=True)
parser = study / "queue/source-v2/analyze_dynamic_trace.py"
subprocess.run([sys.executable,"-B",str(study/"queue/source-v2/analyze_pair.py"),str(raw("l1","ab")),str(out/"level1"),
                "--analyzer",str(parser)],check=True)
spec=importlib.util.spec_from_file_location("export", study/"export.py")
export=importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)
with tarfile.open(out/"raw.tgz","w:gz",compresslevel=2,dereference=True) as arc:
    for mode,order in (("audit","ab"),("l0","ab"),("l0","ba"),("no-profiler","ab"),("no-profiler","ba"),("l1","ab")):
        p=pair(mode,order)
        arc.add(p, "pairs/"+p.name)
        export.add_raw(arc,raw(mode,order))
with (out/"summary.tgz").open("wb") as fp, tarfile.open(fileobj=fp,mode="w:gz",compresslevel=2) as arc:
    arc.add(out/"metrics.json","metrics.json")
    arc.add(out/"level1/pair.json","level1/pair.json")
    for leg in ("A","B"):
        arc.add(out/"level1"/leg/"linked","level1/"+leg+"/linked")
for name in ("raw.tgz","summary.tgz"):
    path=out/name
    with path.open("rb") as f:
        print(name,path.stat().st_size,hashlib.file_digest(f,"sha256").hexdigest(),flush=True)
