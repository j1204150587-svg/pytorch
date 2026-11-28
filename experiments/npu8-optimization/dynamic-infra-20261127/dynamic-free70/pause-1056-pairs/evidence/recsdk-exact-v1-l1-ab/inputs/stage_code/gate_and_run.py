"""Run the built RecSDK primitive and the dynamic training pair through the existing shared lock."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",type=Path)
    parser.add_argument("build",type=Path)
    parser.add_argument("workspace",type=Path)
    a=parser.parse_args()
    study=a.study.resolve(strict=True)
    root=study/"dynamic-free70"
    build=a.build.resolve(strict=True)
    workspace=a.workspace.resolve(strict=True)
    assert build.is_relative_to(root)
    inside=lambda path: Path("/artifacts")/path.relative_to(study.parent)
    def locked(name,cmd):
        subprocess.run([sys.executable,"-u",str(root/"wait_run.py"),name,*map(str,cmd)],cwd=study,check=True)
    def recipe(mode):
        r=build/f"{mode}-env.json"
        dirs=[inside(build)/"opp/vendors"/v for v in ("hstu_dense_forward","hstu_dense_backward")]
        payload={"common":{"ASCEND_CUSTOM_OPP_PATH":os.pathsep.join(map(str,dirs))},"A":{},"B":{}}
        r.write_text(json.dumps(payload,indent=2)+"\n")
        return r
    while not (build/"cpu-build.exit").is_file():
        time.sleep(10)
    if int((build/"cpu-build.exit").read_text().strip())!=0:
        sys.exit("CPU builder failed; inspect cpu-build.log")
    assert (build/"libraries.sha256").is_file()
    primitive=f"recsdk-dense-kernel-{build.parent.name.rsplit('-',1)[1]}"
    check=build/"primitive-comparison.json"
    cmd=[sys.executable,"-B",root/"diagnose.py","--output",root/(primitive+"-{attempt}"),
         "--script",root/"recsdk-exact-src/recsdk_dense_exact/probe.py","--recipe",recipe("whole-kernel"),
         "--","--device","npu","--schedule",inside(workspace)/"configs/dynamic_shapes_npu8.json",
         "--library",inside(build)/"adapter/libhstu_dense_ops.so","--out","{output}"]
    locked(primitive,cmd)
    run=max(root.glob(primitive+"-*/train.exit"),key=lambda x:x.stat().st_mtime).parent
    assert (run/"train.exit").read_text()=="0"
    actual=run/"micro.json"
    data=json.loads(actual.read_text())
    mismatches=[{"shape":r["shape"],"case":r["case"],"failed":{k:v for k,v in r["differences"].items() if v["differing_bf16_words"]}}
                for r in data["cases"] if any(v["differing_bf16_words"] for v in r["differences"].values())]
    passed=data.get("complete") and len(data["cases"])==66 and not mismatches
    check.write_text(json.dumps(dict(run=str(run),passed=passed,cases=len(data["cases"]),mismatches=mismatches),indent=2)+"\n")
    if not passed:
        sys.exit("Primitive numerical differences; see "+str(check))
    spec=root/"runner_tools/recipes/recsdk-exact-v1.json"
    subprocess.run([sys.executable,str(root/"recsdk_dense_exact/make_recipe.py"),str(inside(build)),str(actual),str(spec)],check=True)
    locked("recsdk-exact-v1-pairs",[sys.executable,"-B",root/"runner_tools/run.py",spec,"--workspace",workspace,
                                   "gate","l1/ab","no-profiler/ab","l0/ab","l0/ba","no-profiler/ba"])


if __name__=="__main__":
    main()
