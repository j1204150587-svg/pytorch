"""Produce identical A/B native packages, counter preload only for semantic audit."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def digest(path):
    with path.open("rb") as src:return hashlib.file_digest(src,"sha256").hexdigest()


def main():
    here=Path(__file__).resolve().parent
    root=here.parent
    inner="/artifacts/infra-only-20261128"
    lib=Path("torch_npu/lib/libtorch_npu.so")
    parent=root/"stock-overlay/embedding"
    before=(root/"embedding/build/build-receipt.json")
    assert digest(parent/lib)==json.loads(before.read_bytes())["library_sha256"]
    receipt=json.loads((here/"build/build-receipt.json").read_bytes())
    native=here/"build/libtorch_npu.so"
    assert receipt["library_sha256"]==digest(native)
    dest=root/"stock-overlay/context-state"
    if not dest.exists():
        with tempfile.TemporaryDirectory(dir=dest.parent) as work:
            target=Path(work)/"overlay"
            def skip(path,names):return {lib.name} if Path(path)==(parent/lib).parent else set()
            shutil.copytree(parent,target,ignore=skip,copy_function=os.link,symlinks=True)
            shutil.copyfile(native,target/lib)
            target.rename(dest)
    assert digest(dest/lib)==receipt["library_sha256"]
    symbols='const char *symbols[] = {"npu_addmm_rebase_counter", "npu_mm_rebase_counter"};'
    text=(root/"counter_dump.c").read_text()
    assert text.count(symbols)==1
    text=text.replace(symbols,'const char *symbols[] = {"npu_active_context_counter", "npu_embedding_forward_rebase_counter", "npu_embedding_backward_rebase_counter"};')
    (here/"counter.c").write_text(text)
    so=here/"counter.so"
    dest_inner=inner+"/context_state"
    if not so.exists():
        subprocess.run(["docker","run","--rm","--runtime=runc","--network=none","--cpus","1","-v",str(here)+":"+dest_inner,
            "recsys-gr-deepspeed:host88-ds0.19.0-torch2.10.0","gcc","-O2","-shared","-fPIC",dest_inner+"/counter.c",
            "-o",dest_inner+"/counter.so","-ldl"],check=True)
    for audit in (False,True):
        prefix=inner+"/stock-overlay/context-state"
        common={"PYTHONPATH":prefix,"ACLNN_CACHE_LIMIT":"65536","NPU_DYNAMIC_REBASE_INDICES":"1",
                "NPU_DYNAMIC_REBASE_FORWARD":"1","NPU_DYNAMIC_REBASE_COUNT":str(int(audit))}
        libraries={prefix+"/"+str(lib):receipt["library_sha256"]}
        if audit:
            common.update(LD_PRELOAD=dest_inner+"/counter.so",INFRA_COUNTER_DIR="{output}")
            libraries[dest_inner+"/counter.so"]=digest(so)
        rec=dict(common=common,switch="NPU_CONTEXT_POSITIVE_MASK",A="0",B="1",libraries=libraries)
        (root/"recipes"/("context-state-audit.json" if audit else "context-state.json")).write_text(json.dumps(rec,indent=2)+"\n")

if __name__=="__main__":
    main()
