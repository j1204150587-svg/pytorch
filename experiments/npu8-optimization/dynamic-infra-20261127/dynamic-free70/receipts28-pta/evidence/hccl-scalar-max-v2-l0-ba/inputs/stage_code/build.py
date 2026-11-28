"""Recompile the pinned ProcessGroupHCCL TU and relink only an isolated package."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def replace_once(src,old,new):
    if src.count(old)!=1:raise ValueError("unrecognized allreduce source")
    return src.replace(old,new)


def patched(src):
    src=replace_once(src,"#include <ATen/record_function.h>","#include <ATen/record_function.h>\n#include <atomic>")
    anchor="int g_allreduceID = 0;"
    counters="""
namespace {
std::atomic<uint64_t> rank1MaxHits{0}, rank1MaxHealthFallback{0};

bool rank1MaxEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("NPU_HCCL_RANK1_SCALAR");
        return v && std::strcmp(v, "intmax") == 0
            && std::getenv("HCCL_OP_EXPANSION_MODE") == nullptr;
    }();
    return on;
}
}

extern "C" __attribute__((visibility("default"))) uint64_t hstu_hccl_rank1_count(unsigned counter) {
    return counter == 0 ? rank1MaxHits.load(std::memory_order_relaxed)
                        : rank1MaxHealthFallback.load(std::memory_order_relaxed);
}
"""
    src=replace_once(src,anchor,counters+anchor)
    start=src.index("c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::allreduce(")
    end=src.index("\nc10::intrusive_ptr<c10d::Work>",start+1)
    body=src[start:end]
    label="            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, hcclReduceOp, comm, stream, is_dispatched]() -> int {"
    guard="""            const bool scalarIdentity = rank1MaxEnabled() && getSize() == 1 && getRank() == 0
                && !coalescing_state_ && !desyncDebug_ && tensors.size() == 1
                && opts.reduceOp == c10d::ReduceOp::MAX && hcclReduceOp == HCCL_REDUCE_MAX
                && hcclType == HCCL_DATA_TYPE_INT32 && input.scalar_type() == at::kInt
                && inputDataPtr == outputDataPtr && comm && numel == 1 && input.numel() == 1
                && input.is_contiguous() && c10_npu::option::OptionsManager::GetTaskQueueEnable() == 1
                && !at_npu::native::env::CheckOpHookEnable()
                && c10_npu::currentStreamCaptureStatusMayInitCtx() == c10_npu::CaptureStatus::None;
"""
    body=replace_once(body,label,guard+label.replace("comm, stream","scalarIdentity, comm, stream"))
    anchor="                auto hccl_result = hcclAllReduce("
    treatment="""                if (scalarIdentity) {
                    using Health = HcclResult (*)(HcclComm, HcclResult*);
                    static Health query = reinterpret_cast<Health>(
                        TORCH_NPU_GET_FUNCTION(libhccl, HcclGetCommAsyncError));
                    HcclResult health = HCCL_SUCCESS;
                    if (query && query(comm, &health) == HCCL_SUCCESS && health == HCCL_SUCCESS) {
                        rank1MaxHits.fetch_add(1, std::memory_order_relaxed);
                        *is_dispatched = true;
                        return HCCL_SUCCESS;
                    }
                    rank1MaxHealthFallback.fetch_add(1, std::memory_order_relaxed);
                }
"""
    body=replace_once(body,anchor,treatment+anchor)
    return src[:start]+body+src[end:]


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--builder",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--base",type=Path,default=Path("/artifacts/pta08-queued-replay-20261125-C6eCXV"))
    args=cli.parse_args()
    spec=importlib.util.spec_from_file_location("existing_ninja",args.builder.resolve(strict=True))
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    base=args.base.resolve(strict=True)
    checkout=base/"checkout"
    out=args.output.resolve()
    helper.check(not out.exists() and not out.is_relative_to(base),"requires a new isolated directory")
    out.mkdir(parents=True)
    stage=Path(tempfile.mkdtemp(prefix="compile-",dir=out))
    src="torch_npu/csrc/distributed/ProcessGroupHCCL.cpp"
    pinned="05244bdb68d60ee2fb5f335007248aadf5eed7a9de2b39997f16dc8a8182f3f9"
    helper.check(helper.output(["git","rev-parse","HEAD"],cwd=checkout)==helper.PTA_SHA,"unexpected PTA revision")
    helper.check(helper.digest(checkout/src)==pinned,"unexpected ProcessGroupHCCL source")
    dest=stage/src
    dest.parent.mkdir(parents=True)
    dest.write_text(patched((checkout/src).read_text(encoding="utf8")),encoding="utf8")
    build=checkout/"build"
    db_path=build/"compile_commands.json"
    unit=helper.compile_unit(json.loads(db_path.read_text()),checkout,stage,src)
    linked=stage/"libtorch_npu.so"
    command=helper.relink_cmd(build,[unit],linked)
    receipt=dict(pta_sha=helper.PTA_SHA,original_sha256=pinned,patched_sha256=helper.digest(dest),
                 input_object_sha256=helper.digest(Path(unit["original_object"])),compile=unit,link=command,
                 stock_library_sha256=helper.digest(base/"package/torch_npu/lib/libtorch_npu.so"))
    note=out/"build.json"
    note.write_text(json.dumps(dict(receipt,status="compiling"),indent=2))
    subprocess.run(unit["argv"],cwd=unit["cwd"],check=True)
    subprocess.run(command,cwd=build,check=True)
    original=base/"package"
    package=stage/"package"
    relative=Path("torch_npu/lib/libtorch_npu.so")
    shutil.copytree(original,package,symlinks=True,ignore=lambda path,names:
                    {relative.name} if Path(path)==(original/relative).parent else set())
    helper.check((package/relative).parent.resolve().is_relative_to(package),"destination escapes isolated package")
    os.replace(linked,package/relative)
    package.rename(out/"package")
    note.write_text(json.dumps(dict(receipt,status="built",library_sha256=helper.digest(out/"package"/relative)),indent=2))
