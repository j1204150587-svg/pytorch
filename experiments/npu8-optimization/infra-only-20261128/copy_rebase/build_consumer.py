"""Build only the copy caller and the native allocator owner check on the dual embedding library."""
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import tempfile

from build import digest, replaced


def main():
    here=Path(__file__).resolve().parent
    parent=json.loads((here.parent/"embedding/build/build-receipt.json").read_bytes())
    assert parent["library_sha256"]==digest(Path(parent["library"]))=="4e4adcd8348f437abefb6f48c14b224f4b1b5342b145ddd4b745964f45ec2d8c"
    spec=importlib.util.spec_from_file_location("pta_builder","/artifacts/dynamic-infra-20261127/build_native.py")
    builder=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    checkout=Path(parent["base"])/"checkout"
    build=checkout/"build"
    db=json.loads((build/"compile_commands.json").read_bytes())
    out=here/"build-consumer"
    out.mkdir(exist_ok=True)
    staged=Path(tempfile.mkdtemp(prefix="compile-",dir=out))
    caller="torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp"
    allocator="torch_npu/csrc/core/npu/NPUCachingAllocator.cpp"
    header="third_party/op-plugin/op_plugin/utils/op_api_common.h"
    for name,expected in ((caller,"a71b330cdd1e1955592ea8b7a18d18fd418b034cb6fbfe998fa0c82455c21074"),
                          (allocator,"b1a3ee512a5d70755529215a02bebf83b818ff90b1ec8f4501f29e416e6fcb60"),
                          (header,"6849b4417a97df8fc349d02c8668a6a05c1b8d31dbc93d769e8ddf28181f5484")):
        assert digest(checkout/name)==expected,name
    h=(checkout/header).read_text()
    macro=h[h.index("#define EXEC_NPU_CMD_V2("):h.index("#define EXEC_NPU_CMD(aclnn_api")]
    macro=replaced(macro,"#define EXEC_NPU_CMD_V2(","#define EXEC_OWNED_COPY_V2(")
    macro=replaced(macro,"auto copied_params = CopyTypesV2(__VA_ARGS__);",
                   "auto copied_params = CopyTypesV2(__VA_ARGS__); count_consumer_copy(2);")
    macro=replaced(macro,"[copied_params, acl_stream, deterministic_status, aic_num, aiv_num]()->int {",
                   "[copied_params, dst_owner=dst.storage(), src_owner=src.storage(), acl_stream, deterministic_status, aic_num, aiv_num]()->int { (void)dst_owner; (void)src_owner; count_consumer_copy(3);")
    src=(checkout/caller).read_text()
    src=replaced(src,"namespace at_npu {", (here/"copy_consumer.inc").read_text()+"\n"+macro+"\nnamespace at_npu {")
    src=replaced(src,"    EXEC_NPU_CMD(aclnnInplaceCopy, dst, src);",
                 "    if (use_consumer_copy(dst, src)) {\n        EXEC_OWNED_COPY_V2(aclnnInplaceCopy, dst, src);\n        return;\n    }\n    EXEC_NPU_CMD(aclnnInplaceCopy, dst, src);")
    native=(checkout/allocator).read_text()
    marker="NpuCachingAllocator caching_allocator;"
    # Only native, live DataPtr allocations can pass. The caller retains both owners while the
    # closure runs, and native reuse after launch is ordered on the same allocation stream.
    native=replaced(native,marker,marker+"""

bool ownedByCopyStream(const c10::DataPtr& ptr, aclrtStream stream)
{
    if (!ptr.get() || ptr.get_deleter() != &local_raw_delete || get() != &caching_allocator) {
        return false;
    }
    const auto* block = static_cast<const Block*>(caching_allocator.getBlockPtr(ptr));
    return block && block->stream == stream;
}
""")
    units=[]
    for name,code in ((caller,src),(allocator,native)):
        dest=staged/name
        dest.parent.mkdir(parents=True)
        dest.write_text(code)
        units.append(builder.compile_unit(db,checkout,staged,name))
    link=list(parent["link"])
    lib=out/"libtorch_npu.so"
    builder.replace_flag(link,"-o",lib)
    for row in units:
        matches=[i for i,arg in enumerate(link) if arg.endswith(".o") and (build/arg).resolve()==Path(row["original_object"])]
        assert len(matches)==1,matches
        link[matches[0]]=row["object"]
    record=dict(status="compiling",parent_sha256=parent["library_sha256"],parent_receipt=str(here.parent/"embedding/build/build-receipt.json"),
                stage=str(staged),compile=units,link=link,patched_sources={name:digest(staged/name) for name in (caller,allocator)})
    receipt=out/"build-receipt.json"
    receipt.write_text(json.dumps(record,indent=2))
    def compile_one(row):
        with (staged/(Path(row["source"]).stem+".compile.log")).open("wb") as log:
            subprocess.run(row["argv"],cwd=row["cwd"],stdout=log,stderr=subprocess.STDOUT,check=True)
        row["object_sha256"]=digest(Path(row["object"]))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(compile_one,units))
    subprocess.run(link,cwd=build,check=True)
    record.update(status="built",library=str(lib),library_sha256=digest(lib))
    receipt.write_text(json.dumps(record,indent=2)+"\n")
    print(record["library_sha256"],lib)


if __name__=="__main__":
    main()
