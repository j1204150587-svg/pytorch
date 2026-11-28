"""Stage bidirectional embedding, then specialize the stock Mode 1 Divs key preparation."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
OWNER="op_plugin/ops/opapi/DivKernelNpuOpApi.cpp"
PLUGIN="/artifacts/pta08-queued-replay-20261125-C6eCXV/checkout/third_party/op-plugin"


def once(src,a,b):
    if src.count(a)!=1:
        raise ValueError(a)
    return src.replace(a,b)


def stage(root):
    subprocess.run([sys.executable,"-B",str(HERE.parent/"embedding/apply.py"),str(root)],check=True)
    hdr=Path(os.environ.get("DIV_GEOMETRY_STOCK_HEADER",PLUGIN+"/op_plugin/utils/op_api_common.h")).read_text()
    start=hdr.index("template <typename... Args> bool hit_cache(")
    stop=hdr.index("template <typename ...Ts>",start)
    fn=hdr[start:stop]
    sig=fn[:fn.index("{")+1]
    fn=once(fn,sig,"""static bool geometry_hit_cache(aclrtStream acl_stream, const char *aclnn_api, void *phrase2,
                               const at::Tensor& x, const at::Scalar& scalar, const at::Tensor& out)
{
    using namespace div_geometry;
    if (!enabled) {
        return ::hit_cache(acl_stream, aclnn_api, phrase2, x, scalar, out);
    }
    Key key{};
    if (c10_npu::is_core_control_enabled() || c10_npu::option::OptionsManager::GetTaskQueueEnable() != 1 ||
        !geometry(x, scalar, out, key)) {
        count(Ineligible);
        return ::hit_cache(acl_stream, aclnn_api, phrase2, x, scalar, out);
    }
    using AddAddress = void(*)(void*);
    static const auto bind = reinterpret_cast<AddAddress>(GetOpApiFuncAddr("AddTensorAddrToCachedList"));
    if (!bind) {
        return ::hit_cache(acl_stream, aclnn_api, phrase2, x, scalar, out);
    }""")
    lo=fn.index("    if (c10_npu::is_core_control_enabled())")
    hi=fn.index("    setPTAHashKeyFunc(hashId);",lo)
    fn=fn[:lo]+"""    const auto device = c10_npu::current_device();
    key[10] = (static_cast<uint64_t>(static_cast<unsigned>(device)) << 1) | uint64_t(deterministic);
    key[11] = reinterpret_cast<uintptr_t>(acl_stream);
    bool hit = slot.valid && slot.key == key;
    uint64_t hashId;
    if (!hit || audit) {
        add_param_to_buf(deterministic);
        add_param_to_buf(std::string(aclnn_api), x, scalar, out);
        add_param_to_buf(device);
        add_param_to_buf((uintptr_t)acl_stream);
        hashId = calc_hash_id();
        if (hit) {
            TORCH_CHECK(hashId == slot.hash, "Divs exact geometry hash disagrees with stock");
            count(Checked);
        } else {
            slot = {true, key, hashId};
            count(FirstKey);
        }
    } else {
        bind(const_cast<void *>(x.storage().data()));
        bind(const_cast<void *>(out.storage().data()));
        hashId = slot.hash;
    }
    if (hit) {
        count(Reused);
    }
"""+fn[hi:]
    fn=once(fn,"    if (executor == nullptr) {","    count(executor ? ExecHit : ExecMiss);\n    if (executor == nullptr) {")
    path=root/OWNER
    src=path.read_text()
    inc='#include "op_plugin/utils/op_api_common.h"'
    src=once(src,inc,inc+"\n"+(HERE/"geometry.inc").read_text()+"\n"+fn)
    start=src.index("at::Tensor div(const at::Tensor& self, const at::Scalar& other) {")
    stop=src.index("at::Tensor div(",start+14)
    piece=once(src[start:stop],"    EXEC_NPU_CMD(aclnnDivs, self, other, result);",
               "#define hit_cache geometry_hit_cache\n    EXEC_NPU_CMD(aclnnDivs, self, other, result);\n#undef hit_cache")
    path.write_text(src[:start]+piece+src[stop:])


if __name__=="__main__":
    stage(Path(sys.argv[1]))
