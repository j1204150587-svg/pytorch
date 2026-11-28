"""Patch the pinned op-plugin Addmm functional path, descriptor builder and cache address binding."""
from pathlib import Path
import sys


def once(text, needle, replacement):
    if text.count(needle) != 1:
        raise ValueError(f"unexpected op-plugin source near {needle!r}")
    return text.replace(needle, replacement)


def apply(root):
    here=Path(__file__).resolve().parent
    kernel=root/"op_plugin/ops/opapi/AddmmKernelNpuOpApi.cpp"
    common=root/"op_plugin/utils/op_api_common.cpp"
    ops=kernel.read_text()
    end=ops.index("at::Tensor &addmm_(")
    start=ops.rindex("at::Tensor addmm(\n",0,end)
    target="        EXEC_NPU_CMD(aclnnAddmm, self, mat1, mat2, beta, alpha, result, cube_math_type);"
    ops=ops[:start]+once(ops[start:end],target,"        projection_rebase::Scope scope(self, mat1, mat2);\n"+target)+ops[end:]
    include='#include "op_plugin/utils/op_api_common.h"'
    kernel.write_text(once(ops,include,include+'\n#include "op_plugin/utils/addmm_rebase.h"'))
    common.with_name("addmm_rebase.h").write_bytes((here/"addmm_rebase.h").read_bytes())
    source=common.read_text()
    include='#include "op_api_common.h"'
    source=once(source,include,include+"\n"+(here/"addmm_rebase.inc").read_text())
    start=source.index("void add_param_to_buf(const at::Tensor &at_tensor)")
    end=source.index("void add_param_to_buf(const at::Scalar &",start)
    keys=once(source[start:end],"auto so = at_tensor.storage_offset();",'''const auto *projection = projection_rebase::lookup(at_tensor);
    if (projection) {
        projection_rebase::keys.fetch_add(1, std::memory_order_relaxed);
        const uint64_t domain = 0x4144444D4D313238ULL;
        MEMCPY_TO_BUF(&domain, sizeof(domain));
    }
    auto so = projection ? 0 : at_tensor.storage_offset();''')
    size="at_tensor.storage().nbytes() / at_tensor.itemsize()"
    address="const_cast<void*>(at_tensor.storage().data())"
    keys=once(keys,"storageDims.push_back("+size+");","storageDims.push_back(projection ? projection->elements : "+size+");")
    keys=once(keys,"addTensorAddrToCachedListFunc("+address+");",
              "addTensorAddrToCachedListFunc(projection ? projection->base : "+address+");")
    source=source[:start]+keys+source[end:]
    start=source.index("aclTensor *ConvertType(const at::Tensor &at_tensor)")
    end=source.index("aclScalar *ConvertType(",start)
    desc=once(source[start:end],"at::ScalarType scalar_data_type",'''const auto *projection = projection_rebase::lookup(at_tensor);
    if (projection) {
        projection_rebase::descriptors.fetch_add(1, std::memory_order_relaxed);
    }
    at::ScalarType scalar_data_type''')
    desc=once(desc,"storageDims.push_back("+size+");","storageDims.push_back(projection ? projection->elements : "+size+");")
    desc=once(desc,"at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),",
              "projection ? 0 : at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),")
    desc=once(desc,"const_cast<void *>(at_tensor.storage().data()));","projection ? projection->base : const_cast<void *>(at_tensor.storage().data()));")
    common.write_text(source[:start]+desc+source[end:])


if __name__=="__main__":
    apply(Path(sys.argv[1]))
