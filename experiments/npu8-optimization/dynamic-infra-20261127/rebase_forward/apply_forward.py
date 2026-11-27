"""Apply an isolated Mode 1 embedding forward descriptor experiment."""
import argparse
from pathlib import Path


def once(source, needle, replacement):
    if source.count(needle) != 1:
        raise ValueError(f"source mismatch for {needle!r}")
    return source.replace(needle,replacement)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="op-plugin repository containing op_plugin/")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    common = args.root / "op_plugin/utils/op_api_common.cpp"
    kernel = args.root / "op_plugin/ops/opapi/EmbeddingKernelNpuOpApi.cpp"
    scope_header = common.with_name("dynamic_index_scope.h")
    cpp, ops = common.read_text(encoding="utf-8"), kernel.read_text(encoding="utf-8")
    if scope_header.exists():
        raise ValueError(f"already patched or unexpected file {scope_header}")
    keyword = '#include "op_api_common.h"'
    cpp = once(cpp,keyword,keyword + "\n" + (here/"index_scope.inc").read_text())
    start = cpp.index("void add_param_to_buf(const at::Tensor &at_tensor)")
    end = cpp.index("void add_param_to_buf(const at::Scalar &",start)
    segment = cpp[start:end]
    segment = once(segment,"auto so = at_tensor.storage_offset();","""const auto *view = dynamic_index_rebase::for_tensor(at_tensor);
    if (view) {
        dynamic_index_rebase::key_uses.fetch_add(1, std::memory_order_relaxed);
        const uint64_t domain = 0x4649445852425345ULL;
        MEMCPY_TO_BUF(&domain, sizeof(domain));
    }
    auto so = view ? view->offset : at_tensor.storage_offset();""")
    segment = once(segment,"storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());",
                   "storageDims.push_back(view ? view->storage_elements : at_tensor.storage().nbytes() / at_tensor.itemsize());")
    segment = once(segment,"addTensorAddrToCachedListFunc(const_cast<void*>(at_tensor.storage().data()));",
                   "addTensorAddrToCachedListFunc(view ? view->base : const_cast<void*>(at_tensor.storage().data()));")
    cpp = cpp[:start] + segment + cpp[end:]
    start = cpp.index("aclTensor *ConvertType(const at::Tensor &at_tensor)")
    end = cpp.index("aclScalar *ConvertType(",start)
    segment = cpp[start:end]
    segment = once(segment,"at::ScalarType scalar_data_type", """const auto *view = dynamic_index_rebase::for_tensor(at_tensor);
    if (view) {
        dynamic_index_rebase::descriptor_uses.fetch_add(1, std::memory_order_relaxed);
    }
    at::ScalarType scalar_data_type""")
    segment = once(segment,"storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());",
                   "storageDims.push_back(view ? view->storage_elements : at_tensor.storage().nbytes() / at_tensor.itemsize());")
    segment = once(segment,"at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),",
                   "view ? view->offset : at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),")
    segment = once(segment,"const_cast<void *>(at_tensor.storage().data()));",
                   "view ? view->base : const_cast<void *>(at_tensor.storage().data()));")
    cpp = cpp[:start] + segment + cpp[end:]
    include = '#include "op_plugin/utils/op_api_common.h"'
    ops = once(ops,include,include + '\n#include "op_plugin/utils/dynamic_index_scope.h"')
    target = "    EXEC_NPU_CMD(aclnnEmbedding, weight, indices,"
    ops = once(ops,target,"    dynamic_index_rebase::Scope scope(weight, indices);\n" + target)
    scope_header.write_bytes((here/"index_scope.h").read_bytes())
    common.write_text(cpp, encoding="utf-8", newline="\n")
    kernel.write_text(ops, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
