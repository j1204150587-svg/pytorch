"""Apply the experiment only to the two no-rounding, functional Div implementations."""
import hashlib
from pathlib import Path
import sys

OWNER = "op_plugin/ops/opapi/DivKernelNpuOpApi.cpp"
OWNER_SHA = "c22502e60a9cbc1653e498d99ce0c1923584c0c8a18879cd1a6d40afdc6e0fb0"


def replace_one(src, before, after):
    if src.count(before) != 1:
        raise ValueError(before)
    return src.replace(before, after)


def apply(root):
    path = root / OWNER
    if hashlib.sha256(path.read_bytes()).hexdigest() != OWNER_SHA:
        raise ValueError(f"Unexpected pinned source contents in {path}")
    src = path.read_text()
    include = '#include "op_plugin/utils/op_api_common.h"'
    src = replace_one(src, include, include+"\n"+Path(__file__).with_name("compact_div.inc").read_text())
    sig = "at::Tensor div(const at::Tensor& self, const at::Tensor& other) {"
    pos, stop = src.index(sig), src.index("at::Tensor div(", src.index(sig)+len(sig))
    old = "    div_out_npu_opapi_nocheck(self_cp, other, result);"
    new = """    if (compact_div::enabled() && other.dim() == 0 && other.device().is_cpu()) {
        const c10::Scalar value = other.item();
        if (!compact_div::run(self_cp, value, result, compact_div::TENSOR_HIT)) {
            EXEC_NPU_CMD(aclnnDivs, self_cp, value, result);
        }
    } else {
        div_out_npu_opapi_nocheck(self_cp, other, result);
    }"""
    src = src[:pos]+replace_one(src[pos:stop],old,new)+src[stop:]
    sig = "at::Tensor div(const at::Tensor& self, const at::Scalar& other) {"
    pos, stop = src.index(sig), src.index("at::Tensor div(", src.index(sig)+len(sig))
    old = "    EXEC_NPU_CMD(aclnnDivs, self, other, result);"
    new = """    if (!compact_div::enabled() || !compact_div::run(self, other, result, compact_div::SCALAR_HIT)) {
        EXEC_NPU_CMD(aclnnDivs, self, other, result);
    }"""
    path.write_text(src[:pos]+replace_one(src[pos:stop],old,new)+src[stop:])


if __name__ == "__main__":
    apply(Path(sys.argv[1]))
