"""Use a compact descriptor only on the read-only RHS of the functional BF16 Mm."""
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "native"))
from apply import once, patch_common


if __name__ == "__main__":
    root = Path(sys.argv[1])
    op = root / "op_plugin/ops/opapi/MmKernelNpuOpApi.cpp"
    source = op.read_text()
    begin = source.index("at::Tensor mm(const at::Tensor &self, const at::Tensor &mat2)\n")
    end = source.index("at::Tensor &mm_out(", begin)
    call = "        EXEC_NPU_CMD(aclnnMm, self, mat2, result, cube_math_type);"
    body = once(source[begin:end], call, "        projection_rebase::Scope scope(self, mat2);\n" + call)
    source = source[:begin] + body + source[end:]
    header = '#include "op_plugin/utils/op_api_common.h"'
    op.write_text(once(source, header, header+'\n#include "op_plugin/utils/mm_rebase.h"'))
    patch_common(root / "op_plugin/utils/op_api_common.cpp", here, "mm_rebase", "0x4D4D574754313238ULL")
