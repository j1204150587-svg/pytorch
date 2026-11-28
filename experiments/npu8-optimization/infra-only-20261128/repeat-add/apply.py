from pathlib import Path
import sys
OWNER="op_plugin/ops/opapi/AddKernelNpuOpApi.cpp"
SHA="c0abf8d166728dfc00a9195ac29b5fc2cfa62a63fe85ea51dc96ca92a8ca71a5"

if __name__=="__main__":
    p=Path(sys.argv[1])/OWNER
    s=p.read_text()
    inc='#include "op_plugin/utils/op_api_common.h"'
    start=s.index("at::Tensor add(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha)")
    a=s.index("    add_out_npu_nocheck(self, other, alpha, result);",start)
    s=s[:a]+s[a:].replace("    add_out_npu_nocheck(self, other, alpha, result);",
        "    if (!repeat_add::submit(self, other, alpha, result)) {\n        add_out_npu_nocheck(self, other, alpha, result);\n    }",1)
    assert s.count(inc)==1
    p.write_text(s.replace(inc,inc+"\n"+Path(__file__).with_name("repeat_add.inc").read_text()))
