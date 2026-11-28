"""Insert a guarded consumer-owned eager ACLNN executor bank in the pinned C++ backend."""
import argparse
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    root=parser.parse_args().root
    here=Path(__file__).resolve().parent
    unit=root/"op_plugin/ops/opapi/StructKernelNpuOpApi.cpp"
    source=unit.read_text(encoding="utf8")
    anchor='#include "op_plugin/utils/op_api_common.h"'
    entry='    EXEC_NPU_CMD(aclnnEmbeddingDenseBackward, grad_output, indices,'
    if source.count(anchor)!=1 or source.count(entry)!=1:
        raise ValueError("unexpected op-plugin source")
    source=source.replace(anchor,anchor+"\n"+(here/"index_scope.inc").read_text(encoding="utf8"))
    source=source.replace(entry,
        '    if (repeated_embedding::submit(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq, out)) {\n'
        '        return out;\n    }\n'+entry)
    unit.write_text(source,encoding="utf8",newline="\n")

if __name__=="__main__":
    main()
