import argparse
from pathlib import Path
import sys
from apply import OWNER


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--study",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.study))
    import build_native as b
    a.base=b.BASE
    a.installer=Path(__file__).with_name("apply.py")
    a.jobs=2
    b.SOURCES={OWNER:"c22502e60a9cbc1653e498d99ce0c1923584c0c8a18879cd1a6d40afdc6e0fb0",
        "op_plugin/ops/opapi/EmbeddingKernelNpuOpApi.cpp":"a36d2188732e7a1f5a701bfd640930a9d094a5c48662299a8c630227ed63a1fd",
        "op_plugin/ops/opapi/StructKernelNpuOpApi.cpp":"795e6eb5ee36d3057a11cf70e6f6e3835cfd12a05183022a492eeb59d68848b2",
        "op_plugin/utils/op_api_common.cpp":"e49dbe4a0e6709f23d8c77437addecbbcccf14996c66db2ef66b7f1acbbb17f7"}
    b.run_build(a)
