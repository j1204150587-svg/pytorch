"""Build a private package from the original objects and two patched translation units."""
import argparse
from pathlib import Path
import sys


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--study",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    sys.path.insert(0,str(args.study))
    import build_native
    args.base=build_native.BASE
    args.installer=Path(__file__).with_name("apply.py")
    args.jobs=2
    build_native.SOURCES={
        "op_plugin/ops/opapi/AddmmKernelNpuOpApi.cpp":"51d2013e8a7ef58debca3803ae45a443f69477fd54504512833268200b445ab0",
        "op_plugin/utils/op_api_common.cpp":"e49dbe4a0e6709f23d8c77437addecbbcccf14996c66db2ef66b7f1acbbb17f7",
    }
    build_native.run_build(args)
