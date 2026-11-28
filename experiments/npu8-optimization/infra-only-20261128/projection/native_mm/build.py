"""Compile and relink the two changed Mm translation units in the pinned isolated build."""
from pathlib import Path
import argparse
import sys


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--study",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    sys.path.insert(0,str(args.study))
    import build_native
    args.base=build_native.BASE
    args.installer=Path(__file__).with_name("apply_mm.py")
    args.jobs=2
    build_native.SOURCES={
        "op_plugin/ops/opapi/MmKernelNpuOpApi.cpp":"49a727b997272cb452139549c830beae86ccf6ea3ef1900a1720bf624d38f38f",
        "op_plugin/utils/op_api_common.cpp":"e49dbe4a0e6709f23d8c77437addecbbcccf14996c66db2ef66b7f1acbbb17f7",
    }
    build_native.run_build(args)
