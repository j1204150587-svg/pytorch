"""Build the isolated forward package with the pinned baseline Ninja rules."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import build_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,default=build_native.BASE)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--jobs",type=int,choices=(1,2),default=2)
    args = parser.parse_args()
    args.installer = HERE / "apply_forward.py"
    build_native.SOURCES = {
        "op_plugin/ops/opapi/EmbeddingKernelNpuOpApi.cpp":
            "a36d2188732e7a1f5a701bfd640930a9d094a5c48662299a8c630227ed63a1fd",
        "op_plugin/utils/op_api_common.cpp":
            "e49dbe4a0e6709f23d8c77437addecbbcccf14996c66db2ef66b7f1acbbb17f7",
    }
    build_native.run_build(args)


if __name__ == "__main__":
    main()
