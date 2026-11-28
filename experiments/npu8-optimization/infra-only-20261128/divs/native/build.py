"""Build a separate TorchNPU package using the pinned toolchain and existing objects."""
import argparse
from pathlib import Path
import sys
from apply import OWNER, OWNER_SHA


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study",type=Path,required=True,help="Directory containing the pinned build_native.py")
    ap.add_argument("--output",type=Path,required=True)
    args = ap.parse_args()
    sys.path.insert(0,str(args.study))
    import build_native
    args.base = build_native.BASE
    args.installer = Path(__file__).with_name("apply.py")
    args.jobs = 1
    build_native.SOURCES = {OWNER:OWNER_SHA}
    build_native.run_build(args)
