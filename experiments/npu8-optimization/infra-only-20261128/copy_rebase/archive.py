"""Pack this run's compact raw receipts; keep vendor profiler trees at their source paths."""

import argparse
import importlib.util
import json
from pathlib import Path
import tarfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("pairs",nargs="+")
    parser.add_argument("--residual128", action="store_true")
    parser.add_argument("--consumer", action="store_true")
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    root=here.parent
    spec=importlib.util.spec_from_file_location("export_raw",root/"export.py")
    exporter=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    with tarfile.open(args.output,"w:gz",compresslevel=2) as tar:
        for name in args.pairs:
            path=root/"pairs"/name
            entry=json.loads((path/"result.json").read_bytes())
            if entry["exit"]!=0:
                raise ValueError(f"{name} exit {entry['exit']}")
            tar.add(path,"pairs/"+name)
            raw=Path(entry["directory"])
            exporter.add_raw(tar,raw)
            for leaf in raw.glob("0?-?/npu_*_counter-*.json"):
                tar.add(leaf,"raw/"+raw.name+"/"+leaf.relative_to(raw).as_posix())
        prefix = "copy-consumer" if args.consumer else ("copy-rebase-128" if args.residual128 else "copy-rebase")
        for p in (root/"recipes"/(prefix+".json"),root/"recipes"/(prefix+"-audit.json")):
            tar.add(p,"recipes/"+p.name)
        suffix = "-consumer" if args.consumer else ("-128" if args.residual128 else "")
        tar.add(here/("runs"+suffix), "micro")
        tar.add(here/("series"+suffix+".log"), "series.log")
        tar.add(here/"counter_copy.c","counter_copy.c")


if __name__=="__main__":
    main()
