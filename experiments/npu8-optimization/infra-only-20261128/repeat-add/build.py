"""Compile only Add and relink it with the three already verified dual-embedding objects."""
import argparse
import json
from pathlib import Path
import sys
from apply import OWNER,SHA

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--study",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--parent",type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.study))
    import build_native as b
    parent=json.loads(a.parent.read_bytes())
    if b.digest(Path(parent["library"]))!=parent["library_sha256"] or parent["library_sha256"]!="4e4adcd8348f437abefb6f48c14b224f4b1b5342b145ddd4b745964f45ec2d8c":
        raise ValueError("Unexpected embedding parent")
    for u in parent["compile"]:
        if b.digest(Path(u["object"]))!=u["object_sha256"]:
            raise ValueError(u["source"])
    old=b.relink_cmd
    b.relink_cmd=lambda build,units,path:old(build,[*parent["compile"],*units],path)
    b.SOURCES={OWNER:SHA}
    a.base=b.BASE
    a.installer=Path(__file__).with_name("apply.py")
    a.jobs=1
    b.run_build(a)
