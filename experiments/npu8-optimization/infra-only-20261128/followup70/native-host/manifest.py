"""List only the files needed to reproduce and inspect one candidate in the publication checkout."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    base=args.directory.resolve()
    paths=[p for p in base.rglob("*") if p.is_file() and
           "__pycache__" not in p.parts and p.name!="publish-manifest.json" and
           "raw" not in p.relative_to(base).parts and "working" not in p.relative_to(base).parts]
    entries={}
    for p in sorted(paths):
        with p.open("rb") as f:
            digest=hashlib.file_digest(f,"sha256").hexdigest()
        entries[p.relative_to(base).as_posix()]={"bytes":p.stat().st_size,"sha256":digest}
    (base/"publish-manifest.json").write_text(json.dumps(entries,indent=2)+"\n")


if __name__=="__main__":
    main()
