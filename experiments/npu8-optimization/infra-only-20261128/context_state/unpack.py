"""Verify a portable receipt archive and extract regular files, including long Windows paths."""
import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def main():
    p=argparse.ArgumentParser()
    p.add_argument("archive",type=Path)
    p.add_argument("checksum",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    with args.archive.open("rb") as fp:digest=hashlib.file_digest(fp,"sha256").hexdigest()
    assert digest==args.checksum.read_text().split()[0]
    root=args.out.resolve()
    if os.name=="nt":root=Path("\\\\?\\"+str(root))
    with tarfile.open(args.archive) as t:
        for item in t:
            rel=PurePosixPath(item.name)
            if rel.is_absolute() or ".." in rel.parts or not (item.isfile() or item.isdir()):
                raise ValueError(item.name)
            dest=root.joinpath(*rel.parts)
            if item.isdir():
                dest.mkdir(parents=True,exist_ok=True)
                continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            with t.extractfile(item) as source, dest.open("wb") as out:
                shutil.copyfileobj(source,out)
            assert dest.stat().st_size==item.size
    print(digest)

if __name__=="__main__":
    main()
