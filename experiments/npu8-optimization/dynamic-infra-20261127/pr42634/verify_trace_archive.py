"""Check the raw gzip traces against the original host manifest and gzip CRCs."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def verify(manifest, root):
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    actual = {f.resolve() for f in root.rglob("trace_view.json.gz") if f.is_file()}
    used = set()
    rows = []
    for entry in entries:
        host_path = Path(entry["path"])
        leg = host_path.parent.name
        path = root / entry["label"] / leg / "trace_view.json.gz"
        resolved = path.resolve(strict=True)
        if resolved not in actual or resolved in used:
            raise ValueError(f"missing or duplicate trace: {path}")
        used.add(resolved)
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"unexpected size: {path}")
        h = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(1024*1024):
                h.update(chunk)
        if h.hexdigest() != entry["sha256"]:
            raise ValueError(f"original trace SHA-256 mismatch: {path}")
        with gzip.open(path,"rb") as f:
            unpacked = sum(len(chunk) for chunk in iter(lambda:f.read(1024*1024),b""))
        rows.append((entry["label"],leg,path.relative_to(root).as_posix(),entry["bytes"],unpacked,h.hexdigest()))
    if actual != used:
        raise ValueError(f"{len(actual-used)} traces missing from the original host manifest")
    return rows


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--manifest",type=Path,required=True)
    cli.add_argument("--root",type=Path,required=True)
    cli.add_argument("--csv",type=Path)
    args = cli.parse_args()
    rows = verify(args.manifest,args.root)
    if args.csv:
        with args.csv.open("w",encoding="utf-8",newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("family","leg","trace","gzip_bytes","unpacked_bytes","sha256"))
            writer.writerows(rows)
    print(f"{len(rows)} gzip traces verified; {sum(r[3] for r in rows)} bytes on disk")
