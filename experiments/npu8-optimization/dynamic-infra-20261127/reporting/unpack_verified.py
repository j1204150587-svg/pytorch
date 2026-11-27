"""Extract one locally retrieved manifest archive and verify its contents."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    root = args.out.resolve()
    with tarfile.open(args.archive, "r:gz") as src:
        manifest = json.load(src.extractfile("receipts-manifest.json"))
        expected = {v["member"]: v for v in manifest}
        if len(expected) != len(manifest): raise ValueError("duplicate manifest path")
        done = set()
        for member in src:
            if member.name == "receipts-manifest.json": continue
            target = (root / member.name).resolve()
            if not target.is_relative_to(root) or not member.isfile() or member.name in done:
                raise ValueError(member.name)
            item = expected[member.name]
            target.parent.mkdir(parents=True, exist_ok=True)
            with src.extractfile(member) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
            with target.open("rb") as saved:
                digest = hashlib.file_digest(saved, "sha256").hexdigest()
            if digest != item["sha256"]:
                raise ValueError("bad digest: " + member.name)
            done.add(member.name)
    if done != expected.keys(): raise ValueError("missing member")
    for group in sorted({Path(name).parts[0] for name in done}):
        (root / group / "receipts-manifest.json").write_text(json.dumps(
            [r for r in manifest if r["member"].split("/")[0] == group], indent=2))
    print(args.archive.name, len(done), "verified")

if __name__ == "__main__":
    main()

