"""Package raw dynamic-training evidence on the host, including one trace per profiled leg."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size):
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def candidates(root):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root)
        if "profile" in rel.parts:
            if path.suffix.lower() != ".csv" and path.name not in ("trace_view.json", "trace_view.json.gz"):
                continue
        elif "workspace" in rel.parts or rel.parts[0] == "work":
            if not (path.name == "dynamic_shapes_npu8.json" or path.suffix in (".py", ".sh", ".patch", ".json")):
                continue
        elif path.suffix.lower() not in (".json", ".csv", ".tsv", ".sha256", ".log", ".exit", ".py", ".sh", ".patch") and not (
                path.name.endswith((".csv.gz", ".tsv.gz")) or path.name in ("trace_view.json.gz", "source.sha")):
            continue
        yield path


def files(root):
    by_leg = {}
    leg_dirs = {p.name for p in root.iterdir() if p.is_dir() and re.fullmatch(r"\d+-[AB]", p.name)}
    sole_leg = next(iter(leg_dirs)) if len(leg_dirs) == 1 else ""
    for path in candidates(root):
        if path.name in ("step_trace_time.csv", "trace_view.json", "trace_view.json.gz"):
            first = path.relative_to(root).parts[0]
            name = "trace" if path.name.startswith("trace_view") else "steps"
            key = first if first in leg_dirs else sole_leg
            by_leg.setdefault((key, name), []).append(path)
        else:
            yield path
    for (leg, _), paths in by_leg.items():
        # A previously collected single-leg run can also have a flattened duplicate at the run root.
        # Keep the canonical leg path and prefer its ready-made compressed copy.
        yield min(paths, key=lambda p: (leg != "" and p.relative_to(root).parts[0] != leg,
                                        len(p.parts), p.suffix != ".gz"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, action="append", metavar="LABEL=DIRECTORY")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    entries = []
    labels = set()
    with tarfile.open(args.out, "x:gz", compresslevel=1) as archive:
        for spec in args.run:
            label, directory = spec.split("=", 1)
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", label) or label in (".", "..") or label in labels:
                raise ValueError(f"invalid or duplicate label {label}")
            labels.add(label)
            root = Path(directory).resolve(strict=True)
            for path in files(root):
                arcname = (Path(label) / path.relative_to(root)).as_posix()
                with path.open("rb") as raw:
                    stream = HashReader(raw)
                    archive.addfile(archive.gettarinfo(str(path), arcname=arcname), stream)
                entries.append({"source": str(path), "member": arcname, "bytes": path.stat().st_size,
                                "sha256": stream.digest.hexdigest()})
        manifest = json.dumps(entries, indent=2, ensure_ascii=False).encode("utf-8")
        info = tarfile.TarInfo("receipts-manifest.json")
        info.size = len(manifest)
        archive.addfile(info, io.BytesIO(manifest))
    print(f"{args.out}: {len(entries)} files")


if __name__ == "__main__":
    main()
