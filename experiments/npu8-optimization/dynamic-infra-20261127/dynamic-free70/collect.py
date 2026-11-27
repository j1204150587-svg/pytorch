"""Snapshot immutable card runs, their source and trace locations, and isolated gate attempts."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--out", required=True, type=Path)
    cli.add_argument("--traces", action="store_true")
    cli.add_argument("--only", nargs="*")
    args = cli.parse_args()
    traces = []
    manifest = []
    with tarfile.open(args.out, "x:gz", compresslevel=2) as archive:
        def save(path, target):
            if not path.is_file() or path.is_symlink():
                return
            archive.add(path, arcname=target, recursive=False)
            manifest.append(dict(path=str(target), bytes=path.stat().st_size, sha256=sha(path)))

        for runlist in sorted((ROOT / "runs").glob("*/runs.json")):
            family = runlist.parent.name
            if args.only and family not in args.only:
                continue
            save(runlist, f"registries/{family}/runs.json")
            for name in ("recipe.json", "runner-sha256.json"):
                save(runlist.with_name(name), f"registries/{family}/{name}")
            for entry in json.loads(runlist.read_bytes()):
                if entry["exit"]:
                    continue
                root = Path(entry["directory"])
                label = f'{family}-{entry["mode"]}-{entry["order"]}'
                for path in sorted(root.rglob("*")):
                    if not path.is_file() or path.is_symlink():
                        continue
                    rel = path.relative_to(root)
                    if "profile" in rel.parts or "__pycache__" in rel.parts or path.name.endswith(".pyc"):
                        continue
                    if path.name == "trace_view.json.gz":
                        traces.append(dict(label=label,path=str(path),sha256=sha(path),bytes=path.stat().st_size))
                        if not args.traces:
                            continue
                    elif not (path.suffix in (".py",".sh",".json",".csv",".sha256",".exit",".patch")
                              or path.name in ("run.log","train.log")):
                        continue
                    save(path, f"evidence/{label}/{rel.as_posix()}")
        for family in (ROOT / "runs").glob("*"):
            if args.only and family.name not in args.only:
                continue
            for gate in sorted(family.glob("gate-attempt-*/gate")):
                for path in gate.rglob("*"):
                    if path.is_file() and path.suffix in (".json",".exit") and "workspace" not in path.relative_to(gate).parts:
                        save(path, f"gates/{family.name}/{gate.parent.name}/{path.relative_to(gate).as_posix()}")
        for run in sorted(ROOT.iterdir()):
            if args.only or not run.is_dir() or not (run / "command.json").is_file():
                continue
            for path in [run/"command.json",run/"train.exit",run/"before.card.json",run/"after.card.json",
                         run/"micro.json",run/"run.log",run/"rank000/steps-4-6.prof"]:
                save(path, f"diagnostics/{run.name}/{path.relative_to(run).as_posix()}")
        for name,data in (("trace-index.json",traces),("receipt-sha256.json",manifest)):
            payload=json.dumps(data, indent=2).encode()
            info=tarfile.TarInfo(name)
            info.size=len(payload)
            archive.addfile(info,io.BytesIO(payload))


if __name__ == "__main__":
    main()
