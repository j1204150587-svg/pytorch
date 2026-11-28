"""Export completed paired receipts. Raw profiler trees remain at the indexed host locations."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parent

def add_raw(archive, run):
    wanted = {"container.json","plan.json","workspace.sha256","status.json","evidence.sha256",
              "runner.exit","train.exit","analysis.exit","postflight.exit","preflight.exit","validated.json",
              "command.json","run.log","metrics.json","runtime.json","state_audit.json","overflow_scan.json",
              "before.card.json","after.card.json","step_trace_time.csv","trace_view.json.gz"}
    for file in sorted(run.rglob("*")):
        if file.is_file() and (file.name in wanted or file.name.startswith(("npu_addmm_rebase_counter-", "npu_mm_rebase_counter-",
                "npu_embedding_forward_rebase_counter-", "npu_embedding_backward_rebase_counter-"))):
            archive.add(file, "raw/"+run.name+"/"+file.relative_to(run).as_posix(),recursive=False)
    for file in sorted(run.rglob("kernel_details.csv")):
        archive.add(file,"raw/"+run.name+"/"+file.relative_to(run).as_posix(),recursive=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("pairs", nargs="+")
    args=ap.parse_args()
    with tarfile.open(args.out,"w:gz",compresslevel=3) as archive:
        for name in args.pairs:
            pair = ROOT/"pairs"/name
            r=json.loads((pair/"result.json").read_bytes())
            if r["exit"]:
                raise ValueError(r)
            archive.add(pair,"pairs/"+name)
            add_raw(archive,Path(r["directory"]))
        for name in ("reference/workspace.sha256","archived-runner/dynamic_shapes_npu8.json","chip.jsonl"):
            if (ROOT/name).exists():
                archive.add(ROOT/name,name)
if __name__ == "__main__":
    main()

