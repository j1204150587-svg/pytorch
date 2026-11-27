"""Index measured traces, raw CSVs and commands from validated host receipts."""
import argparse
import hashlib
import json
from pathlib import Path

RAW_FILES = {
    "metrics.json", "trace_view.json", "trace_view.json.gz", "step_trace_time.csv",
    "command.json", "dynamic_shapes_npu8.json", "comparison.json", "native_activation.json",
    "semantic.json", "micro.json", "scheduler.csv", "scheduler.json", "target.json", "clock.jsonl",
    "before.card.json", "after.card.json", "runtime.json", "model_summary.json",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evidence", type=Path)
    args = ap.parse_args()
    root = args.evidence.resolve()
    records = {}
    manifests = [root / "receipts-manifest.json", *sorted(root.glob("*/receipts-manifest.json"))]
    for p in manifests:
        if not p.is_file():
            continue
        rows = json.loads(p.read_text(encoding="utf-8"))
        for row in rows:
            member = row["member"]
            file = Path(member)
            raw_profile_table = file.suffix == ".csv" and "profile" in file.parts
            if file.name in RAW_FILES or raw_profile_table:
                if member in records and records[member]["sha256"] != row["sha256"]:
                    raise ValueError(f"conflicting receipts for {member}")
                records[member] = row
    extra = root / "diagnostic-sources.json"
    if extra.is_file():
        for row in json.loads(extra.read_text(encoding="utf-8")):
            local = root / row["member"]
            with local.open("rb") as src:
                digest = hashlib.file_digest(src, "sha256").hexdigest()
            if digest != row["sha256"]:
                raise ValueError(f"invalid diagnostic source {local}")
            records[row["member"]] = {**row, "bytes": local.stat().st_size}
    ordered = [records[name] for name in sorted(records)]
    (root / "source-index.json").write_text(json.dumps(ordered, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    catalog = ["# 原始 trace 与缓存查找记录", "",
               "完整文件在实测宿主保留。下列路径与本地证据按 SHA-256 对齐，可用 `sha256sum` 校验。", ""]
    for row in ordered:
        if not row["member"].endswith(("trace_view.json", "trace_view.json.gz", "raw-lookup-markers.tgz")):
            continue
        catalog.extend([f'### {row["member"]}', "", f'宿主文件 `{row["source"]}`。',
                        f'共 {row["bytes"]:,} 字节，SHA-256 为 `{row["sha256"]}`。', ""])
    (root / "raw-files.md").write_text("\n".join(catalog), encoding="utf-8")
    print(len(records), "records")

if __name__ == "__main__":
    main()

