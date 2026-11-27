"""Index measured traces, raw CSVs and commands from validated host receipts."""
import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evidence", type=Path)
    args = ap.parse_args()
    root = args.evidence.resolve()
    records = []
    for p in sorted(root.glob("*/receipts-manifest.json")):
        rows = json.loads(p.read_text())
        for row in rows:
            name = Path(row["member"]).name
            if name in {"metrics.json", "trace_view.json.gz", "step_trace_time.csv", "command.json",
                        "dynamic_shapes_npu8.json","comparison.json","native_activation.json","semantic.json"}:
                records.append(row)
    (root / "source-index.json").write_text(json.dumps(records,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(len(records), "records")

if __name__ == "__main__":
    main()

