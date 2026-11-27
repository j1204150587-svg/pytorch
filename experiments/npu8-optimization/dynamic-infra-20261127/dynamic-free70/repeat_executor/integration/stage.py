"""Write the native counters after the dynamic eager training loop completes."""
import argparse
from pathlib import Path
import shutil


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("workspace", type=Path)
    dest = cli.parse_args().workspace
    entry = dest / "train_deepspeed.py"
    txt = entry.read_text(encoding="utf8")
    anchor = '        (rank_dir / "metrics.json").write_text(json.dumps({"summary": result, "steps": metrics}, indent=2))'
    if txt.count(anchor) != 1 or "write_repeat_counts" in txt:
        raise ValueError("unexpected dynamic training source")
    addition = '\n        from repeat_counts import write_repeat_counts\n        write_repeat_counts(rank_dir, args.steps)'
    shutil.copyfile(Path(__file__).with_name("repeat_counts.py"), dest / "repeat_counts.py")
    entry.write_text(txt.replace(anchor, anchor + addition), encoding="utf8")
