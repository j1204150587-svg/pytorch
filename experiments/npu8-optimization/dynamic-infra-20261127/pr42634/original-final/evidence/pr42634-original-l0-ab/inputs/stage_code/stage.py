"""Attach a launch receipt to a copy of the uncompiled HSTU training script."""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    dest = parser.parse_args().workspace
    entry = dest / "train_deepspeed.py"
    text = entry.read_text(encoding="utf8")
    if "launch_receipt = Receipt()" in text:
        raise ValueError("workspace already instrumented")
    changes = (
        ("    import torch_npu\n", "    from fast_launch_receipt import Receipt\n    launch_receipt = Receipt()\n"),
        ("            for step, (batch_size, sequence_length) in enumerate(shapes):\n",
         "                launch_receipt.begin(step, (batch_size, sequence_length))\n"),
        ('        (rank_dir / "metrics.json").write_text(json.dumps({"summary": result, "steps": metrics}, indent=2))\n',
         '        launch_receipt.finish(rank_dir / "pr42634_receipt.json", expected_steps=args.steps)\n'))
    for anchor, addition in changes:
        if text.count(anchor) != 1:
            raise ValueError("unexpected training source at " + anchor)
        text = text.replace(anchor, anchor + addition)
    shutil.copyfile(Path(__file__).with_name("fast_launch_receipt.py"), dest / "fast_launch_receipt.py")
    entry.write_text(text, encoding="utf8")


if __name__ == "__main__":
    main()
