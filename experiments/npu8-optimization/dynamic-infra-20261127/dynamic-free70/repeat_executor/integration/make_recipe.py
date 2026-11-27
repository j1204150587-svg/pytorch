"""Generate a recipe for the existing locked runner using the actual package mount."""
import argparse
import json
from pathlib import Path


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--package", required=True, help="container path to build_repeat.py's package")
    cli.add_argument("--stage", required=True, help="directory with stage.py and repeat_counts.py, relative to dynamic-free70")
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--baseline", choices=("fresh","0"), default="fresh")
    a = cli.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    spec = dict(stage=a.stage, common={"PYTHONPATH":a.package},
                A={"NPU_DYNAMIC_REPEAT_EMBEDDING":a.baseline}, B={"NPU_DYNAMIC_REPEAT_EMBEDDING":"repeat"},
                activation=[dict(file="repeat_embedding_counts.json",
                                 A=dict(mode=a.baseline, passed=True, steps="$steps"),
                                 B=dict(mode="repeat", passed=True, steps="$steps"))])
    a.output.write_text(json.dumps(spec, indent=2)+"\n")
