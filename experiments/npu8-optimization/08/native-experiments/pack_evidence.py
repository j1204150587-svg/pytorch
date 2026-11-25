"""Pack the small experiment receipts that are sufficient to recompute reported metrics."""

from pathlib import Path
import sys
import tarfile


HERE = Path(__file__).resolve().parent
RECEIPTS = {
    "candidate-runtime.json", "candidate-wheel.sha256", "source.sha", "build.exit", "finish-build.exit",
    "result.json", "metrics.json", "step_trace_time.csv", "train.exit", "command.json", "correctness.json",
    "hstu_graph.json", "pta_wrapper.json", "before.card.json", "after.card.json", "before.usages.log",
    "after.usages.log", "runtime.json", "effective_deepspeed_config.json",
    "trace-markers.json",
}


def main():
    destination = HERE / "evidence.tar.gz"
    for directory in sys.argv[1:]:
        assert Path(directory).name == directory and (HERE / directory).is_dir()
    paths = [path for path in HERE.iterdir() if path.name in RECEIPTS and path.is_file()]
    for directory in sys.argv[1:]:
        paths.extend(path for path in (HERE / directory).rglob("*")
                     if path.name in RECEIPTS and path.is_file() and not path.is_symlink())
    with tarfile.open(destination, "x:gz") as archive:
        for path in sorted(paths):
            archive.add(path, arcname=path.relative_to(HERE))


if __name__ == "__main__":
    main()
