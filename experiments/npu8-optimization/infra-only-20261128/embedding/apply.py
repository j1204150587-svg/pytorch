"""Stage both embedding call sites and the shared key/descriptor implementation."""

from pathlib import Path
import shutil
import subprocess
import sys


def apply(root):
    here = Path(__file__).resolve().parent
    for name in ("dynamic_embedding_scope.h", "dynamic_embedding_scope.inc"):
        if (root / "op_plugin/utils" / name).exists():
            raise ValueError(f"unexpected preexisting header {name}")
    cmd = ["git", "-c", "core.autocrlf=false", "apply", "--no-index", "--whitespace=nowarn"]
    subprocess.run([*cmd, "--check", str(here / "embedding.patch")], cwd=root, check=True)
    subprocess.run([*cmd, str(here / "embedding.patch")], cwd=root, check=True)
    for suffix in ("h", "inc"):
        shutil.copyfile(here / f"index_scope.{suffix}", root / "op_plugin/utils" / f"dynamic_embedding_scope.{suffix}")


if __name__ == "__main__":
    apply(Path(sys.argv[1]))
