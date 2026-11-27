"""Specialize the archived dynamic runner for isolated native pairs."""
from pathlib import Path
import sys

def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(old)
    return text.replace(old, new)

def main(path):
    s = path.read_text()
    s = replace(s, 'STUDY = "dynamic-shapes-20261127"', 'STUDY = "dynamic-infra-20261127"')
    s = replace(s, 'if args.extension_patch and args.stage not in {"ab", "ba", "screen"}:', 'if args.extension_patch and args.stage not in ORDERS:')
    s = replace(s, 'if args.stage in {"ab", "ba", "screen"}:', 'if args.stage in {"ab", "ba", "screen"} or args.extension_patch:')
    s = replace(s, 'fcntl.LOCK_EX | fcntl.LOCK_NB', 'fcntl.LOCK_EX')
    s = replace(s, '    docker_env = [item for name, value in env.items()', '''    recipe_file = os.environ.get("DYNAMIC_INFRA_RECIPE")
    if recipe_file:
        config = json.loads(Path(recipe_file).read_text())
        env.pop("RECSYS_DYNAMIC_NATIVE", None)
        env.update(config.get("common", {}))
        env.update(config.get(variant, {}))
        if config.get("diagnostic"):
            env["NPU_CACHE_DIAG_DIR"] = inner_out
        write_json(out / "recipe.json", config)
    docker_env = [item for name, value in env.items()''')
    s = replace(s, '            for name in ("run_npu8.py", "evidence.py"):', '''            recipe_file = os.environ.get("DYNAMIC_INFRA_RECIPE")
            if recipe_file and json.loads(Path(recipe_file).read_text()).get("diagnostic"):
                sys.path.insert(0, str(HERE.parent / "cache"))
                from mark_steps import mark
                mark(workspace / "train_deepspeed.py")
                hashes = files_sha256(workspace)
                write_sha256(session / "workspace.sha256", hashes)
            for name in ("run_npu8.py", "evidence.py"):''')
    path.write_text(s)

if __name__ == "__main__":
    main(Path(sys.argv[1]))

