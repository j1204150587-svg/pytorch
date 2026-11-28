"""Install the opt-in flat stats adapter after the existing 07 prerequisites."""
import argparse
from pathlib import Path
import shutil


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace",type=Path)
    args=parser.parse_args()
    target=args.workspace/"train_deepspeed.py"
    source=target.read_text(encoding="utf8")
    anchor="        norm_stats = install_norm(engine.optimizer)\n"
    if source.count(anchor)!=1 or "from flat_stats import install" in source:
        raise ValueError("expected clean dynamic trainer with 07 batched norm")
    source=source.replace(anchor,anchor+
        "    if os.environ.get('RECSYS_DYNAMIC_FLAT_STATS', 'off') != 'off':\n"
        "        from flat_stats import install as install_dynamic_flat_stats\n"
        "        install_dynamic_flat_stats(engine.optimizer, norm_stats, rank_dir)\n")
    dst=args.workspace/"flat_stats.py"
    if dst.exists():
        raise ValueError(f"refusing to overwrite {dst}")
    shutil.copyfile(Path(__file__).with_name("flat_stats.py"),dst)
    target.write_text(source,encoding="utf8",newline="\n")


if __name__=="__main__":
    main()
