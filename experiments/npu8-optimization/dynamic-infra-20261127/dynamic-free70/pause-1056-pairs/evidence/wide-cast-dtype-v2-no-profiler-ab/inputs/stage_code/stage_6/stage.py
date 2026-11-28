"""Install the one-switch wide cast into the existing flat-stats workload."""
from pathlib import Path
import shutil
import sys


if __name__=="__main__":
    ws=Path(sys.argv[1])
    entry=ws/"train_deepspeed.py"
    text=entry.read_text(encoding="utf8")
    mark='        from flat_stats import install as install_dynamic_flat_stats'
    if text.count(mark)!=1 or "install_wide_norm" in text:
        raise ValueError("stage after the dynamic flat-statistics installer")
    before='        from wide_norm import install as install_wide_norm\n        install_wide_norm(rank_dir)\n'
    entry.write_text(text.replace(mark,before+mark),encoding="utf8")
    shutil.copyfile(Path(__file__).with_name("wide_norm.py"),ws/"wide_norm.py")
