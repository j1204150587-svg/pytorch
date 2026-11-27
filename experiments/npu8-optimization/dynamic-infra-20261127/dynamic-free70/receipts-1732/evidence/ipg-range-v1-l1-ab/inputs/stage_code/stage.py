"""Patch the isolated dynamic training entry after the baseline ZeRO optimizer is created."""
from pathlib import Path
import shutil
import sys


if __name__=="__main__":
    ws=Path(sys.argv[1])
    path=ws/"train_deepspeed.py"
    src=path.read_text(encoding="utf8")
    anchor="    engine.train()\n"
    metrics='        (rank_dir / "metrics.json").write_text(json.dumps({"summary": result, "steps": metrics}, indent=2))'
    if src.count(anchor)!=1 or src.count(metrics)!=1 or "install_ipg_range" in src:
        raise ValueError("unexpected dynamic trainer")
    src=src.replace(anchor,anchor+"    from range_copy import install as install_ipg_range\n"
                    "    ipg_range=install_ipg_range(engine)\n")
    src=src.replace(metrics,metrics+"\n        ipg_range.finish(rank_dir)")
    compile(src,str(path),"exec")
    path.write_text(src,encoding="utf8")
    shutil.copyfile(Path(__file__).with_name("range_copy.py"),ws/"range_copy.py")
