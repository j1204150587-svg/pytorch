from pathlib import Path
import shutil
import sys


if __name__=="__main__":
    ws=Path(sys.argv[1])
    entry=ws/"train_deepspeed.py"
    txt=entry.read_text(encoding="utf8")
    init="    engine.train()\n"
    done='        (rank_dir / "metrics.json").write_text(json.dumps({"summary": result, "steps": metrics}, indent=2))'
    if txt.count(init)!=1 or txt.count(done)!=1:raise ValueError("unrecognized stock trainer")
    txt=txt.replace(init,init+"    from hccl_rank1_counts import Counts\n    hccl_probe=Counts()\n")
    txt=txt.replace(done,done+"\n        hccl_probe.finish(rank_dir,args.steps)")
    compile(txt,str(entry),"exec")
    entry.write_text(txt,encoding="utf8")
    shutil.copyfile(Path(__file__).with_name("counts.py"),ws/"hccl_rank1_counts.py")
