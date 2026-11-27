"""Package only completed Free-target pairs and a passed semantic gate using the established receipt format."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from run import FREE70, STUDY, check, read


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("recipe",type=Path)
    cli.add_argument("--output",type=Path,default=FREE70/"receipts")
    cli.add_argument("--only",action="append",help="restrict to MODE/ORDER or gate")
    args=cli.parse_args()
    name=args.recipe.stem
    work=FREE70/"runs"/name
    args.output.mkdir(parents=True,exist_ok=True)
    jobs=[]
    if (work/"runs.json").is_file():
        for row in read(work/"runs.json"):
            if row["exit"]==0:
                jobs.append((row["mode"]+"/"+row["order"],Path(row["directory"])))
    if (work/"gate/comparison.json").is_file() and read(work/"gate/comparison.json")["passed"]:
        jobs.append(("gate",(work/"gate").resolve()))
    if args.only:
        jobs=[(job,path) for job,path in jobs if job in args.only]
        check(set(args.only)=={job for job,path in jobs},"requested pair has no completed receipt")
    check(len(jobs)==len({tag for tag,path in jobs}),"duplicate success for one job")
    for job,source in jobs:
        label=name+"-"+job.replace("/","-")
        target=args.output/(label+".tgz")
        if target.is_file():
            continue
        with tempfile.TemporaryDirectory(dir=args.output) as tmp:
            partial=Path(tmp)/(label+".tgz")
            subprocess.run([sys.executable,"-B",str(STUDY/"reporting/collect_receipts.py"),
                            "--run",f"{label}={source}","--out",str(partial)],check=True)
            partial.replace(target)
        print(target,flush=True)


if __name__=="__main__":
    main()
