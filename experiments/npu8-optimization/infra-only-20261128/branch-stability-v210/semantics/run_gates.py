"""Run both existing diagnostics in one physical-card reservation per arm."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading

HERE=Path(__file__).resolve().parent
STUDY=HERE.parents[1]
ART=STUDY.parent

def observe_processes(outer,stop,observed):
    while not stop.wait(.25) and len(observed)<2:
        todo={kind:outer/kind/"receipt.json" for kind in ("embedding","overflow") if kind not in observed}
        todo={kind:json.loads(file.read_bytes()) for kind,file in todo.items() if file.exists() and file.stat().st_size}
        if not todo: continue
        rows=subprocess.check_output(["docker","top","recsys-gr-ds019-npu8","-eo","pid"],text=True).splitlines()[1:]
        for row in rows:
            proc=Path("/proc")/row.strip()
            try:
                line=next(x for x in (proc/"status").read_text().splitlines() if x.startswith("NSpid:"))
                nspid=int(line.split()[-1]);args=(proc/"cmdline").read_bytes().split(b"\0")
                for kind,metadata in todo.items():
                    target="/artifacts/"+(outer/kind/"result.json").relative_to(ART).as_posix()
                    if nspid==metadata["pid"] and metadata["original"].encode() in args and target.encode() in args:
                        observed[kind]=dict(host_pid=int(proc.name),container_pid=nspid,original=metadata["original"])
            except (FileNotFoundError,ProcessLookupError): pass

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package",required=True,help="container overlay parent containing torch_npu")
    ap.add_argument("--sha256",required=True,help="verified branch libtorch_npu.so SHA-256")
    ap.add_argument("--dest",type=Path,required=True,help="new receipts directory inside host artifacts")
    args=ap.parse_args()
    dest=args.dest.resolve()
    dest.relative_to(ART)
    dest.mkdir(parents=True,exist_ok=False)
    inside="/artifacts/"+(HERE/"probe.py").relative_to(ART).as_posix()
    for arm in "AB":
        cmd=["flock","-x","/run/lock/recsys-gr-ds019-npu8-infra-study.lock",sys.executable,"-B",
             str(STUDY/"operator_probe.py"),"--package",args.package,str(dest/arm),
             inside,"both",arm,"@WORK","@OUT",args.package,args.sha256]
        print(" ".join(cmd),flush=True)
        stop=threading.Event();observed={}
        thread=threading.Thread(target=observe_processes,args=(dest/arm,stop,observed),daemon=True)
        thread.start()
        try: subprocess.run(cmd,check=True)
        finally:
            stop.set();thread.join()
            (dest/(arm+"-processes.json")).write_text(json.dumps(observed,indent=2)+"\n")
    subprocess.run([sys.executable,"-B",str(HERE/"verify.py"),"--probes",str(dest),"--output",str(dest/"verified.json")],check=True)

if __name__=="__main__":main()

