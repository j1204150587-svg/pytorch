"""Run pre-registered complete NPU8 legs in order, across separate exclusive-device windows."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid

from run import FREE70, STUDY, check, digest, read, recipe_info, save


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def epoch(stamp):
    return datetime.fromisoformat(stamp).timestamp()


def file_hash(path):
    return digest(path.read_bytes())


def make_plan(families, modes):
    families_info = {}
    jobs = []
    for name in families:
        check(re.fullmatch(r"[\w-]+", name) and name not in families_info, "duplicate or invalid family")
        work = FREE70 / "runs" / name
        recipe, _ = recipe_info(work / "recipe.json")
        index = read(work / "runner-sha256.json")
        for rel, expected in index.items():
            check(file_hash(work / "runner" / rel) == expected, "changed registered runner: " + rel)
        gate = (work / "gate/comparison.json").resolve(strict=True)
        check(read(gate)["passed"], "semantic gate has not passed: " + name)
        families_info[name] = {"recipe_sha256": file_hash(work / "recipe.json"),
                               "runner_files": index, "gate": str(gate), "gate_sha256": file_hash(gate),
                               "endpoint": recipe.get("comparison") == "stock01-to-combination"}
        for mode in modes:
            for order in ("ab", "ba"):
                jobs.append({"family":name,"mode":mode,"order":order})
    return {"version":1,"families":families_info,"jobs":jobs,"warmup":10,"measure":50,
            "level0_steps":[4,5,6],"schedule_shape_count":60,
            "scheduler_sha256":file_hash(Path(__file__))}


def original_command(runner, variant, mode, token):
    stage={"A":"baseline","B":"screen"}[variant]
    cmd=[sys.executable,"-B",str(runner/"run_npu8.py"),"run","--stage",stage,"--mode",mode,
         "--approval-ref",token,"--extension-switch","RECSYS_DYNAMIC_NATIVE"]
    for arg, name, sha_arg in [
        ("candidate-patch","candidate.patch","candidate-patch"),
        ("candidate-module","overflow.py","candidate-module"),
        ("extension-patch","extension.patch","extension-patch"),
        ("extension-module","graphed_hstu.py","extension-module"),
        ("shape-schedule","dynamic_shapes_npu8.json","shape-schedule"),
        ("shared-norm-module","zero2_batch_norm.py","shared-norm")]:
        path=runner/name
        cmd.extend(["--"+arg,str(path),"--"+sha_arg+"-sha256",file_hash(path)])
    return cmd


def chip_sample(container):
    stamp=now()
    result=subprocess.run(["npu-smi","info","-t","proc-mem","-i","4","-c","0"],
                          capture_output=True,text=True,timeout=6)
    procs=[]
    for pid,name,mb in re.findall(r"Process id:\s*(\d+)\s+Process name:(.*?)\s+Process memory\(MB\):\s*(\d+)",
                                  result.stdout):
        try:
            owner="training" if container in Path(f"/proc/{pid}/cgroup").read_text() else "external"
        except OSError:
            owner="unknown"
        procs.append({"pid":int(pid),"name":name.strip(),"memory_mib":int(mb),"owner":owner})
    valid=result.returncode==0 and bool(procs or "No process in device" in result.stdout)
    return {"utc":stamp,"end_utc":now(),"valid":valid,"processes":procs,
            "diagnostic":"" if valid else (result.stdout+result.stderr)[-640:]}


def safe_sample(container):
    try:
        return chip_sample(container)
    except (OSError,subprocess.TimeoutExpired) as error:
        return {"utc":now(),"end_utc":now(),"valid":False,"processes":[],"diagnostic":str(error)}


def observe(stop, container, log, interval):
    with log.open("x",encoding="utf8",buffering=1) as output:
        while True:
            print(json.dumps(safe_sample(container)),file=output)
            if stop.wait(interval):
                break


def audit_samples(before, after, rows, max_gap):
    start=epoch(before["utc"])
    end=epoch(after["utc"])+1  # The original runner records boundary times at second resolution.
    inside=[r for r in rows if start <= epoch(r["utc"]) < end]
    touch=[start]+[epoch(r["utc"]) for r in inside]+[end]
    largest=max(b-a for a,b in zip(touch,touch[1:]))
    foreign=[r for r in inside if any(p["owner"]=="external" for p in r["processes"])]
    unknown=[r for r in inside if not r["valid"] or any(p["owner"]=="unknown" for p in r["processes"])]
    result={"start_utc":before["utc"],"end_utc":after["utc"],"sample_count":len(inside),
            "max_gap_seconds":largest,"external":foreign,"uncertain":unknown}
    if foreign:
        return "external",result
    if not inside or unknown or largest>max_gap:
        return "observation_failed",result
    return "accepted",result


def inspect_attempt(path, request, samples, source, max_gap):
    stdout=(path/"runner.log").read_text(encoding="utf8",errors="replace")
    lines=stdout.splitlines()
    check(lines and Path(lines[0]).is_absolute(), "runner did not record its new source directory")
    session=Path(lines[0]).resolve(strict=True)
    check(session.parent==FREE70.resolve(), "runner wrote outside this experiment")
    plan=read(session/"plan.json")
    variant=request["variant"]
    check(plan["approval_ref"]==request["token"] and plan["order"]==[variant] and
          plan["mode"]==request["mode"], "unregistered or imported leg")
    status=read(session/"status.json")
    receipt={"directory":str(session),"variant":variant,"runner_status":status,"finished_utc":now()}
    leg=session/("01-"+variant)
    if status["exit"]!=0:
        detail=status.get("detail","")
        blocked=False
        if status["exit"]==12 and (leg/"status.json").is_file():
            failed=read(leg/"status.json")
            marker={"preflight":"before","postflight":"after"}.get(failed.get("phase"))
            if marker:
                fuser=leg/(marker+".fuser.exit")
                holder=leg/(marker+".fuser.log")
                proven_holder=fuser.is_file() and fuser.read_text().strip()=="0" and holder.is_file() and bool(re.search(r"\d",holder.read_text()))
                container=read(session/"container.json")["id"]
                external_ids={p["pid"] for s in samples for p in s["processes"] if p["owner"]=="external"}
                text=holder.read_text().replace("/dev/davinci8","") if proven_holder else ""
                pids={int(v) for v in re.findall(r"\b\d+\b",text)}
                if "occupied by host PIDs" in detail:
                    pids.update(int(v) for v in re.findall(r"\d+",detail.partition("occupied by host PIDs")[2]))
                for pid in pids:
                    try:
                        if container not in Path(f"/proc/{pid}/cgroup").read_text():
                            external_ids.add(pid)
                    except OSError:
                        pass
                blocked=bool(pids & external_ids) or ("training container has other processes" in detail and bool(external_ids))
                receipt["observed_holders"]=sorted(pids)
                receipt["confirmed_external_pids"]=sorted(external_ids)
        return ("external" if blocked else "runner_failed"), receipt
    check(int((leg/"runner.exit").read_text())==0 and
          all(int((leg/(tag+".exit")).read_text())==0 for tag in ("preflight","train","postflight","analysis")),
          "runner leg did not pass every boundary")
    before,after=[read(leg/(b+".card.json")) for b in ("before","after")]
    for c in (before,after):
        check((c["logical_device"],c["board"],c["chip"],c["physical_id"])==(8,4,0,8) and
              c["health"]=="OK" and not c["host_pids"] and c["orphan_context"] is None,"unclean boundary")
    verdict,audit=audit_samples(before,after,samples,max_gap)
    cmd=read(leg/"command.json")
    plain=source["endpoint"] and variant=="A"
    ws="workspace-01" if plain else "workspace"
    check(cmd["workspace"].endswith("/"+session.name+"/"+ws), "wrong endpoint workspace")
    receipt.update(audit=audit,leg=str(leg),workspace_sha256=file_hash(session/(ws+".sha256")),
                   candidate_workspace_sha256=file_hash(session/"workspace.sha256"),
                   container=read(session/"container.json"), runtime=read(leg/"rank000/runtime.json"),
                   command=cmd, actual_args=read(leg/"rank000/run_arguments.json"))
    check(read(leg/"recipe.json")==read(Path(request["recipe"])), "leg used a different recipe")
    return verdict,receipt


def attempt_result(folder, source, max_gap):
    if (folder/"decision.json").is_file():
        return read(folder/"decision.json")
    req=read(folder/"request.json")
    samples=[json.loads(line) for line in (folder/"chip.jsonl").read_text().splitlines() if line]
    verdict,detail=inspect_attempt(folder,req,samples,source,max_gap)
    record={"verdict":verdict,"token":req["token"],**detail}
    save(folder/"decision.json",record)
    return record


def run_slot(slot, job, variant, source, lockfd, args):
    slot.mkdir(exist_ok=True)
    retries=0
    while True:
        attempts=sorted(p for p in slot.glob("try-*") if p.is_dir())
        if attempts:
            for previous in attempts:
                earlier=read(previous/"request.json")
                check(all(earlier[k]==v for k,v in job.items()) and earlier["variant"]==variant,
                      "an attempt does not belong to this pre-registered leg")
            for failed in attempts[:-1]:
                check(read(failed/"decision.json")["verdict"]=="external","cannot skip an earlier non-external outcome")
            prior=attempt_result(attempts[-1],source,args.max_sample_gap)
            check(prior["variant"]==variant, "unexpected prior slot variant")
            if prior["verdict"]=="accepted":
                return {**prior,"attempt":str(attempts[-1]),"attempt_count":len(attempts)}
            check(prior["verdict"]=="external", "current slot needs inspection: "+str(attempts[-1]))
            if retries:
                time.sleep(args.poll)
        idle=safe_sample(args.container_id)
        check(idle["valid"],"independent chip query failed: "+str(idle))
        if idle["processes"]:
            time.sleep(args.poll)
            continue
        work=FREE70/"runs"/job["family"]
        runner=work/"runner"
        check(file_hash(work/"recipe.json")==source["recipe_sha256"] and
              file_hash(Path(source["gate"]))==source["gate_sha256"],"recipe or gate changed before launch")
        for rel,expected in source["runner_files"].items():
            check(file_hash(runner/rel)==expected,"registered runner source changed: "+rel)
        folder=slot/f"try-{len(attempts)+1:05d}"
        folder.mkdir()
        token="ordered-"+uuid.uuid4().hex
        command=original_command(runner,variant,job["mode"],token)
        request=dict(**job,variant=variant,token=token,recipe=str(work/"recipe.json"),
                     registered_utc=now(),argv=command,shell=shlex.join(command),preliminary=idle)
        save(folder/"request.json",request)
        stop=threading.Event()
        worker=threading.Thread(target=observe,args=(stop,args.container_id,folder/"chip.jsonl",args.sample_every))
        worker.start()
        try:
            with (folder/"runner.log").open("xb") as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                                         env={**os.environ,"DYNAMIC_INFRA_RECIPE":request["recipe"]},
                                         pass_fds=(lockfd,))
            save(folder/"process.json",{"exit":completed.returncode,"utc":now()})
        finally:
            stop.set()
            worker.join()
        with (folder/"chip.jsonl").open("a",encoding="utf8") as observations:
            print(json.dumps(safe_sample(args.container_id)),file=observations)
        result=attempt_result(folder,source,args.max_sample_gap)
        print(now(),job["family"],job["mode"],job["order"],variant,folder.name,result["verdict"],flush=True)
        retries+=1


def link_tree(source,dest):
    shutil.copytree(source,dest,copy_function=os.link)


def install_pair(path, job, accepted, sources):
    from report import parse_pair, validate_controls
    family=job["family"]
    target=path/"evidence"/f"{family}-{job['mode']}-{job['order']}"
    if (target/"status.json").is_file():
        return target
    target.parent.mkdir(exist_ok=True)
    stage=target.parent/(".preparing-"+uuid.uuid4().hex)
    stage.mkdir()
    base=FREE70/"runs"/family
    recipe,switch=recipe_info(base/"recipe.json")
    first,second=accepted
    check(first["candidate_workspace_sha256"]==second["candidate_workspace_sha256"],"candidate source changed between legs")
    if not sources["endpoint"]:
        check(first["workspace_sha256"]==second["workspace_sha256"],"workspace changed between legs")
    for key in ("id","image_id","init_pid","cpu_quota","cpuset","memory_bytes","devices"):
        check(first["container"][key]==second["container"][key],"container fingerprint differs: "+key)
    gap=epoch(second["audit"]["start_utc"])-epoch(first["audit"]["end_utc"])
    check(gap>=0 and first["runtime"]==second["runtime"],"leg times overlap or runtime differs")
    done=[]
    for i,entry in enumerate(accepted,1):
        label=f"{i:02d}-{entry['variant']}"
        link_tree(Path(entry["leg"]),stage/label)
        done.append({"directory":label,"variant":entry["variant"],"source":entry["directory"],"attempt":entry["attempt"]})
    session=Path(first["directory"])
    for name in ("workspace","workspace-01","inputs"):
        if (session/name).is_dir():
            link_tree(session/name,stage/name)
    save(stage/"pair-provenance.json",{"plan":job,"gap_seconds_at_seconds_resolution":gap,
         "order":job["order"].upper(),"legs":accepted,"source":sources,"timing":"separate whole-leg windows"})
    save(stage/"status.json",{"phase":"completed","exit":0,"completed_runs":done,"utc":now()})
    parsed=parse_pair(f"r1/{job['order']}/{job['mode']}={stage}",read(base/"runner/dynamic_shapes_npu8.json"),
                      None if sources["endpoint"] else switch)
    if sources["endpoint"]:
        a,b=recipe["A"],recipe["B"]
        changes={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
        check(parsed["changed_environment"]==changes,"unexpected endpoint environment")
    for leg in parsed["legs"]:
        validate_controls(leg,recipe)
    stage.rename(target)
    return target


def main():
    import fcntl
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--registry",required=True,help="new persistent campaign name under ordered-resume; reuse it to resume")
    cli.add_argument("--families",nargs="+",default=["shape-norm-nocast-v1","stock01-to-native-rms-v2"])
    cli.add_argument("--modes",nargs="+",choices=["no-profiler","l0","l1"],default=["no-profiler","l0"])
    cli.add_argument("--sample-every",type=float,default=2.0)
    cli.add_argument("--max-sample-gap",type=float,default=8.0)
    cli.add_argument("--poll",type=float,default=4.0)
    cli.add_argument("--register-only",action="store_true")
    args=cli.parse_args()
    check(re.fullmatch(r"[\w-]+",args.registry) and len(set(args.modes))==len(args.modes),"invalid registry name or duplicate mode")
    check(0<args.poll<=60 and 0<args.sample_every<args.max_sample_gap<=60,"invalid sampling interval")
    path=FREE70/"ordered-resume"/args.registry
    path.mkdir(parents=True,exist_ok=True)
    with (path/"registry.lock").open("a") as registry_lock:
        fcntl.flock(registry_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=make_plan(args.families,args.modes)
        plan.update(registry=args.registry,sample_every=args.sample_every,max_sample_gap=args.max_sample_gap)
        spec=path/"plan.json"
        if spec.exists():
            check(read(spec)==plan,"registration, gate or source changed; inspect this campaign")
        else:
            save(spec,plan)
        if args.register_only:
            print(spec)
            return
        args.container_id=subprocess.check_output(["docker","--host","unix:///var/run/docker.sock","inspect",
            "-f","{{.Id}}","recsys-gr-ds019-npu8"],text=True).strip()
        check(re.fullmatch(r"[0-9a-f]{64}",args.container_id),"unrecognized container ID")
        entries=[]
        for ordinal,job in enumerate(plan["jobs"],1):
            pair_root=path/f"{ordinal:02d}-{job['family']}-{job['mode']}-{job['order']}"
            pair_root.mkdir(exist_ok=True)
            source=plan["families"][job["family"]]
            legs=[run_slot(pair_root/f"{i:02d}-{v}",job,v,source,registry_lock.fileno(),args)
                  for i,v in enumerate(job["order"].upper(),1)]
            destination=install_pair(path,job,legs,source)
            entries.append({**job,"exit":0,"directory":str(destination)})
            save(path/"runs.json",entries)


if __name__=="__main__":
    main()
