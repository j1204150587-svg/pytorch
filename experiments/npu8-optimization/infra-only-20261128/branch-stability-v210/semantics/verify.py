"""Verify real rank/standalone process counters and the original numerical gate receipts."""
import argparse
from datetime import datetime,timedelta
import json
from pathlib import Path
import re

from probe import COUNTER, COUNTER_SHA, HEAD, SWITCHES

def read(p): return json.loads(p.read_bytes())
def native(folder,pid,lib):
    result={}
    for direction in ("backward","forward"):
        sym="npu_embedding_"+direction+"_rebase_counter"
        row=read(folder/f"{sym}-{pid}.json")
        assert row["pid"]==pid and row["symbol"]==sym and row["library"]==lib,row
        result[direction]=[row[k] for k in ("accepted","fallback","keys","descriptors")]
    return result

def wake(log,arm):
    lines=re.findall(r"^PTA_QUEUE_PRODUCER_WAKE device=0 .*$",log.read_text(errors="replace"),re.M)
    assert len(lines)==1,(log,lines)
    row={k:int(v) for k,v in re.findall(r"(\w+)=(\d+)",lines[0])}
    assert row["enabled"]==(arm=="B") and row["drained"]>0 and not row["waiters"],row
    if arm=="A":assert row["writes_without_observed_waiter"]>0,row
    else:
        assert row["writes_without_observed_waiter"]==0,row
        if not row["registrations"]:assert not row["writes"],row
    return row

def truth(settings,arm):
    assert all(settings.get(k)==str(int(arm=="B")) for k in SWITCHES),settings
    assert settings["NPU_DYNAMIC_REBASE_COUNT"]==settings["TORCH_NPU_QUEUE_WAKE_AUDIT"]=="1"
    assert settings["ACLNN_CACHE_LIMIT"]=="65536"

def card(p):
    for at in ("before","after"):
        v=read(p/(at+".card.json"))
        assert (v["logical_device"],v["board"],v["chip"],v["physical_id"],v["health"],v["host_pids"])==(8,4,0,8,"OK",[]),v

def probes(folder):
    out={};checked=[]
    journal=Path(__file__).resolve().parents[1]/"chip.jsonl"
    snapshots=[json.loads(line) for line in journal.read_bytes().splitlines() if line] if journal.is_file() else []
    for arm in "AB":
        outer=folder/arm
        assert (outer/"train.exit").read_text().strip()=="0"
        card(outer)
        workers_file=folder/(arm+"-processes.json")
        workers=read(workers_file) if workers_file.is_file() else {}
        left=datetime.fromisoformat(read(outer/"before.card.json")["utc"])
        right=datetime.fromisoformat(read(outer/"after.card.json")["utc"])+timedelta(seconds=1)
        rows=[v for v in snapshots if left<=datetime.fromisoformat(v["utc"])<right]
        seen={p["pid"] for v in rows for p in v["processes"]}
        known={v["host_pid"] for v in workers.values()}
        if len(workers)==2:assert not(seen-known), (arm,seen-known)
        assert all(v["valid"] for v in rows)
        for kind in ("embedding","overflow"):
            leg=outer/kind
            assert (leg/"exit").read_text().strip()=="0"
            op=read(leg/"result.json");receipt=read(leg/"receipt.json")
            truth(receipt["settings"],arm)
            if kind in workers:assert workers[kind]["container_pid"]==receipt["pid"]
            assert receipt["arm"]==arm and receipt["target_head"]==HEAD and receipt["counter_sha256"]==COUNTER_SHA
            counts=native(leg,receipt["pid"],receipt["library"])
            q=wake(leg/"run.log",arm)
            if arm=="A": assert all(a==[0,0,0,0] for a in counts.values()),counts
            else: assert all(a[0]>0 and a[2]>=a[0] and a[3]>0 for a in counts.values()),counts
            if kind=="embedding":
                assert op["exact_cpu"] and len(op["checks"])==64
                assert {tuple(x[:2]) for x in op["checks"]}=={(1,19),(2,14),(4,32)}
                for direction,factor in (("forward",2),("backward",1)):
                    before,target,last=[op[k][direction] for k in ("before","target","fallback")]
                    if arm=="B":
                        assert target[0]-before[0]==64 and last[1]-target[1]==1 and target[2]-before[2]==64*factor
                    else: assert before==target==last==[0,0,0,0]
                summary=dict(exact_cpu=True,per_direction_cases=64,shapes=[[1,19],[2,14],[4,32]],fallback_rank=2)
            else:
                assert op["library_sha"]=={receipt["library"]:receipt["sha256"]} and op["arm"]==arm
                assert [(v["case"],v["skipped"]) for v in op["steps"]]==[("finite",False),("inf",True),("finite",False),("nan",True),("finite",False),("-inf",True),("finite",False)]
                for step in op["steps"]:
                    assert (step["before"]==step["after"])==step["skipped"]
                    if not step["skipped"]:assert step["before"]["master"]!=step["after"]["master"]
                checked.append(op["steps"])
                summary=dict(finite_updates=4,nonfinite_skips=3,scan=op["scan"])
            out[kind+"-"+arm]=dict(receipt=receipt,counters=counts,queue=q,checks=summary,card_snapshots=len(rows),
                matched_worker=workers.get(kind),seen_in_chip_samples=workers.get(kind,{}).get("host_pid") in seen,
                unidentified_chip_pids=sorted(seen-known))
    assert checked[0]==checked[1],"A/B loss, gradient hash, master/BF16 weights or optimizer states differ"
    assert len({v["receipt"]["sha256"] for v in out.values()})==1
    return out

def audit(pair,sha):
    result=read(pair/"result.json");assert result["exit"]==0 and result["audit_passed"] and result["audited_steps"]==8
    assert result["gradient_equal"] and result["weights_and_adamw_equal"] and result["norm_max_delta"]==0
    plan=read(pair/"recipe.json");raw=Path(result["directory"]);arms={}
    for arm in "AB":
        label=("01-" if arm=="A" else "02-")+arm
        worker=read(pair/label/"worker.json");truth(worker["env"],arm)
        assert str(COUNTER) in worker["libraries"] and worker["env"]["LD_PRELOAD"]==str(COUNTER)
        lib,=[p for p in worker["libraries"] if p.endswith("/libtorch_npu.so")]
        assert str(Path(lib).parents[2]) in worker["env"]["PYTHONPATH"].split(":")
        folder=raw/label;card(folder)
        cnt=native(folder,worker["container_pid"],lib)
        for direction,w in (("forward",2),("backward",1)):
            eligible,fallback,keys,desc=cnt[direction]
            if arm=="A":assert cnt[direction]==[0,0,0,0]
            else:assert eligible==8*1024 and keys==w*eligible and desc>0,(direction,cnt[direction])
        q=wake(folder/"run.log",arm)
        # The host-pair recipe shape may be one common recipe, or separate A/B endpoint recipes.
        observed={}; nested=[plan]
        while nested:
            node=nested.pop()
            if not isinstance(node,dict):continue
            if isinstance(node.get("libraries"),dict):observed.update(node["libraries"])
            nested.extend(v for v in node.values() if isinstance(v,dict))
        if lib not in observed:
            ds=read(pair/label/"dispatch.json")
            observed.update(ds.get("libraries",{}))
        assert observed[lib]==sha and observed[str(COUNTER)]==COUNTER_SHA
        ov=read(folder/"rank000/overflow_scan.json")
        assert ov["valid"] and ov["fast_calls"]==8 and ov["fallback_calls"]==ov["errors"]==0
        states=read(folder/"rank000/state_audit.json")
        assert all(i==8 for i in states["adamw_steps"]) and states["adamw_steps"]
        schedule=[(s["batch_size"],s["seq_len"]) for s in read(folder/"rank000/metrics.json")["steps"]]
        assert len(set(schedule))==8
        arms[arm]=dict(pid=worker["pid"],container_pid=worker["container_pid"],library=lib,sha256=sha,counts=cnt,queue=q,
                       schedule=schedule,overflow=ov,adamw_steps=states["adamw_steps"])
    assert arms["A"]["schedule"]==arms["B"]["schedule"] and arms["A"]["library"]==arms["B"]["library"]
    return dict(source=HEAD,arms=arms,loss_max_delta=result["loss_max_delta"],norm_max_delta=result["norm_max_delta"],
                gradient_equal=True,weights_and_adamw_equal=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probes",type=Path);p.add_argument("--audit",type=Path)
    p.add_argument("--sha256");p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();out={}
    assert args.probes or args.audit
    if args.probes:out["probes"]=probes(args.probes)
    if args.audit:
        assert args.sha256 and len(args.sha256)==64
        out["original_training_audit"]=audit(args.audit,args.sha256)
    if args.probes and args.audit:
        assert all(v["receipt"]["sha256"]==args.sha256 for v in out["probes"].values())
    args.output.write_text(json.dumps(out,indent=2)+"\n")
    print(args.output,flush=True)

if __name__=="__main__":main()

