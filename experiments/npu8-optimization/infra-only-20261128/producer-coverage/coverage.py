"""Reconstruct all-stream holes and intersect unique physical successors with two producer lanes."""
from bisect import bisect_left,bisect_right
from collections import Counter,defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
sys.path.insert(0,str(STUDY.parent/"dynamic-infra-20261127/analysis"))
from analyze_dynamic_trace import Span,Lanes,enclosing_aten,ns,is_work
sys.path.insert(0,str(STUDY/"div-geometry"))
from trace_budget import read,exclusive_cpu,partition


def merge(pairs):
    result=[]
    for a,b in sorted(pairs):
        if b<=a: continue
        if result and result[-1][1]>=a:
            result[-1]=(result[-1][0],max(b,result[-1][1]))
        else: result.append((a,b))
    return result


class Mask:
    def __init__(self,pairs):
        self.points=[x for pair in merge(pairs) for x in pair]
    def inside(self,t):
        return bisect_right(self.points,t)%2!=0
    def cuts(self,l,r):
        return self.points[bisect_right(self.points,l):bisect_left(self.points,r)]


class Curve:
    """Exact extrema of a sum of overlap lengths under one shared constant clock offset."""
    def __init__(self,lo,hi):
        self.lo,self.hi=lo,hi
        self.start=self.slope=0
        self.events=Counter()
    def add(self,l,r,a,b):
        self.start+=max(0,min(r+self.lo,b)-max(l+self.lo,a))
        for t,ds in ((a-r,1),(b-r,-1),(a-l,-1),(b-l,1)):
            if t<=self.lo: self.slope+=ds
            elif t<self.hi: self.events[t]+=ds
    def result(self):
        value=self.start
        # Use an actual zero-offset knot even if no observation boundary is exactly zero.
        self.events[0]+=0
        self.events[self.hi]+=0
        slope,old=self.slope,self.lo
        minimum=maximum=(value,old)
        base=self.start
        nominal=None
        for t,ds in sorted(self.events.items()):
            value+=(t-old)*slope
            if value<minimum[0]:minimum=(value,t)
            if value>maximum[0]:maximum=(value,t)
            if t==0:nominal=value
            if t==self.hi:break
            old=t
            slope+=ds
        return dict(earliest_ms=base/1e6,nominal_ms=nominal/1e6,latest_ms=value/1e6,
                    min_ms=minimum[0]/1e6,max_ms=maximum[0]/1e6,
                    min_at_offset_ns=minimum[1],max_at_offset_ns=maximum[1])


def run():
    path=STUDY/"evidence/raw/20261128T043752Z-ab-l1-7m6ais34/02-B/trace_view.json.gz"
    csv_dir=STUDY/"embedding/analysis/emb-fwd/B"
    with gzip.open(path) as f: z=json.load(f)
    ev=z if isinstance(z,list) else z["traceEvents"]
    pids={e["args"]["name"]:e["pid"] for e in ev if e["name"]=="process_name"}
    cpu,enq,deq,api,steps=[],{},{},[],{}
    anchor,src,task=defaultdict(list),defaultdict(list),{}
    for e in ev:
        n=e["name"]
        if e.get("cat")=="HostToDevice" and e["ph"]=="s" and e["pid"]==pids["CANN"]:
            src[str(e["id"])].append((e["tid"],ns(e["ts"])))
        if e.get("ph")!="X":continue
        s=Span.read(e)
        if n.startswith("ProfilerStep#"):steps[int(n.split("#")[1])]=s
        if e.get("cat")=="cpu_op" and n.startswith("aten::"):cpu.append(s)
        if e.get("cat") in ("enqueue","dequeue"):
            (enq if e["cat"]=="enqueue" else deq)[s.args["correlation_id"]]=s
        if e["pid"]==pids["CANN"]:
            anchor[(s.tid,s.start,s.args.get("connection_id"))].append(s)
            if n.startswith(("AscendCL@aclrtLaunchKernel","AscendCL@aclrtMemcpy")):api.append(s)
        if e["pid"]==pids["Ascend Hardware"] and is_work(s.args.get("Task Type","")):
            key=(s.start,str(s.args["Physic Stream Id"]),str(s.args.get("Task Id")))
            if key in task: raise ValueError(key)
            task[key]=s
    qrows=read(csv_dir/"queue_calls.csv.gz")
    seen=Counter((r["role"],int(r["producer_tid"])) for r in qrows)
    primary={max((v,t) for (k,t),v in seen.items() if k==role)[1] for role in ("main","worker")}
    owners=enclosing_aten(cpu,enq.values())
    apis=Lanes(api)
    enql=Lanes(enq.values())
    coverage=[]
    for q,v in owners.items():
        if v.tid in primary:coverage.append((v.start,v.end))
    # Direct queue/API activity is observable even if there is no ATen scope.
    direct=[(s.start,s.end) for s in [*enq.values(),*api] if s.tid in primary]
    inner=Mask(coverage+direct)
    actual_ancestors=[s for s in cpu if s.tid in primary and enql.children(s)]
    names={s.name for s in actual_ancestors}
    broad=Mask([(s.start,s.end) for s in cpu if s.tid in primary and s.name in names]+direct)
    all_aten=Mask([(s.start,s.end) for s in cpu if s.tid in primary]+direct)
    consumer_busy=Mask([(s.start,s.end) for s in deq.values()])
    if set(v.tid for v in actual_ancestors)!=primary:raise ValueError(primary)
    exclusive={tid:exclusive_cpu(ev,tid) for tid in primary}
    clocks=json.loads((HERE/"clock.json").read_bytes())
    gaps=read(csv_dir/"device_gaps.csv.gz")
    queues={str(k):v for k,v in enq.items()}
    ds={str(k):v for k,v in deq.items()}
    own={str(k):v for k,v in owners.items()}
    table=[]
    output={"trace_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"producer_tids":sorted(primary),
            "other_producers":{str(t):dict(Counter(r["op"] for r in qrows if int(r["producer_tid"])==t))
                               for role,t in seen if t not in primary},"steps":{}}
    for step,scope in sorted(steps.items()):
        clock=clocks["ProfilerStep#"+str(step)]
        lo,hi=clock["clock_lower_ns"],clock["clock_upper_ns"]
        curves=defaultdict(lambda:Curve(lo,hi))
        nom=Counter()
        cpu_labels=Counter()
        succ_ops=Counter()
        pairs=Counter()
        own_by_role=defaultdict(Counter)
        for g in (x for x in gaps if int(x["step"])==step):
            l=scope.start+int(g["step_offset_ns"])
            r=l+int(g["gap_ns"])
            this=Counter()
            row=dict(step=step,offset_ns=l-scope.start,gap_ns=r-l,stream=g["stream"],task=g["task"],task_id=g["task_id"],
                     queue_id=g["queue_id"],role=g["role"],op=g["op"],state=g["state"])
            if g["state"]!="linked":
                this["unlinked"]=r-l
                this["group_remaining"]=r-l
                curves["unlinked"].add(l,r,l+lo,r+hi)
                curves["group_remaining"].add(l,r,l+lo,r+hi)
            else:
                cid=g["queue_id"]
                q,d,owner=queues[cid],ds[cid],own.get(cid)
                t=task[r,g["stream"],g["task_id"]]
                tid,st=src[g["flow_id"]][0]
                a,=anchor[(tid,st,t.args.get("connection_id"))]
                launch,=apis.children(a)
                publish=min(q.end,d.start)
                if q.tid not in primary or not q.start<=publish<=d.start<=launch.start<=launch.end:
                    raise ValueError((g,q,d,launch))
                left,right=l+lo,r+hi
                cuts=[left,right,*(p for p in (owner.start if owner else left,q.start,publish,d.start,launch.start,launch.end) if left<p<right),
                      *inner.cuts(left,right),*broad.cuts(left,right),*all_aten.cuts(left,right),*consumer_busy.cuts(left,right)]
                cuts=sorted(set(cuts))
                row.update(producer_tid=q.tid,consumer_tid=d.tid,aten=owner.name if owner else "",
                           aten_start_ns=owner.start-scope.start if owner else "",enqueue_begin_ns=q.start-scope.start,
                           accepted_by_ns=publish-scope.start,dequeue_start_ns=d.start-scope.start,
                           acl_start_ns=launch.start-scope.start,acl_return_ns=launch.end-scope.start)
                for a,b in zip(cuts,cuts[1:]):
                    if a>=launch.end: label="acl_returned"
                    elif a>=launch.start: label="acl_in_progress"
                    elif a>=d.start:label="dequeued_before_acl"
                    elif a>=publish:label="accepted_waiting_consumer"
                    elif a>=q.start:label="enqueue_in_progress"
                    elif not owner:label="pre_enqueue_unknown_owner"
                    elif a>=owner.start:label="successor_aten_before_enqueue"
                    elif inner.inside(a):label="before_successor_other_submitting_aten"
                    else:label="before_successor_both_outside_actual_submitters"
                    applicable=[label]
                    if label not in ("acl_returned","before_successor_both_outside_actual_submitters"):
                        applicable.append("group_remaining")
                    if a>=publish:applicable.append("group_accepted_by_pta")
                    if label=="before_successor_both_outside_actual_submitters":
                        if not broad.inside(a):applicable.append("subset_outside_any_observed_submitting_name")
                        if not all_aten.inside(a):applicable.append("subset_outside_any_aten")
                        applicable.append("subset_other_dequeue_busy" if consumer_busy.inside(a) else "subset_outside_recorded_dequeue")
                    clipped_l=max(l,a)
                    clipped_r=min(r,b)
                    for k in applicable:
                        curves[k].add(l,r,a,b)
                        this[k]+=max(0,clipped_r-clipped_l)
                    if label=="before_successor_both_outside_actual_submitters" and clipped_r>clipped_l:
                        own_lane=Counter()
                        partition(exclusive[q.tid],clipped_l,clipped_r,own_lane)
                        if "no CPU scope on worker" in own_lane:
                            own_lane["no CPU scope on successor producer"]+=own_lane.pop("no CPU scope on worker")
                        own_lane["outside_recorded_cpu_lane"]+=clipped_r-clipped_l-sum(own_lane.values())
                        cpu_labels.update(own_lane)
                        own_by_role[g["role"]].update(own_lane)
                        for k,v in own_lane.items():pairs[(g["op"],k)]+=v
                        succ_ops[g["op"]]+=clipped_r-clipped_l
            if sum(v for k,v in this.items() if not k.startswith(("subset_","group_")))!=r-l:raise ValueError(g)
            nom.update(this)
            row.update({k+"_ns":v for k,v in this.items()})
            table.append(row)
        agg={k:v.result() for k,v in curves.items()}
        if any(abs(agg[k]["nominal_ms"]*1e6-v)>.001 for k,v in nom.items()):raise ValueError(nom)
        if round(agg["acl_returned"]["nominal_ms"],6)!=clock["after_return_ms"]["nominal"]:raise ValueError("does not reproduce clock bound")
        def ranked(counter):
            return {str(k):round(v/1e6,6) for k,v in counter.most_common() if v}
        first=next(g for g in gaps if int(g["step"])==step)
        report=dict(shape=[int(first["batch"]),int(first["seq_len"])],offset_bound_ns=[lo,hi],
                    gaps=sum(1 for g in gaps if int(g["step"])==step),
                    total_gap_ms=sum(v for k,v in nom.items() if not k.startswith(("subset_","group_")))/1e6,classes=agg,
                    before_next_by_cpu_scope_nominal_ms=ranked(cpu_labels),before_next_by_successor_nominal_ms=ranked(succ_ops),
                    before_next_by_role_nominal_ms={k:ranked(v) for k,v in own_by_role.items()},
                    before_next_by_cpu_scope_and_successor_nominal_ms=ranked(pairs))
        output["steps"][step]=report
    fields=dict.fromkeys(k for row in table for k in row)
    with gzip.open(HERE/"physical-gap-coverage.csv.gz","wt",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    (HERE/"coverage.json").write_text(json.dumps(output,indent=2)+"\n")


if __name__=="__main__":run()
