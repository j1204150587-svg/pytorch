"""Compare all physical stream sequences and matched IPG/HCCL intervals per shape."""
import argparse
import gzip
import json
import hashlib
import sys
from collections import Counter,defaultdict
from decimal import Decimal
from pathlib import Path
from difflib import SequenceMatcher

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"hccl_rank1"))
from trace_scalar import ns,overlap,union


def summarize(leg,kind):
    path=leg/"trace_view.json.gz"
    with gzip.open(path,"rt",encoding="utf-8-sig") as f:
        raw=json.load(f)
    ev=raw if isinstance(raw,list) else raw["traceEvents"]
    pid={e["args"]["name"]:e["pid"] for e in ev if e["name"]=="process_name"}
    events=[]
    for e in ev:
        if e["ph"]=="X":
            s=ns(e["ts"])
            events.append((s,s+ns(e["dur"]),e))
    hw_pid=pid["Ascend Hardware"]
    hard=[(s,t,e) for s,t,e in events if e["pid"]==hw_pid and "Task Id" in e["args"]]
    free=union((s,t) for s,t,e in events if e["pid"]==pid["Overlap Analysis"] and e["name"]=="Free")
    marks=sorted((s,t,e) for s,t,e in events if e["name"].startswith("ProfilerStep#"))
    reports=[]
    for (s,t,scope),shape in zip(marks,[[1,19],[2,14]],strict=True):
        h=sorted(((u,v,e) for u,v,e in hard if s<=u<t),key=lambda v:(v[0],v[2]["name"]))
        streams=defaultdict(list)
        for u,v,e in h:
            a=e["args"]
            streams[str(a["Physic Stream Id"])].append((e["name"],a["Task Type"]))
        work=[(u,v,e) for u,v,e in h if e["args"]["Task Type"].startswith(("AI_","MIX_","SDMA","HDMA","RDMA"))]
        r=dict(step=scope["name"],shape=shape,hardware=len(h),work=len(work),
               stream_seq={k:dict(n=len(v),sha256=hashlib.sha256(json.dumps(v).encode()).hexdigest(),count=dict(Counter(str(i) for i in v)))
                           for k,v in streams.items()})
        if kind=="ipg":
            idx,=[i for i,(_,_,e) in enumerate(work) if str(e["args"]["Physic Stream Id"])=="44" and "TensorMove" in e["name"]]
            prev,tar,nxt=work[idx-1:idx+2]
            def anchor(e):
                return [e["name"],e["args"]["Physic Stream Id"],e["args"]["Task Id"]]
            def interval(start,end):
                cpu=[(max(u,start),min(v,end),x["name"]) for u,v,x in events
                     if u<end and v>start and x["tid"]==scope["tid"] and x.get("cat")=="cpu_op" and x["name"].startswith("aten::")]
                return dict(wall_ms=(end-start)/1e6,official_free_ms=overlap((start,end),free)/1e6,
                            cpu_op_count=dict(Counter(x for _,_,x in cpu)),main_aten_union_ms=sum(v-u for u,v in union((u,v) for u,v,_ in cpu))/1e6)
            r["range"]=dict(previous=anchor(prev[2]),copy=anchor(tar[2]),following=anchor(nxt[2]),
                            before=interval(prev[1],tar[0]),launch_duration_us=(tar[1]-tar[0])/1e3,after=interval(tar[1],nxt[0]),
                            total_empty_ms=(tar[0]-prev[1]+nxt[0]-tar[1])/1e6)
        else:
            qids={e["args"]["correlation_id"]:(u,v,e) for u,v,e in events if s<=u<t and e["name"]=="Enqueue@HcclAllreduce"}
            dq={e["args"]["correlation_id"]:(u,v,e) for u,v,e in events if e["name"]=="Dequeue@HcclAllreduce"}
            rows=[]
            for qid,(ea,eb,e) in sorted(qids.items(),key=lambda kv:kv[1][0]):
                da,db,dc=dq[qid]
                hs=[(u,v,x) for u,v,x in events if x["tid"]==dc["tid"] and da<=u and v<=db and x["name"]=="AscendCL@hcom_allReduce_"]
                cu,cv,ce,=next(x for x in events if x[2]["name"]=="c10d::allreduce_" and x[2]["tid"]==e["tid"] and x[0]<=ea and eb<=x[1])
                item=dict(qid=qid,shape=ce.get("args",{}).get("Input Dims"),dtype=ce.get("args",{}).get("Input type"),
                          enqueue_us=(eb-ea)/1e3,wait_us=(da-eb)/1e3,service_us=(db-da)/1e3,producer_us=(ea-cu)/1e3,
                          consumer_free_us=overlap((da,db),free)/1e3)
                if hs:
                    (ha,hb,_),=hs
                    la,lb,le=next((u,v,x) for u,v,x in events if x["tid"]==dc["tid"] and x["name"]=="Node@launch" and (u,v)==(ha,hb))
                    conn=le["args"]["connection_id"]
                    meta=next(x["args"] for u,v,x in events if x["pid"]==pid["Communication"] and x["name"].startswith("hcom_allReduce__")
                              and x["args"].get("connection_id")==conn)
                    phy=[e for u,v,e in work if e["args"].get("connection_id")==conn]
                    item.update(hccl=True,dtype=meta["data_type"],count=meta["count"],hccl_us=(hb-ha)/1e3,
                                hccl_free_us=overlap((ha,hb),free)/1e3,
                                tasks=[[x["name"],x["args"]["Physic Stream Id"],x["args"]["Task Id"]] for x in phy])
                else:
                    item.update(hccl=False, underlying_cann=[x["name"] for u,v,x in events if x["tid"]==dc["tid"] and da<=u and v<=db and x["name"].startswith("AscendCL@")])
                rows.append(item)
            r["collectives"]=rows
        reports.append((r,streams))
    return reports


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("kind",choices=["ipg","hccl"])
    cli.add_argument("root",type=Path)
    a=cli.parse_args()
    left,right=[summarize(a.root/l,a.kind) for l in ("01-A","02-B")]
    result=[]
    for (x,sx),(y,sy) in zip(left,right,strict=True):
        assert x["shape"]==y["shape"]
        diff={}
        for stream in sorted(set(sx)|set(sy)):
            aseq=sx[stream];bseq=sy[stream]
            if aseq==bseq:
                diff[stream]=dict(identical=True)
            else:
                ca=Counter(aseq);cb=Counter(bseq)
                diff[stream]=dict(identical=False,A_only={str(k):n for k,n in (ca-cb).items()},B_only={str(k):n for k,n in (cb-ca).items()})
        result.append(dict(A=x,B=y,sequence_diff=diff))
    (a.root/"physical.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
