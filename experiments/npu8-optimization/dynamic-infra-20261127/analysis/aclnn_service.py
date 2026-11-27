"""Separate CANN tiling, launch and ACLNN/handler time for embedding backward."""
import argparse
from bisect import bisect_left
from collections import defaultdict
import gzip
import json
from pathlib import Path
from analyze_dynamic_trace import Span, stats, write_csv


def union(spans):
    total=0
    frontier=0
    for s in sorted(spans,key=lambda x:x.start):
        total+=max(0,s.end-max(s.start,frontier))
        frontier=max(frontier,s.end)
    return total


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leg",type=Path)
    parser.add_argument("out",type=Path)
    parser.add_argument("--op",default="aclnnEmbeddingDenseBackward")
    args=parser.parse_args()
    with gzip.open(args.leg/"trace_view.json.gz","rt",encoding="utf8") as f:
        events=json.load(f)
    if isinstance(events,dict):
        events=events["traceEvents"]
    ops,steps,child = [], {}, defaultdict(list)
    for e in events:
        if e["ph"]!="X":
            continue
        name=e["name"]
        if name.startswith("ProfilerStep#"):
            steps[int(name.split("#")[-1])]=Span.read(e)
        if name=="Dequeue@"+args.op:
            ops.append(Span.read(e))
        if name.startswith("AscendCL@") or name=="Node@launch":
            child[e["tid"]].append(Span.read(e))
    del events
    index={}
    for tid,rows in child.items():
        rows.sort(key=lambda s:s.start)
        index[tid]=[s.start for s in rows]
    out=defaultdict(lambda:defaultdict(list))
    for op in ops:
        step=next((n for n,scope in steps.items() if scope.start<=op.start<op.end<=scope.end),None)
        if step is None:
            continue
        starts=index[op.tid]
        nested=[s for s in child[op.tid][bisect_left(starts,op.start):bisect_left(starts,op.end)] if s.end<=op.end]
        aclnn=[s for s in nested if s.name=="AscendCL@"+args.op]
        if len(aclnn)!=1:
            continue
        api=aclnn[0]
        inner=[s for s in nested if api.start<=s.start< s.end<=api.end and s is not api]
        tiling=[s for s in inner if s.name.endswith("_Tiling")]
        nodes=[s for s in inner if s.name=="Node@launch"]
        launches=[s for s in inner if s.name.startswith("AscendCL@aclrtLaunchKernel")]
        queries=[s for s in inner if s.name.startswith("AscendCL@aclrtGet")]
        key=step,len(tiling),len(launches)
        total=api.end-api.start
        metrics=dict(handler_outside_aclnn=(op.end-op.start)-total,aclnn=total,
                     all_tiling=union(tiling),all_launch_nodes=union(nodes),all_launch_APIs=union(launches),
                     all_runtime_queries=union(queries),
                     aclnn_outside_tiling_launch_nodes_queries=total-union(tiling+nodes+queries))
        for label,value in metrics.items():
            out[key][label].append(value)
    write_csv(args.out,[dict(step=step,op=args.op,tiling_calls_per_dequeue=t,kernel_calls_per_dequeue=k,
                             metric=label,**stats(values))
                        for (step,t,k),m in sorted(out.items()) for label,values in m.items()])


if __name__=="__main__":
    main()
