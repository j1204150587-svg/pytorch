"""Identify how much of the idle time preceding an Addmm occurred inside its eager linear wrapper."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
from collections import defaultdict

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"runner_tools"))
from clock_bounded_gaps import Span, common_extrema, join


def main(args):
    b,_,_,steps,enq,_,owners,_=join(args.leg)
    low=b["lower"]["shift_ns"]-1
    high=b["upper"]["shift_ns"]+1
    with gzip.open(args.leg/"trace_view.json.gz","rt",encoding="utf8") as f:
        events=json.load(f)
    events=events if isinstance(events,list) else events["traceEvents"]
    parents=[Span.read(e) for e in events if e.get("ph")=="X" and e["name"]=="aten::linear"]
    children=[Span.read(e) for e in events if e.get("ph")=="X" and e["name"] in
              ("aten::reshape","aten::view","aten::t","aten::transpose","aten::as_strided")]
    segments=defaultdict(list)
    ownership={}
    with gzip.open(args.gaps,"rt",encoding="utf8") as f:
        for gap in csv.DictReader(f):
            if gap["state"]!="linked" or gap["op"]!="aclnnAddmm":
                continue
            step=int(gap["step"])
            cid=int(gap["queue_id"])
            add=owners[cid]
            lin,=[s for s in parents if s.tid==add.tid and s.start<=add.start and add.end<=s.end]
            g0,g1=int(gap["device_begin_ns"]),int(gap["device_end_ns"])
            def add_segment(name,a,b):
                if a<b:
                    segments[step,name].append((g0,g1,a,b))
            add_segment("before_linear",g0+low,lin.start)
            add_segment("linear_before_addmm",lin.start,add.start)
            nested=sorted((s.start,s.end,s.name) for s in children if s.tid==lin.tid and lin.start<=s.start<s.end<=add.start)
            union=[]
            for a,b,_ in nested:
                if union and a<=union[-1][1]:
                    union[-1]=(union[-1][0],max(union[-1][1],b))
                else:
                    union.append((a,b))
            for a,b in union:
                add_segment("views_inside_linear",a,b)
            ownership[cid]=dict(step=step,queue_id=cid,producer_tid=enq[cid].tid,
                               linear_pre_us=(add.start-lin.start)/1000,
                               nested_views=[dict(name=n,us=(b-a)/1000) for a,b,n in nested])
    ranges={}
    for (step,name),items in sorted(segments.items()):
        a,b=common_extrema(items,low,high)
        ranges.setdefault(step,{})[name]=[a/1e6,b/1e6]
    args.output.write_text(json.dumps(dict(ranges_ms=ranges,owners=list(ownership.values())),indent=2))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for f in ("leg","gaps","output"):
        cli.add_argument(f,type=Path)
    main(cli.parse_args())
