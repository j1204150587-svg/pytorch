"""Pair actual ATen invocation ordinals and their physical per-stream task slots."""
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path
from analyze_dynamic_trace import stats,write_csv


def read_calls(path,aten):
    with gzip.open(path/"queue_calls.csv.gz","rt",newline="") as handle:
        calls=[r for r in csv.DictReader(handle) if r["aten"]==aten]
    owned=defaultdict(list)
    with gzip.open(path/"work_tasks.csv.gz","rt",newline="") as handle:
        for r in csv.DictReader(handle):
            if r["queue_id"]:
                owned[r["step"],r["queue_id"]].append((r["work_sequence_sha256"],int(r["stream_work_slot"]),
                                                     r["task_type"],r["task"],int(r["stream"])))
    tids=defaultdict(set)
    keyed=defaultdict(list)
    for c in calls:
        tids[c["step"],c["role"]].add(c["producer_tid"])
        key=int(c["step"]),c["role"],c["aten_ordinal_in_thread"]
        keyed[key].append((c,sorted(owned[c["step"],c["queue_id"]])))
    if any(len(v)>1 for v in tids.values()):
        raise ValueError("multiple producer threads in the same role; ordinal match needs a stricter thread key")
    return keyed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair",type=Path,help="output of compare_dynamic_legs.py, including A/ and B/")
    parser.add_argument("--aten",required=True)
    parser.add_argument("--op",help="select only this exact queue name in both legs")
    parser.add_argument("--changed-op-only",action="store_true",help="select pairs whose PTA queue names changed")
    parser.add_argument("--single-sdma",action="store_true")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    a,b=(read_calls(args.pair/leg,args.aten) for leg in ("A","B"))
    statuses=defaultdict(Counter)
    rows=[]
    grouped=defaultdict(lambda:defaultdict(list))
    compare=json.loads((args.pair/"submissions_compare.json").read_bytes())
    ambiguous={r["step"]:r["ambiguous_duplicate_work_streams"] for r in compare["steps"]}
    for key in sorted(a.keys()|b.keys()):
        step,role,ordinal=key
        status=statuses[step,role]
        aa,bb=a.get(key,[]),b.get(key,[])
        status["all_ordinals"]+=1
        if not ordinal or len(aa)!=1 or len(bb)!=1:
            status["missing_or_ambiguous_queue"]+=1
            continue
        (x,xt),(y,yt)=aa[0],bb[0]
        if args.op and (x["op"]!=args.op or y["op"]!=args.op):
            status["other_queue_filtered"]+=1
            continue
        if args.changed_op_only and x["op"]==y["op"]:
            status["unchanged_queue_filtered"]+=1
            continue
        if args.single_sdma and not (len(xt)==len(yt)==1 and xt[0][2].startswith("SDMA") and yt[0][2].startswith("SDMA")):
            status["non_single_sdma_or_unlinked_filtered"]+=1
            continue
        status["selected"]+=1
        api_equal=x["acl_api_sequence"]==y["acl_api_sequence"]
        linked=(len(xt)==int(x["launch_api_count"])>0 and len(yt)==int(y["launch_api_count"])>0)
        slot_equal=[v[:4] for v in xt]==[v[:4] for v in yt] and linked and not ambiguous[step]
        physical_x=sorted((stream,slot,kind,name) for _,slot,kind,name,stream in xt)
        physical_y=sorted((stream,slot,kind,name) for _,slot,kind,name,stream in yt)
        physical_slots=linked and [t[:2] for t in physical_x]==[t[:2] for t in physical_y]
        physical_names=linked and physical_x==physical_y
        status["same_API_sequence"]+=int(api_equal)
        status["same_linked_work_stream_slots"]+=int(slot_equal)
        status["same_physical_stream_ID_and_slot"]+=int(physical_slots)
        status["same_physical_stream_slot_and_kernel_name"]+=int(physical_names)
        for leg,c,tasks in (("A",x,xt),("B",y,yt)):
            row=dict(step=step,role=role,ordinal=ordinal,leg=leg,batch=c["batch"],seq_len=c["seq_len"],
                     op=c["op"],queue_id=c["queue_id"],api_count=c["launch_api_count"],
                     linked_tasks=len(tasks),same_stream_slots=int(slot_equal),same_api_sequence=int(api_equal),
                     same_physical_stream_ID_and_slot=int(physical_slots),
                     same_physical_stream_slot_and_kernel_name=int(physical_names))
            for label in ("aten_to_enqueue_ns","enqueue_ns","signed_queue_wait_ns","dequeue_ns","workspace_ns","acl_api_sum_ns"):
                row[label]=c.get(label,"")
                if row[label]!="":
                    grouped[step,role,leg][label].append(int(row[label]))
            rows.append(row)
    args.out.mkdir(parents=True,exist_ok=True)
    write_csv(args.out/"matched_calls.csv.gz",rows,compressed=True)
    write_csv(args.out/"counts.csv",[dict(step=s,role=r,**c) for (s,r),c in sorted(statuses.items())])
    write_csv(args.out/"latency.csv",[dict(step=s,role=r,leg=l,metric=k,**stats(v))
                        for (s,r,l),d in sorted(grouped.items()) for k,v in d.items()])


if __name__=="__main__":
    main()
