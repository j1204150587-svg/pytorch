"""Recover guard conditions and per-PTA-stream censuses from the native audit worker."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def read_tsv(path):
    with path.open(newline="") as src:
        return list(csv.DictReader(src, delimiter="\t"))


def metadata(value):
    dtype, size, *details = value.split(":")
    parsed = {"dtype":dtype, "shape":tuple(int(s) for s in size.split(",") if s)}
    for d in details:
        if "=" in d:
            k,v=d.split("=",1)
            parsed[k]=int(v)
    return parsed


def analyze(folder):
    source, = folder.glob("copy-guard-*.tsv")
    events, = folder.glob("copy-events-*.tsv")
    samples = {}
    counts, traces, alignment = Counter(), defaultdict(Counter), defaultdict(Counter)
    classes = []
    for r in read_tsv(source):
        a,b=(metadata(r[k]) for k in ("dst","src"))
        if (a["dtype"] not in ("BFloat16","Float") or a["dtype"] != b["dtype"]):
            reason="dtype"
        elif a["shape"] != b["shape"]:
            reason="shape"
        elif int(r["same_storage"]):
            reason="shared_storage"
        elif not (a.get("ct") and b.get("ct")) or a.get("fmt")!=2 or b.get("fmt")!=2:
            reason="layout_or_format"
        elif any(t["base512"] for t in (a,b)):
            reason="backing_base_alignment"
        elif any(t["off512"] for t in (a,b)):
            reason="view_offset_alignment"
        elif a["len512"] or b["len512"]:
            reason="length_alignment"
        elif all(t["off0"] and t["store"]==t["bytes"] for t in (a,b)):
            reason="already_exact"
        else:
            reason="meets_recorded_guards"
        count=int(r["count"])
        counts[reason]+=count
        samples[r["class_id"]]=(reason,a,b,count)
        classes.append(dict(id=int(r["class_id"]), count=count, reason=reason, dst=a, src=b))
    actual=Counter()
    for i, row in enumerate(read_tsv(events)):
        assert i == int(row["ordinal"])
        key=row["class_id"]
        reason,a,b,_=samples[key]
        stream=row["pta_stream_id"]
        traces[stream][reason]+=1
        actual[key]+=1
        if reason in ("backing_base_alignment","view_offset_alignment","length_alignment"):
            label=tuple(t[k] for t in (a,b) for k in ("base512","off512","ptr512","len512"))
            alignment[stream][str(label)]+=1
    assert actual==Counter({k:r[3] for k,r in samples.items()})
    return {"total":sum(counts.values()),"reasons":counts,"by_pta_stream":traces,
            "alignment_by_pta_stream":alignment,"metadata_classes":sorted(classes,key=lambda z:-z["count"])}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("leg",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    a.output.write_text(json.dumps(analyze(a.leg),indent=2)+"\n")
