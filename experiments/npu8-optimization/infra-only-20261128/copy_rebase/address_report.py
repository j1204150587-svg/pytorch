"""Count exact self-copies and adjacency of real (anonymized) address intervals."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
from itertools import groupby
import json
from pathlib import Path


def adjacent(rows):
    counts=Counter()
    for p,q in zip(rows,rows[1:]):
        sf=p["src_end"]==q["src_start"]
        df=p["dst_end"]==q["dst_start"]
        sr=q["src_end"]==p["src_start"]
        dr=q["dst_end"]==p["dst_start"]
        for key,val in (("src_forward",sf),("dst_forward",df),("src_reverse",sr),("dst_reverse",dr),
                        ("both_forward",sf and df),("both_reverse",sr and dr),
                        ("both_adjacent_different_direction",sf and dr or sr and df)):
            counts[key]+=int(val)
    counts["compared_pairs"]=max(0,len(rows)-1)
    return counts


def main():
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("tsv",type=Path)
    cli.add_argument("out",type=Path)
    args=cli.parse_args()
    with args.tsv.open(newline="") as src:
        data=[{k:int(v) for k,v in r.items()} for r in csv.DictReader(src,delimiter="\t")]
    assert [r["ordinal"] for r in data]==list(range(len(data)))
    identical=[r for r in data if r["dst_start"]==r["src_start"] and r["dst_end"]==r["src_end"]
               and r["same_metadata"] and r["dst_bytes"]==r["src_bytes"] and r["dst_bytes"]>0 and not r["same_impl"]]
    streams=defaultdict(list)
    for r in data:
        streams[r["physical_stream"]].append(r)
    result={"count":len(data),"different_impl_identical_range_and_metadata":len(identical),
            "identical_by_stream_and_bytes":dict(Counter(str((x["physical_stream"],x["dst_bytes"],x["same_storage"])) for x in identical)),
            "stream_address_census":{}}
    for stream,records in sorted(streams.items()):
        # These native rows describe the contiguous copy eligibility. A new run starts when a different copy
        # is seen on this stream. Other native operators are outside this address-only diagnostic.
        def selected(r):
            return r["same_metadata"] and r["contiguous"] and r["same_dtype"] and r["dst_format"]==2 and r["src_format"]==2 and r["dst_bytes"]>0
        census={}
        census["total_copy_calls"]=len(records)
        census["pta_streams"]=dict(Counter(r["pta_stream"] for r in records))
        census["contiguous_groups"]=[len(list(g)) for yes,g in groupby(records,key=selected) if yes]
        census["in_contiguous_groups"]={}
        counts=Counter()
        lengths=[]
        for yes,group in groupby(records,key=selected):
            if yes:
                block=list(group)
                lengths.append(len(block))
                counts.update(adjacent(block))
        census["in_contiguous_groups"]=dict(counts)
        census["length_distribution"]=dict(Counter(r["dst_bytes"] for r in records if selected(r)))
        census["first_copy_bytes"]=[r["dst_bytes"] for r in records[:38]]
        census["last_copy_bytes"]=[r["dst_bytes"] for r in records[-38:]]
        starts=[i for i,row in enumerate(records) if row in identical]
        if starts:
            boundaries=starts+[len(records)]
            blocks=[records[a:b] for a,b in zip(boundaries,boundaries[1:])]
            census["steps_from_identical_range"]=[
                dict(step=i,first_ordinal=group[0]["ordinal"],self_copy_bytes=group[0]["dst_bytes"],
                     other_calls=len(group)-1,eligible_other_calls=sum(selected(r) for r in group[1:]),
                     other_adjacency=dict(adjacent(group[1:]))) for i,group in enumerate(blocks,1)]
        result["stream_address_census"][str(stream)]=census
    args.out.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
