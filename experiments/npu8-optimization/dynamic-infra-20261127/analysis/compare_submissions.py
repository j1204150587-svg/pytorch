"""Compare paired Level1 shapes, ATen operators, real ACL submissions and physical streams."""
import argparse
from collections import Counter
import json
from pathlib import Path


def streams(row, prefix):
    return sorted((s[prefix+"_n"],s[prefix+"_sha256"])
                  for s in row["streams"].values() if s[prefix+"_n"])


def lanes(row):
    return sorted((s["role"],s["n"],s["sha256"]) for s in row["aten_thread_sequences"])


def changes(a,b):
    aa,bb = Counter(a),Counter(b)
    return {k:bb[k]-aa[k] for k in sorted(set(aa)|set(bb)) if aa[k]!=bb[k]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a",type=Path,help="A leg submissions.json")
    parser.add_argument("b",type=Path,help="B leg submissions.json")
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    a,b=(json.loads(p.read_bytes()) for p in (args.a,args.b))
    rows = []
    left,right=({row["step"]:row for row in report["steps"]} for report in (a,b))
    if left.keys()!=right.keys():
        raise ValueError("different active steps")
    for number in sorted(left):
        aa,bb=left[number],right[number]
        if (aa["batch"],aa["seq_len"])!=(bb["batch"],bb["seq_len"]):
            raise ValueError("different active shapes")
        rows.append(dict(step=number,batch=aa["batch"],seq_len=aa["seq_len"],
                         aten_count_delta=changes(aa["aten_counts"],bb["aten_counts"]),
                         aten_per_thread_sequences_equal=lanes(aa)==lanes(bb),
                         enqueue_count_delta=changes(aa["enqueue_counts"],bb["enqueue_counts"]),
                         acl_submit_delta=changes(aa["acl_submission_apis"],bb["acl_submission_apis"]),
                         hardware_delta=changes(aa["hardware_types"],bb["hardware_types"]),
                         per_stream_work_count_and_sequence_equal=streams(aa,"work")==streams(bb,"work"),
                         per_stream_all_count_and_sequence_equal=streams(aa,"all")==streams(bb,"all"),
                         ambiguous_duplicate_work_streams=any(any(n>1 for n in Counter(streams(leg,"work")).values())
                                                              for leg in (aa,bb))))
    args.output.write_text(json.dumps(dict(a_sha256=a["trace_sha256"],b_sha256=b["trace_sha256"],steps=rows),
                                      indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
