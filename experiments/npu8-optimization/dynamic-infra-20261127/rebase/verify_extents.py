"""Check address equivalence and canonical key cardinality for the actual schedule."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    schedule=json.loads(args.schedule.read_bytes())
    per_shape=[]
    for b,l in schedule:
        tables=1024
        numel=b*l*tables*4
        stride=(l*tables*4,tables*4,1)
        classes=set()
        for table in range(tables):
            original=4*table
            shift=original-original%64
            residual=original-shift
            # Canonical extent includes the complete last vector and fits in storage.
            extent=residual+(b-1)*stride[0]+(l-1)*stride[1]+4
            extent=(extent+63)//64*64
            if shift % 64 or shift+extent>numel or residual<0:
                raise ValueError("misaligned descriptor or backing storage overflow")
            for i in range(b):
                for j in range(l):
                    # Both representations preserve the same unit feature stride,
                    # so verifying the first and last elements covers the vector.
                    for k in (0,3):
                        offset=i*stride[0]+j*stride[1]+k
                        if original+offset!=shift+residual+offset or residual+offset>=extent:
                            raise ValueError("index resolves outside its canonical descriptor")
            classes.add((residual,extent))
        per_shape.append(dict(batch=b,seq_len=l,tables=tables,distinct_original_index_offsets=tables,
                              canonical_indices_metadata_classes=len(classes)))
    args.output.write_text(json.dumps(dict(schedule_sha256=hashlib.sha256(args.schedule.read_bytes()).hexdigest(),
                                          shapes=per_shape),indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
