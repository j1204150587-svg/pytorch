"""Check forward-only 512-byte rebasing with separate weights and index buffers."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import torch
import torch_npu


def counters():
    candidates={line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                if "/libtorch_npu.so" in line}
    if len(candidates)!=1:
        raise ValueError(f"expected one loaded libtorch_npu.so, found {sorted(candidates)}")
    native=ctypes.CDLL(candidates.pop()).npu_dynamic_index_rebase_counter
    native.argtypes=(ctypes.c_uint,)
    native.restype=ctypes.c_uint64
    return {k:native(i) for i,k in enumerate(("eligible","fallback","key_uses","descriptor_uses"))}


def check(weight, device_weight, indices, device_indices):
    embedding=torch.ops.aten.embedding.default
    expected=embedding(weight,indices,-1,False,False)
    actual=embedding(device_weight,device_indices,-1,False,False).cpu()
    if not torch.equal(expected,actual):
        raise ValueError(f"lookup differs at input {tuple(indices.shape)}, offsets "
                         f"{weight.storage_offset()}/{indices.storage_offset()}")
    return hashlib.sha256(actual.float().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    torch.npu.set_device(0)
    tables=64
    before=counters()
    rows=[]
    # The two owners have different data at every location. Returning the wrong
    # owner's address on a cache hit is visible even when its metadata is identical.
    for generation,(batch,length) in enumerate(((1,19),(2,14),(2,22),(4,32),(1,19))):
        ids=[]; weights=[]
        for salt in (0,1):
            index_flat=torch.arange(batch*length*tables*4,dtype=torch.long)
            index_bank=((index_flat*3+index_flat//5+salt*7+generation)%16).reshape(batch,length,tables,4)
            ids.append((index_bank,index_bank.npu()))
            weight_flat=((torch.arange(tables*4096+512,dtype=torch.int32)*19+salt*17+generation)%251-125).to(torch.bfloat16)
            weights.append((weight_flat,weight_flat.npu()))
        for table in (0,1,15,16,31,32,48,63):
            for salt in (0,1):
                ic,idn=ids[salt]
                wc,wd=weights[1-salt]
                # Cross 512B boundaries while preserving nonzero residuals and
                # testing a separate backing storage with the same canonical key.
                offset=257+table*4096
                w=wc.narrow(0,offset,4096).view(64,64)
                wn=wd.narrow(0,offset,4096).view(64,64)
                i,inu=ic[:,:,table,:],idn[:,:,table,:]
                rows.append(dict(shape=[batch,length],table=table,salt=salt,index_offset=i.storage_offset(),
                                 weight_offset=w.storage_offset(),sha256=check(w,wn,i,inu)))
    after=counters()
    # Rank-two indices must take the ordinary forward descriptor.
    two=i[0]
    guard_sha=check(w,wn,two,inu[0])
    after_fallback=counters()
    # A forward-only build must not activate even when backward receives a
    # qualifying three-dimensional index view and an eligible [64,64] output.
    grad=torch.ones((*i.shape,64),dtype=torch.bfloat16)
    backward=torch.ops.aten.embedding_dense_backward.default
    expected=backward(grad,i,64,-1,False)
    actual=backward(grad.npu(),inu,64,-1,False).cpu()
    if not torch.equal(expected,actual):
        raise ValueError("backward gradient differs from CPU")
    final=counters()
    if after_fallback!=final:
        raise ValueError("forward counters advanced during backward")
    mode=os.environ.get("NPU_DYNAMIC_REBASE_FORWARD","0")
    if mode=="1":
        accepted=after["eligible"]-before["eligible"]
        if accepted!=len(rows) or after["key_uses"]-before["key_uses"]!=2*accepted:
            raise ValueError(f"native key/eligibility mismatch: {before} -> {after} for {len(rows)} calls")
        if after_fallback["eligible"]!=after["eligible"] or after_fallback["fallback"]!=after["fallback"]+1:
            raise ValueError("rank-two fallback count differs")
    elif final!=before:
        raise ValueError("candidate activated on the control leg")
    args.output.write_text(json.dumps(dict(mode=mode,before=before,after_target=after,after_fallback=after_fallback,
        final=final,checks=rows,rank_two_sha256=guard_sha),indent=2))


if __name__=="__main__":
    main()
