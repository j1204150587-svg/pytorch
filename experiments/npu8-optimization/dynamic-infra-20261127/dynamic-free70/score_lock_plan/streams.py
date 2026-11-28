"""Concurrent raw launches with writer-stream allocation, immediate release and allocation reuse."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import traceback

import torch
import torch_npu


def run(a,report):
    sys.path.extend([str(a.fold_dir),str(a.scores_dir)])
    import locked
    import operator_impl as impl
    torch.npu.set_device(0)
    fw_table,bw_table=impl.tables(torch.device("npu:0"))
    ready=torch.npu.current_stream().record_event()
    shape=(2,4,22,22)
    streams=[torch.npu.Stream() for _ in range(2)]
    inputs=[torch.randn(shape,generator=torch.Generator().manual_seed(81+i)).bfloat16() for i in range(2)]
    producer=torch.npu.current_stream()
    refs=[]
    for host_x in inputs:
        x=host_x.npu()
        gf=torch.empty_like(x)
        gb=torch.empty_like(x)
        locked.launch(impl.causal_score_forward,(x,fw_table,gf),False)
        locked.launch(impl.causal_score_backward,(x,x,bw_table,gb),False)
        # Also prepare both PR plan objects before concurrent execution.
        locked.launch(impl.causal_score_forward,(x,fw_table,torch.empty_like(x)),True)
        locked.launch(impl.causal_score_backward,(x,x,bw_table,torch.empty_like(x)),True)
        refs.append((gf.cpu(),gb.cpu()))
    barrier=threading.Barrier(2)
    def work(i):
        torch.npu.set_device(0)
        allocated=torch.npu.Stream()
        run_stream=streams[i]
        observed=[]
        with torch.npu.stream(run_stream):
            run_stream.wait_event(ready)
        barrier.wait(timeout=30)
        for _ in range(32):
            with torch.npu.stream(allocated):
                x=inputs[i].npu()
                input_ready=allocated.record_event()
            with torch.npu.stream(run_stream):
                run_stream.wait_event(input_ready)
                y=torch.empty_like(x)
                z=torch.empty_like(x)
                locked.launch(impl.causal_score_forward,(x,fw_table,y),True)
                locked.launch(impl.causal_score_backward,(x,x,bw_table,z),True)
                observed.append((y,z))
            # Reuse pressure on the input's allocation stream while the consumer
            # stream still has work enqueued. It must keep x's storage valid.
            del x
            with torch.npu.stream(allocated):
                torch.empty_like(y).fill_(i+17.)
        run_stream.synchronize()
        return {"thread":i,"stream":int(run_stream.npu_stream),"pairs":len(observed),
            "different_words":sum(int((t.cpu().contiguous().view(torch.int16)!=ref.contiguous().view(torch.int16)).sum())
                                  for row in observed for t,ref in zip(row,refs[i],strict=True))}
    with ThreadPoolExecutor(max_workers=2) as pool:
        report["streams"]=list(pool.map(work,range(2)))
    torch.npu.synchronize()
    report["counts"]=locked.service_counts()
    report["accepted"]=all(r["different_words"]==0 for r in report["streams"]) and all(
        v==[66,66,66] for v in report["counts"].values()) and len(set(r["stream"] for r in report["streams"]))==2


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("fold-dir","scores-dir","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();report={"accepted":False}
    try:
        run(a,report)
    except Exception:
        report["error"]=traceback.format_exc()
        raise
    finally:
        a.output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report),flush=True)
