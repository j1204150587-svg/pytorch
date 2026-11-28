"""Exercise real Python/ATen lazy-init entries in fresh processes and the real fork hook."""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def forked(conn):
    import torch_npu
    errors = []
    for _ in range(2):
        try:
            torch_npu._C._npu_getDevice()
        except RuntimeError as err:
            assert "Cannot re-initialize NPU in forked subprocess" in str(err)
            errors.append(type(err).__name__)
        else:
            raise AssertionError("forked entry unexpectedly succeeded")
    conn.send(dict(errors=errors,bad_fork=torch_npu.npu._is_in_bad_fork(),initialized=torch_npu.npu.is_initialized()))
    conn.close()


def check(mode):
    import torch
    import torch_npu
    npu = torch_npu.npu
    native = torch_npu._C._npu_getDevice
    assert not npu.is_initialized(), "test needs a cold process"
    if mode == "retry":
        calls = []
        def once_fails():
            calls.append("call")
            if len(calls) == 1:
                raise RuntimeError("lazy-init-test-first-call")
        npu._lazy_call(once_fails)
        try:
            native()
        except Exception as err:
            chain=[]
            while err is not None:
                chain.append([type(err).__name__,str(err).splitlines()[0]])
                err=err.__cause__ or err.__context__
            assert any("lazy-init-test-first-call" in text for _,text in chain) and not npu.is_initialized()
        else:
            raise AssertionError("first initialization did not propagate the exception")
        assert native() == 0 and npu.is_initialized() and native() == 0 and len(calls) == 2
        return dict(callbacks=len(calls),recovered=True,failure_chain=chain)
    if mode == "two_threads":
        entered,release = threading.Event(),threading.Event()
        calls,completed,errors = [],[],[]
        def slow_start():
            calls.append(threading.get_ident())
            entered.set()
            assert release.wait(15)
        npu._lazy_call(slow_start)
        def get_device():
            try:
                completed.append((native(),npu.is_initialized()))
            except BaseException as exc:
                errors.append(repr(exc))
        one=threading.Thread(target=get_device)
        two=threading.Thread(target=get_device)
        one.start()
        assert entered.wait(30)
        two.start()
        time.sleep(.04)
        pending = not completed
        release.set()
        one.join(30)
        two.join(30)
        assert not one.is_alive() and not two.is_alive() and not errors and pending
        assert sorted(completed)==[(0,True),(0,True)] and len(calls)==1
        return dict(two_threads=completed,callbacks=len(calls),both_waited=pending)
    if mode == "reentrant":
        calls=[]
        def from_initializer():
            calls.append([native(),npu.is_initialized()])
        npu._lazy_call(from_initializer)
        assert native()==0 and npu.is_initialized() and native()==0 and calls==[[0,False]]
        return dict(inside=calls,initialized=npu.is_initialized())
    if mode == "fork":
        assert native()==0 and npu.is_initialized()
        ctx=multiprocessing.get_context("fork")
        parent,child=ctx.Pipe(duplex=False)
        process=ctx.Process(target=forked,args=(child,))
        process.start()
        child.close()
        assert parent.poll(30), "no fork response"
        state=parent.recv()
        process.join(20)
        assert process.exitcode==0 and state==dict(errors=["RuntimeError"]*2,bad_fork=True,initialized=False)
        assert native()==0 and torch.ones(1,device="npu").cpu().item()==1
        return dict(child=state,parent=0)
    if mode == "threads_after_commit":
        assert native()==0
        results=[]
        def allocate():
            for i in range(64):
                x=torch.full((i%5+1,),float(i),device="npu")
                assert x.cpu().tolist()==[float(i)]*x.numel()
            results.append(64)
        workers=[threading.Thread(target=allocate) for _ in range(2)]
        for w in workers: w.start()
        for w in workers: w.join(30)
        assert results==[64,64]
        return dict(completed=results)
    raise ValueError(mode)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("switch",choices=("0","1"))
    parser.add_argument("--mode")
    args=parser.parse_args()
    os.environ.update(TORCH_NPU_LAZY_GIL_FAST=args.switch,TORCH_NPU_LAZY_GIL_AUDIT="1")
    if args.mode:
        result=check(args.mode)
    else:
        result={}
        for mode in ("retry","two_threads","reentrant","fork","threads_after_commit"):
            output=args.output.with_name(mode+".json")
            subprocess.run([sys.executable,"-B",__file__,str(output),args.switch,"--mode",mode],check=True,timeout=85)
            result[mode]=json.loads(output.read_bytes())
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
