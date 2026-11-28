"""Identical backward for both legs; replace only the forward on the eligible shapes."""
from collections import Counter
import math
import struct
import threading
import time

import torch


COUNTS=Counter()
BUILDS=[]
TABLES={}
LOCK=threading.Lock()
EPS=1e-6


def key(x,eps):
    return x.device.type,x.device.index,struct.pack("!d",eps)


def get_table(x,eps):
    k=key(x,eps)
    with LOCK:
        if k not in TABLES:
            begin=time.perf_counter_ns()
            # Host construction preserves the full 16-bit bit pattern on H2D.
            source=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to(x.device)
            # Native arithmetic in a row-scalar layout matching the original RMS.
            table=(source.reshape(1,65536,1)+eps).rsqrt().reshape(65536)
            event=stream=None
            if x.device.type=="npu":
                stream=torch.npu.current_stream(x.device)
                event=stream.record_event()
            TABLES[k]=(table,stream,event)
            BUILDS.append({"device":str(x.device),"eps":eps,"numel":table.numel(),
                           "retained_bytes":table.numel()*table.element_size(),
                           "build_host_us":(time.perf_counter_ns()-begin)/1000})
            COUNTS["table_created"]+=1
        table,creator,event=TABLES[k]
    if creator is not None:
        current=torch.npu.current_stream(x.device)
        if current != creator:
            # A fixed event describes table initialization alone; the table
            # remains alive and immutable for this process.
            current.wait_event(event)
            COUNTS["other_stream_wait"]+=1
    return table


def supported(x,w,eps):
    # The retained cache is bounded to one canonical HSTU epsilon per device.
    return (type(eps) in (float,int) and math.isfinite(eps) and eps==EPS and x.ndim==3
            and x.shape[-1]==64 and x.shape[0]>0 and x.shape[1]>0 and w.shape==(64,)
            and x.dtype==w.dtype==torch.bfloat16 and x.device==w.device
            and x.is_contiguous() and w.is_contiguous() and x.device.type in ("cpu","npu"))


def parts(x,w,table,debug=False):
    if x.device.type=="cpu":
        sq=x.pow(2)
        mean=sq.mean(dim=-1,keepdim=True)
        idx=mean.view(torch.int16).to(torch.int64).bitwise_and(65535)
        inv=table[idx]
        z=x*inv
        y=z*w
        return y,inv,z,(sq,mean,inv,z,y) if debug else ()
    from lut_kernel import lut_forward
    inv=x.new_empty((*x.shape[:-1],1))
    z,y=torch.empty_like(x),torch.empty_like(x)
    trace=x.new_empty((*x.shape[:-1],5,64)) if debug else x
    lut_forward[(x.numel()//64,)](x,w,table,inv,z,y,trace,debug)
    return y,inv,z,trace.unbind(-2) if debug else ()


class SharedRMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x_direct,x_variance,w,eps,use_lut):
        if use_lut and supported(x_direct,w,eps):
            y,inv,z,_=parts(x_direct,w,get_table(x_direct,eps))
            COUNTS["triton_forward" if x_direct.device.type=="npu" else "cpu_lookup"]+=1
        else:
            COUNTS["fallback" if use_lut else "stock_forward"]+=1
            y,inv,z=torch.ops.npu_hstu_rms_lut.forward_parts.default(x_direct,w,eps)
        ctx.save_for_backward(x_variance,w,inv,z)
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,dy):
        direct,var,dw=torch.ops.npu_hstu_rms_lut.backward_parts.default(*ctx.saved_tensors,dy)
        return direct,var,dw,None,None


def run(x,w,eps=EPS,fast=True):
    return SharedRMS.apply(x,x,w,eps,fast)
