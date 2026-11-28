"""Compare the shipped and private CANN Host entries on identical tensors on a reserved chip."""
import ctypes as c
import hashlib
import json
from pathlib import Path
import sys
import torch
import torch_npu


here = Path(__file__).resolve().parent
schedule = json.loads(Path(sys.argv[1]).read_bytes())
output = Path(sys.argv[2])
assert len(set(map(tuple,schedule))) == len(schedule) == 60
torch.npu.set_device(0)
lib = here/"private-host.so"
receipt = json.loads((here/"build-receipt.json").read_bytes())
assert hashlib.sha256(lib.read_bytes()).hexdigest()==receipt["files"][str(lib)]
stock = c.CDLL("libopapi.so")
private = c.CDLL(str(lib))
P, U, I, D = c.c_void_p, c.c_uint64, c.c_int64, c.POINTER(c.c_int64)
meta = c.CDLL("libnnopbase.so")
meta.aclCreateTensor.argtypes = [D,U,c.c_int,D,I,c.c_int,D,U,P]
meta.aclCreateTensor.restype = P
meta.aclDestroyTensor.argtypes = [P]


def entry(library, name):
    work = getattr(library,name+"GetWorkspaceSize")
    work.argtypes = [P,P,U,U,c.c_bool,P,c.POINTER(U),c.POINTER(P)]
    launch = getattr(library,name)
    launch.argtypes = [P,U,P,P]
    return work, launch


apis={"A":entry(stock,"aclnnEmbeddingDenseBackward"),"B":entry(private,"aclnnStudyBf16Embedding")}


def descriptor(t):
    n = t.dim()
    shape = (I*n)(*t.shape)
    stride = (I*n)(*t.stride())
    storage = t.untyped_storage()
    dims = (I*1)(storage.nbytes()//t.element_size())
    raw = meta.aclCreateTensor(shape,n,{torch.int64:9,torch.bfloat16:27}[t.dtype],stride,
                               t.storage_offset(),2,dims,1,storage.data_ptr())
    assert raw
    return raw


def run(leg, grad, ids, pad, stream):
    work, launch = apis[leg]
    out = torch.full((64,64),-73,dtype=torch.bfloat16,device=grad.device)
    torch.npu.synchronize()
    descriptors = list(map(descriptor,(grad,ids,out)))
    size,executor=U(),P()
    rc=work(*descriptors[:2],64,pad,False,descriptors[2],c.byref(size),c.byref(executor))
    assert rc==0 and executor.value,("prepare",leg,rc)
    buf=torch.empty(size.value,dtype=torch.uint8,device=grad.device) if size.value else None
    rc=launch(buf.data_ptr() if buf is not None else None,size,executor,stream.npu_stream)
    assert rc==0,("execute",leg,rc)
    torch.npu.synchronize()
    for item in descriptors:
        assert meta.aclDestroyTensor(item)==0
    return out.cpu().view(torch.int16),size.value


def checkpoint():
    output.write_text(json.dumps(report,indent=2))


report=dict(soc=torch.npu.get_device_name(0),rows=[],passed=False)
streams=[torch.npu.current_stream(),torch.npu.Stream()]
gen=torch.Generator().manual_seed(1234)
extra=[("cancellation",[2,22]),("bf16_extremes",[1,19]),("nonfinite",[4,21])]
cohorts=[("exact",s) for s in schedule]+[("random",s) for s in schedule]+extra
for k,(mode,(batch,length)) in enumerate(cohorts):
    pad=7 if k%2 else -1
    host=torch.randint(0,32,(batch,length,1024,4),generator=gen)
    g=torch.randint(-5,6,(batch,length,4,64),generator=gen).bfloat16()
    if mode=="random":
        g=torch.randn(g.shape,generator=gen).bfloat16()
    elif mode in ("cancellation","bf16_extremes","nonfinite"):
        host.remainder_(3)
        if mode=="cancellation":
            pat=[8388608.,1.,-8388608.,1.,-1.,0.00390625,-0.00390625]
        elif mode=="bf16_extremes":
            pat=[0.,-0.,1.1754944e-38,9.18355e-41,-1.1754944e-38,3.3895313e38,-3.3895313e38]
        else:
            pat=[float("inf"),float("-inf"),float("nan"),1.,-1.,0.]
        pattern=torch.tensor(pat).bfloat16()
        g=pattern.repeat((g.numel()+len(pat)-1)//len(pat))[:g.numel()].reshape_as(g)
    stream=streams[k%2]
    with torch.npu.stream(stream):
        grad=g.npu()
        ids=host.npu()[:,:,(17*k+15)%1024,:]
        vals=[]
        sizes=[]
        for leg in ("A","B","B","A"):
            bits,size=run(leg,grad,ids,pad,stream)
            vals.append(bits)
            sizes.append(size)
        counts=[int((vals[i]!=vals[j]).sum()) for i,j in [(0,3),(1,2),(0,1),(3,2)]]
        row=dict(mode=mode,shape=[batch,length],padding=pad,workspace_ABBA=sizes,
                 different_elements_AA_BB_AB_AB=counts)
        report["rows"].append(row)
        if any(counts):
            mismatch=(vals[0]!=vals[1]).nonzero()[:8].tolist()
            row["examples"]=[dict(pos=p,ABBA=[int(v[tuple(p)]) for v in vals]) for p in mismatch]
            checkpoint()
            print("Numerical gate stopped on observed bit differences",row,flush=True)
            break
    if k%10==0:
        checkpoint()
else:
    report["passed"]=True
checkpoint()
