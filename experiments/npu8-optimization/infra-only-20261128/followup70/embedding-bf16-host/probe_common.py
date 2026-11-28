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


