"""Test BF16 representation and two-input rounding independently of atomic reduction order."""
import json
import torch
from probe_common import output, run, schedule

report=dict(shapes=schedule,checks=[],passed=False)
streams=[torch.npu.current_stream(),torch.npu.Stream()]
shape=[2,22]
assert shape in schedule
batch,length=shape
count=batch*length*4
owners=torch.arange(count,dtype=torch.int64).remainder(64)


def check(label,bank,host_ids,offset,mode="controlled"):
    s=streams[offset%2]
    with torch.npu.stream(s):
        host=torch.zeros((batch,length,1024,4),dtype=torch.int64)
        table=(17*offset+15)%1024
        host[:,:,table,:]=host_ids.reshape(batch,length,4)
        ids=host.npu()[:,:,table,:]
        grad=bank.reshape(batch,length,4,64).npu()
        vals=[run(leg,grad,ids,-1,s)[0] for leg in ("A","B","B","A")]
    counts=[int((vals[i]!=vals[j]).sum()) for i,j in [(0,3),(1,2),(0,1),(3,2)]]
    rows=(vals[0]!=vals[1]).nonzero()[:5].tolist()
    result=dict(label=label,mode=mode,diff_AA_BB_AB_AB=counts,
                examples=[dict(pos=p,ABBA=[int(v[tuple(p)]) for v in vals]) for p in rows])
    report["checks"].append(result)
    output.write_text(json.dumps(report,indent=2))
    return not any(counts)


# Each output channel/row receives exactly one nonzero source. Enumerate every BF16
# encoding including signed zero, subnormal values, NaNs, infinities and both signs.
for i in range(16):
    data=torch.zeros((count,64),dtype=torch.bfloat16)
    bits=torch.arange(4096,dtype=torch.int32)+4096*i
    data[:64]=bits.to(torch.int16).view(torch.bfloat16).reshape(64,64)
    check("all_encodings_"+str(i),data,owners,i)

# Exactly two contributions per output. This includes halfway cases on either side
# of an even mantissa, cancellation, underflow, overflow and mixed infinities.
examples=[(1.,2.**-8),(1.+2.**-7,2.**-8),(-1.,-2.**-8),(-1.-2.**-7,-2.**-8),
          (1.,-1.),(2.**-126,-2.**-133),(2.**-133,2.**-133),
          (3.38953139e38,3.38953139e38),(float("inf"),float("-inf")),(float("inf"),2.),
          (float("-inf"),-3.),(float("nan"),1.),(0.,-0.)]
data=torch.zeros((count,64),dtype=torch.bfloat16)
for j in range(64):
    a,b=examples[j%len(examples)]
    data[:64,j]=a
    data[64:128,j]=b
check("two_addends",data,owners,16)

# Stable gradients also exercise every padding branch with collisions and signed
# values. Powers-of-two / small integers sum exactly in FP32.
for i,pow2 in enumerate((-8,-4,0,4,8)):
    ids=torch.arange(count).remainder(3)
    values=(torch.arange(count*64).remainder(13)-6).float().mul(2.**pow2)
    check("exact_cancellation_2pow"+str(pow2),values.reshape(count,64).bfloat16(),ids,20+i)
report["passed"]=all(not any(x["diff_AA_BB_AB_AB"]) for x in report["checks"])
output.write_text(json.dumps(report,indent=2))
