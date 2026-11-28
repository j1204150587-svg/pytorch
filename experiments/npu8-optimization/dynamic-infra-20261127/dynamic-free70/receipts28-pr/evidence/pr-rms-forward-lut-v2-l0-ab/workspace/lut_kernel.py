import triton
import triton.language as tl


@triton.jit
def lut_forward(x, w, lut, inv_out, z_out, y_out, trace, DEBUG: tl.constexpr):
    row=tl.program_id(0)
    c=tl.arange(0,64)
    off=row*64+c
    a=tl.load(x+off).to(tl.float32)
    weight=tl.load(w+c).to(tl.float32)
    square=(a*a).to(tl.bfloat16)
    mean=(tl.sum(square.to(tl.float32),axis=0)*(1.0/64)).to(tl.bfloat16)
    # Sign bit is part of the BF16 table key. Negative and nonfinite bit
    # patterns map to the stock value computed on the same logical device.
    index=mean.to(tl.int16,bitcast=True).to(tl.int32) & 65535
    inv=tl.load(lut+index)
    z=(a*inv.to(tl.float32)).to(tl.bfloat16)
    y=(z.to(tl.float32)*weight).to(tl.bfloat16)
    tl.store(inv_out+row,inv)
    tl.store(z_out+off,z)
    tl.store(y_out+off,y)
    if DEBUG:
        base=trace+row*5*64+c
        tl.store(base,square)
        tl.store(base+64,mean)
        tl.store(base+128,inv)
        tl.store(base+192,z)
        tl.store(base+256,y)
