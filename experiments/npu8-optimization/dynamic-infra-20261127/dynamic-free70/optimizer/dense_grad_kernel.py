import triton
import triton.language as tl


@triton.jit
def side_grad(ids_ptr, grad_ptr, out_ptr, tokens, tables: tl.constexpr, tile: tl.constexpr):
    table = tl.program_id(0)
    row = tl.arange(0,64)
    col = tl.arange(0,64)
    local = tl.arange(0,tile)
    acc = tl.full((64,64),0,tl.float32)
    for block in range(0,tl.cdiv(tokens,tile)):
        q = block*tile + local
        offset = (q*tables + table)*4
        count = tl.full((64,tile),0,tl.int32)
        for feature in tl.static_range(4):
            local_id = tl.load(ids_ptr+offset+feature,mask=q<tokens,other=-1).to(tl.int32)
            count += (row[:,None] == local_id[None,:]).to(tl.int32)
        grad = tl.load(grad_ptr + q[:,None]*64 + col[None,:],mask=q[:,None]<tokens,other=0)
        grad = (grad.to(tl.float32)*0.25).to(tl.bfloat16)
        grad_f32 = grad.to(tl.float32)
        nonfinite = (grad_f32 != grad_f32) | (tl.abs(grad_f32) == float("inf"))
        if tl.sum(tl.sum(nonfinite.to(tl.int32),0),0) != 0:
            # In an IEEE matrix product 0 * Inf and 0 * NaN would poison
            # rows which do not occur in this table. Add only selected rows.
            for token_index in range(0,tile):
                token = block*tile + token_index
                contribution = tl.load(grad_ptr+token*64+col,mask=token<tokens,other=0)
                contribution = (contribution.to(tl.float32)*0.25).to(tl.bfloat16).to(tl.float32)
                for feature in tl.static_range(4):
                    target = tl.load(ids_ptr+(token*tables+table)*4+feature,mask=token<tokens,other=-1)
                    acc += tl.where(row[:,None] == target,contribution[None,:],0.0)
        else:
            acc = tl.dot(count.to(tl.bfloat16),grad,acc)
    tl.store(out_ptr+table*4096+row[:,None]*64+col[None,:],acc.to(tl.bfloat16))
