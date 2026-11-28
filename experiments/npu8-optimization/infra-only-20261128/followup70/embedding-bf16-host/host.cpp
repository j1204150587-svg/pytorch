// Assemble a CANN Host executor using the installed DAV_2201 BF16 small-dimension kernel.
#include "acl/acl_rt.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

namespace l0op {
const aclTensor* EmbeddingDenseGradV2(const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
                                      uint64_t, uint64_t, bool, aclOpExecutor*);
}

using namespace op;

extern "C" aclnnStatus aclnnStudyBf16EmbeddingGetWorkspaceSize(
    const aclTensor* grad, const aclTensor* indices, uint64_t weights, uint64_t pad, bool scale,
    const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    if (!grad || !indices || !out || !workspaceSize || !executor) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    int64_t deterministic = -1;
    const auto& g = grad->GetViewShape();
    const auto& i = indices->GetViewShape();
    const auto& o = out->GetViewShape();
    if (GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_2201 ||
        aclrtGetSysParamOpt(ACL_OPT_DETERMINISTIC, &deterministic) != ACL_SUCCESS || deterministic != 0 ||
        weights != 64 || scale || (pad != UINT64_MAX && pad >= weights) ||
        grad->GetDataType() != DataType::DT_BF16 || out->GetDataType() != DataType::DT_BF16 ||
        indices->GetDataType() != DataType::DT_INT64 || g.GetDimNum() != 4 || i.GetDimNum() != 3 ||
        o.GetDimNum() != 2 || i.GetDim(0)<1 || i.GetDim(0)>4 || i.GetDim(1)<12 || i.GetDim(1)>32 ||
        i.GetDim(2)!=4 || g.GetDim(0)!=i.GetDim(0) || g.GetDim(1)!=i.GetDim(1) ||
        g.GetDim(2)!=4 || g.GetDim(3)!=64 || o.GetDim(0)!=64 || o.GetDim(1)!=64) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    L2_DFX_PHASE_1(aclnnStudyBf16Embedding, DFX_IN(grad, indices, weights, pad, scale), DFX_OUT(out));
    auto unique = CREATE_EXECUTOR();
    if (!unique.get()) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto gCont = l0op::Contiguous(grad, unique.get());
    auto iCont = l0op::Contiguous(indices, unique.get());
    if (!gCont || !iCont) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto i32 = l0op::Cast(iCont, DataType::DT_INT32, unique.get());
    auto accum = unique->AllocTensor(o, DataType::DT_BF16);
    if (!i32 || !accum) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto result = l0op::EmbeddingDenseGradV2(gCont, i32, i32, accum, weights, pad, scale, unique.get());
    if (!result || !l0op::ViewCopy(result, out, unique.get())) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    *workspaceSize = unique->GetWorkspaceSize();
    unique.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnStudyBf16Embedding(void* ws, uint64_t size, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnStudyBf16Embedding);
    return CommonOpExecutorRun(ws,size,executor,stream);
}
