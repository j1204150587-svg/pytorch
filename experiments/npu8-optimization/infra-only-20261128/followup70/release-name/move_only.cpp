#include "torch_npu/csrc/framework/NPUDefine.h"
using Payload = at_npu::native::ExecuteParasOpApi;

// Match the target's out-of-line member call with the same two explicit ABI arguments.
__attribute__((noinline)) void moveOnly(Payload &dst, Payload &src) {
    dst.customHandler = std::move(src.customHandler);
}
