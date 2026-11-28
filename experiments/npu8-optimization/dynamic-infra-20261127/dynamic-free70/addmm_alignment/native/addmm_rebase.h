#pragma once

#include <ATen/Tensor.h>
#include <cstdint>

namespace projection_rebase {
struct View {
    const c10::TensorImpl *owner = nullptr;
    void *base = nullptr;
    int64_t elements = 0;
};

class Scope {
public:
    Scope(const at::Tensor &bias, const at::Tensor &input, const at::Tensor &weight);
    ~Scope();
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
private:
    const View *previous_;
    View views_[2];
};
}  // namespace projection_rebase
