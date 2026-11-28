#pragma once
#include <ATen/Tensor.h>

namespace projection_rebase {
struct View {
    const c10::TensorImpl* owner;
    void* base;
    int64_t elements;
};

class Scope {
public:
    Scope(const at::Tensor& input, const at::Tensor& weight);
    ~Scope();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
private:
    const View* previous_;
    View view_{};
};
}
