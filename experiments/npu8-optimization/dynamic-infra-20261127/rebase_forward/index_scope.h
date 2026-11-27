#pragma once

#include <ATen/Tensor.h>
#include <cstdint>

namespace dynamic_index_rebase {
struct View {
    const c10::TensorImpl *tensor = nullptr;
    void *base = nullptr;
    int64_t offset = 0;
    int64_t storage_elements = 0;
};

class Scope {
public:
    Scope(const at::Tensor &weight, const at::Tensor &indices);
    ~Scope();
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
private:
    View views_[2];
    const View *previous_;
};
}  // namespace dynamic_index_rebase
