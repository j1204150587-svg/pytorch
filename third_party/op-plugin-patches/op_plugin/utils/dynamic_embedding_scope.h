#pragma once

#include <ATen/Tensor.h>
#include <cstdint>

namespace dynamic_embedding_rebase {

struct Counters;

struct View {
    const c10::TensorImpl *tensor = nullptr;
    void *base = nullptr;
    int64_t offset = 0;
    int64_t storage_elements = 0;
    uint64_t domain = 0;
    Counters *counters = nullptr;
};

class Scope {
public:
    explicit Scope(const at::Tensor &indices);
    ~Scope();
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    friend const View *for_tensor(const at::Tensor &);
private:
    const Scope *previous_;
    View views_[1];
    unsigned count_ = 0;
};

}  // namespace dynamic_embedding_rebase
