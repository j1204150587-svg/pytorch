// The generated stock launcher obtains both entry points from this exact
// libnputils shared object. A pending queue callback owns a reference to it.
struct ScoreLockApi {
  using Allocate = void* (*)(uint64_t, void*, void**);
  using Release = void (*)(void*);
  std::shared_ptr<void> library;
  Allocate allocate = nullptr;
  Release release = nullptr;
  std::atomic<uint64_t> allocations{0}, initialized{0}, kernelReturns{0};

  explicit ScoreLockApi(const std::string& path) {
    void* handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    TORCH_CHECK(handle, "Cannot open Triton NPU utility library");
    library = std::shared_ptr<void>(handle, [](void* h) { dlclose(h); });
    allocate = reinterpret_cast<Allocate>(dlsym(handle, "triton_allocate_sync_block_lock"));
    release = reinterpret_cast<Release>(dlsym(handle, "triton_release_retained_tensor"));
    TORCH_CHECK(allocate && release, "Triton NPU lock helpers are unavailable");
  }
};
