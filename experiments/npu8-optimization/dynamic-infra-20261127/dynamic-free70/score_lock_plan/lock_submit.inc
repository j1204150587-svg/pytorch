    std::shared_ptr<void> retained;
    if (lockApi) {
      constexpr int64_t zeros[8] = {};
      void* handle = nullptr;
      // allocate_workspace(size, stream) and retained handle follow the
      // loaded eager driver's allocation, stream ownership and release.
      void* pointer = lockApi->allocate(sizeof(zeros), packed.stream, &handle);
      retained = std::shared_ptr<void>(handle, lockApi->release);
      TORCH_CHECK(pointer && handle, "Triton score lock allocation failed");
      lockApi->allocations.fetch_add(1, std::memory_order_relaxed);
      const auto status = rtMemcpy(pointer, sizeof(zeros), zeros, sizeof(zeros), RT_MEMCPY_HOST_TO_DEVICE);
      if (status != RT_ERROR_NONE) {
        return static_cast<int>(status);
      }
      lockApi->initialized.fetch_add(1, std::memory_order_relaxed);
      WritePointerAt(packed.args, syncLockOffset, pointer);
    }
