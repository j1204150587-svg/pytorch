#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"

extern "C" __attribute__((visibility("default"))) long option_test_value(const char *value)
{
    if (value) {
        c10_npu::option::register_options::OptionRegister::GetInstance()->Set("TORCH_NPU_USE_COMPATIBLE_IMPL", value);
    }
    int observed[4] = {};
    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&,i] { observed[i] = at_npu::native::env::CheckCompatibleImpl(); });
    }
    for (auto &worker : readers) { worker.join(); }
    int sum = 0;
    for (auto v : observed) { sum += v; }
    return sum;
}

extern "C" __attribute__((visibility("default"))) uint64_t option_test_time(unsigned n, unsigned *truths)
{
    unsigned count = 0;
    auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < n; ++i) { count += at_npu::native::env::CheckCompatibleImpl(); }
    auto end = std::chrono::steady_clock::now();
    *truths = count;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
}
