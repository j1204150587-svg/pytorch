"""Use one process-wide atomic sequence while keeping NPUStorageImpl layout intact."""
from pathlib import Path

SOURCE = "torch_npu/csrc/core/NPUStorageImpl.cpp"


def apply(source):
    before = '''    std::lock_guard<std::mutex> lock(unique_id_mutex_);
    static uint64_t idx = 0;
    unique_id_ = idx++;'''
    after = '''    const bool audit = auditIds();
    const auto start = audit ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (atomicIds()) {
        static std::atomic<uint64_t> next_id{0};
        unique_id_ = next_id.fetch_add(1, std::memory_order_relaxed);
    } else {
        std::lock_guard<std::mutex> lock(unique_id_mutex_);
        static uint64_t idx = 0;
        unique_id_ = idx++;
    }
    if (audit) countId(start);'''
    mark="namespace torch_npu {"
    if source.count(mark)!=1 or source.count(before)!=1:
        raise ValueError("unexpected pinned storage constructor")
    return source.replace(mark,Path(__file__).with_name("config.inc").read_text()+"\n"+mark).replace(before,after)
