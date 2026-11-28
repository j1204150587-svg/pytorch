"""Modify only the pinned PTA repository consumer/producer notification protocol."""
from pathlib import Path
import sys

SOURCE = "torch_npu/csrc/core/npu/NPUQueue.cpp"


def apply(source):
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError("source differs around " + old[:72])
        source = source.replace(old, new)

    mark = "#include <acl/acl_rt.h>"
    replace(mark, mark + "\n" + Path(__file__).with_name("waiters.inc").read_text())
    mark = "        if (ret == false) {\n            SetWriteWorking(false);"
    replace(mark, mark.replace("            SetWriteWorking(false);",
            "            FullWaitRegistration registration(wakeStates[device_idx]);\n            SetWriteWorking(false);"))
    # An exiting producer must observe shutdown even if an earlier ClearQueue broadcast already retired.
    mark = "            if (IsFullQueue()) {"
    replace(mark, """            if (IsFullQueue() && (!scopedProducerWake() ||
                    GetStatus() == RUN || GetStatus() == INIT)) {
                if (auditProducerWake())
                    wakeStates[device_idx].fullReads.fetch_add(1, std::memory_order_relaxed);""")
    mark = "        while (!IsWriteWorking()) {"
    replace(mark, """        auto& wake = wakeStates[device_idx];
        const bool auditWake = auditProducerWake();
        const bool waitersOnly = scopedProducerWake();
        if (auditWake) wake.drained.fetch_add(1, std::memory_order_relaxed);
        // The consumer has published read_idx and performed the stock full fence.
        // A simultaneous waiter either sees the free slot or is visible to this load.
        const bool hasWaiter = (auditWake || waitersOnly) &&
                               wake.waiters.load(std::memory_order_seq_cst) != 0;
        while (waitersOnly ? hasWaiter : !IsWriteWorking()) {""")
    mark = 'ASCEND_LOGE("notify producer failed. s=%zd, errno=%s.", s, strerror(errno));\n                return;\n            }\n            break;'
    replace(mark, mark.replace("            break;", """            if (auditWake) {
                wake.normalWrites.fetch_add(1, std::memory_order_relaxed);
                if (!hasWaiter) wake.writesWithoutWaiter.fetch_add(1, std::memory_order_relaxed);
            }
            break;"""))
    mark = "    eventfd_write(efd_write, 1);"
    replace(mark, """    auto& wake = wakeStates[device_idx];
    // Mode 1 has one dequeue worker and potentially several blocked enqueue callers.
    // Each semaphore read consumes one credit. A clear must wake all registered callers.
    uint64_t credits = scopedProducerWake() ? wake.waiters.load(std::memory_order_seq_cst) : 1;
    if (scopedProducerWake()) {
        int ret;
        do {
            ret = eventfd_write(efd_write, credits == 0 ? 1 : credits);
        } while (ret < 0 && errno == EINTR);
        if (ret == 0 && auditProducerWake()) wake.clearWrites.fetch_add(1, std::memory_order_relaxed);
    } else {
        int ret = eventfd_write(efd_write, 1);
        if (ret == 0 && auditProducerWake()) wake.clearWrites.fetch_add(1, std::memory_order_relaxed);
    }""")
    mark = "    efd_write = eventfd(0, 0);"
    replace(mark, """    TORCH_INTERNAL_ASSERT(device_id >= 0 && device_id < C10_COMPILE_TIME_MAX_NPUS);
    if (auditProducerWake()) installProducerWakeReport();
    std::fprintf(stderr, "PTA_QUEUE_PRODUCER_WAKE_CONFIG device=%d enabled=%d audit=%d\\n",
                 static_cast<int>(device_id), scopedProducerWake() ? 1 : 0, auditProducerWake() ? 1 : 0);
    efd_write = eventfd(0, scopedProducerWake() ? EFD_SEMAPHORE : 0);""")
    return source


if __name__ == "__main__":
    path = Path(sys.argv[1]) / SOURCE
    path.write_text(apply(path.read_text()))
