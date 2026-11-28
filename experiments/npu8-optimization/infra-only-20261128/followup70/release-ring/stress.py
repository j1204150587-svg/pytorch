"""Exercise the two patched method bodies on the host with callbacks that check FIFO ownership."""
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
source = (here / "build/torch_npu/csrc/core/npu/NPUQueue.cpp").read_text()
def body(first, last):
    return source[source.index(first):source.index(last, source.index(first))]
ring = body("static constexpr size_t kReleaseQueueCapacity", "void ReleaseQueue::PushToReleaseQueue")
ring += body("bool ReleaseQueue::ReadFromReleaseQueue", "void ReleaseQueue::PopFromReleaseQueue")
code = r'''
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <thread>
static constexpr unsigned count = 1000000;
static unsigned published = 0, consumed = 0;
struct Slot { unsigned sequence = 0; unsigned check = 0; };
struct Manager {
    void CopyRealseParam(Slot *data, unsigned slot, void *arg) {
        unsigned n = *static_cast<unsigned *>(arg);
        assert(data[slot].sequence == 0 && ++published == n);
        data[slot] = {n, ~n};
    }
    void ReleaseParam(Slot *data, unsigned slot) {
        Slot s = data[slot];
        assert(s.sequence == ++consumed && s.check == ~s.sequence);
        data[slot] = {};
    }
};
static Manager mgr;
static Manager &releaseManager() { return mgr; }
static bool releaseAcquireEnabled() { return true; }
static bool auditReleaseRing() { return false; }
static std::atomic<unsigned> releasePublished, releaseConsumed;
struct ReleaseQueue {
    struct { volatile unsigned idx = 0; } read_idx, write_idx;
    Slot datas[8192];
    bool IsFullQueue() const { return (write_idx.idx+1)%8192 == read_idx.idx; }
    bool IsEmptyQueue() const { return read_idx.idx == write_idx.idx; }
    bool ReadFromReleaseQueue();
    bool WriteToReleaseQueue(void *);
};
''' + ring + r'''
int main() {
    ReleaseQueue q;
    assert(!q.ReadFromReleaseQueue());
    for(unsigned n=1; n<8192; ++n) assert(q.WriteToReleaseQueue(&n));
    unsigned next=8192;
    assert(!q.WriteToReleaseQueue(&next));
    std::thread consumer([&]{ while(consumed<count) if (!q.ReadFromReleaseQueue()) std::this_thread::yield(); });
    for(unsigned n=8192; n<=count; ++n) while(!q.WriteToReleaseQueue(&n)) std::this_thread::yield();
    consumer.join();
    assert(!q.ReadFromReleaseQueue() && consumed==published);
    std::printf("published=%u consumed=%u capacity=8192 initial_empty=pass full=pass final_empty=pass\n",published,consumed);
}
'''
file = here / "build/stress.cpp"
file.write_text(code)
exe = file.with_suffix("")
subprocess.run(["g++", "-O2", "-std=c++17", "-pthread", str(file), "-o", str(exe)], check=True)
for _ in range(3):
    subprocess.run([str(exe)], check=True, timeout=60)
