// Read-only timing around the CANN driver's public submit function.
#include <ascend_hal_base.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace {
using Submit = decltype(&halSqTaskSend);
struct Row {
    uint64_t start, end;
    uint32_t device, ts, sq, count, position;
    int32_t status;
};
static_assert(sizeof(Row) == 40, "unexpected trace format");
struct Header { uint64_t capacity, used, dropped, recordSize; };
constexpr size_t capacity = 400000;
struct Log {
    Header *header = nullptr;
    Row *rows = nullptr;
    Log()
    {
        const char *folder = std::getenv("NPU_HAL_SUBMIT_LOG");
        const char *enable = std::getenv("NPU_HAL_SUBMIT_OBSERVE");
        const char *rank = std::getenv("LOCAL_RANK");
        if (!folder || !enable || std::strcmp(enable,"1") || !rank || std::strcmp(rank,"0")) {
            return;
        }
        char name[PATH_MAX];
        const auto tid = syscall(SYS_gettid);
        int n = std::snprintf(name,sizeof(name),"%s/%ld-%ld.bin",folder,static_cast<long>(getpid()),tid);
        if (n < 0 || static_cast<size_t>(n) >= sizeof(name)) { return; }
        int file = open(name,O_CREAT|O_EXCL|O_RDWR|O_CLOEXEC,0600);
        if (file < 0) { return; }
        size_t bytes = sizeof(Header) + capacity * sizeof(Row);
        void *mem = MAP_FAILED;
        if (ftruncate(file,bytes) == 0) { mem = mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,file,0); }
        close(file);
        if (mem != MAP_FAILED) {
            header = static_cast<Header*>(mem);
            *header = {capacity,0,0,sizeof(Row)};
            rows = reinterpret_cast<Row*>(header+1);
        }
    }
    // CANN may submit work from later thread-exit handlers. Keep the diagnostic
    // mapping available until process exit, including during TLS teardown.
};
uint64_t nanos()
{
    struct timespec time;
    clock_gettime(CLOCK_REALTIME,&time);
    return static_cast<uint64_t>(time.tv_sec)*1000000000ULL + time.tv_nsec;
}
}
extern "C" drvError_t halSqTaskSend(uint32_t device, struct halTaskSendInfo *info)
{
    static Submit real = [] {
        auto sym = dlsym(RTLD_NEXT,"halSqTaskSend");
        if (!sym) {
            void *lib = dlopen("libascend_hal.so",RTLD_NOW|RTLD_NOLOAD);
            if (lib) { sym = dlsym(lib,"halSqTaskSend"); }
        }
        if (!sym) { std::fprintf(stderr,"HAL submit observer cannot locate driver symbol\n"); std::abort(); }
        return reinterpret_cast<Submit>(sym);
    }();
    int entry_errno = errno;
    static thread_local Log log;
    errno=entry_errno;
    if (!log.header || !info) { return real(device,info); }
    if (log.header->used >= capacity) { ++log.header->dropped; return real(device,info); }
    Row row{};
    row.device=device;
    row.ts=info->tsId;
    row.sq=info->sqId;
    row.count=info->sqe_num;
    row.start=nanos();
    drvError_t result=real(device,info);
    int saved_errno=errno;
    row.end=nanos();
    if (result==0) { row.position=info->pos; }
    row.status=static_cast<int>(result);
    log.rows[log.header->used]=row;
    ++log.header->used;
    errno=saved_errno;
    return result;
}
