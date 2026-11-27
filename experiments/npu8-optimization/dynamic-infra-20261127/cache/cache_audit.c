/* Diagnostic-only glibc rtld-audit module for the Mode 1 CANN cache API. */
#define _GNU_SOURCE
#include <link.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

typedef bool (*can_use_fn)(const char *);
typedef void *(*get_exec_fn)(uint64_t, uint64_t *);
static can_use_fn original_can_use;
static get_exec_fn original_get_exec;
static const char *get_exec_library;
static __thread const char *current_op;
static __thread FILE *events;

unsigned int la_version(unsigned int version)
{
    return version >= LAV_CURRENT ? LAV_CURRENT : 0;
}

unsigned int la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie)
{
    (void)lmid;
    *cookie = (uintptr_t)map;
    return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}

static FILE *event_file(void)
{
    if (!events) {
        const char *directory = getenv("NPU_CACHE_DIAG_DIR");
        char name[4096];
        if (!directory || snprintf(name, sizeof(name), "%s/lookup-%d-%ld.tsv", directory,
                                   (int)getpid(), syscall(SYS_gettid)) >= (int)sizeof(name)) {
            return NULL;
        }
        events = fopen(name, "a");
        if (events) {
            fprintf(events, "# source\t%s\n", get_exec_library ? get_exec_library : "pending");
        }
    }
    return events;
}

static void record(const char *outcome, uint64_t key, uint64_t workspace)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    FILE *stream = event_file();
    if (stream) {
        fprintf(stream, "%lld\t%s\t%s\t%016llx\t%llu\n",
                (long long)now.tv_sec * 1000000000LL + now.tv_nsec,
                current_op ? current_op : "UNKNOWN", outcome,
                (unsigned long long)key, (unsigned long long)workspace);
    }
}

static bool trace_can_use(const char *api)
{
    bool allowed = original_can_use(api);
    current_op = api;
    if (!allowed) {
        record("ineligible", 0, 0);
    }
    return allowed;
}

static void *trace_get_exec(uint64_t key, uint64_t *workspace)
{
    void *exec = original_get_exec(key, workspace);
    record(exec ? "hit" : "miss", key, exec && workspace ? *workspace : 0);
    return exec;
}

uintptr_t la_symbind64(Elf64_Sym *symbol, unsigned int index,
                       uintptr_t *caller, uintptr_t *owner,
                       unsigned int *flags, const char *name)
{
    (void)index;
    (void)caller;
    (void)flags;
    uintptr_t addr = symbol->st_value;
    if (strcmp(name, "PTAGetExecCache") == 0) {
        if (original_get_exec && (uintptr_t)original_get_exec != addr) return addr;
        original_get_exec = (get_exec_fn)addr;
        get_exec_library = ((struct link_map *)*owner)->l_name;
        return (uintptr_t)&trace_get_exec;
    }
    if (strcmp(name, "CanUsePTACache") == 0) {
        if (original_can_use && (uintptr_t)original_can_use != addr) return addr;
        original_can_use = (can_use_fn)addr;
        return (uintptr_t)&trace_can_use;
    }
    return addr;
}

__attribute__((destructor)) static void flush_events(void)
{
    fflush(NULL);
}
