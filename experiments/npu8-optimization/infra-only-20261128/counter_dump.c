/* Collect the existing op-plugin counters at process exit in diagnostic runs. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <inttypes.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int record(struct dl_phdr_info *info, size_t size, void *context)
{
    (void)size;
    const char *name = strrchr(info->dlpi_name, '/');
    if (name == NULL || strcmp(name, "/libtorch_npu.so") != 0)
        return 0;
    void *handle = dlopen(info->dlpi_name, RTLD_NOLOAD | RTLD_LAZY);
    if (handle == NULL)
        return 0;
#ifdef EMBEDDING_COUNTERS
    const char *symbols[] = {"npu_embedding_backward_rebase_counter", "npu_embedding_forward_rebase_counter"};
#else
    const char *symbols[] = {"npu_addmm_rebase_counter", "npu_mm_rebase_counter"};
#endif
    for (unsigned i = 0; i < sizeof(symbols) / sizeof(symbols[0]); ++i) {
        uint64_t (*counter)(unsigned) = dlsym(handle, symbols[i]);
        if (counter == NULL)
            continue;
        char filename[4096];
        int n = snprintf(filename, sizeof(filename), "%s/%s-%d.json",
                         (const char *)context, symbols[i], (int)getpid());
        if (n < 0 || n >= (int)sizeof(filename))
            continue;
        FILE *out = fopen(filename, "wx");
        if (out == NULL)
            continue;
        fprintf(out, "{\"pid\":%d,\"symbol\":\"%s\",\"library\":\"%s\","
                     "\"accepted\":%" PRIu64 ",\"fallback\":%" PRIu64
                     ",\"keys\":%" PRIu64 ",\"descriptors\":%" PRIu64 "}\n",
                (int)getpid(), symbols[i], info->dlpi_name, counter(0), counter(1), counter(2), counter(3));
        fclose(out);
    }
    dlclose(handle);
    return 1;
}
__attribute__((destructor)) static void dump_counters(void)
{
    const char *directory = getenv("INFRA_COUNTER_DIR");
    if (directory != NULL)
        dl_iterate_phdr(record, (void *)directory);
}

