#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__attribute__((destructor)) static void report_pending_counts(void)
{
    const char *dir=getenv("INFRA_COUNTER_DIR");
    if (!dir || !*dir) return;
    uint64_t (*read_count)(unsigned)=(uint64_t (*)(unsigned))dlsym(RTLD_DEFAULT,"npu_cann_pending_count");
    if (!read_count) return;
    uint64_t ll=read_count(0), lse=read_count(1);
    if (!ll && !lse) return;
    Dl_info info={0};
    if (!dladdr((void*)read_count,&info) || !info.dli_fname) return;
    char path[4096];
    int n=snprintf(path,sizeof(path),"%s/cann-pending-%ld.json",dir,(long)getpid());
    if (n<=0 || (size_t)n>=sizeof(path)) return;
    FILE *out=fopen(path,"w");
    if (!out) return;
    fprintf(out,"{\"llsc\":%llu,\"lse\":%llu,\"hwcap_atomics\":%llu,\"selected\":%llu,\"mapped\":\"%s\"}\n",
            (unsigned long long)ll,(unsigned long long)lse,(unsigned long long)read_count(2),
            (unsigned long long)read_count(3),info.dli_fname);
    fclose(out);
}
