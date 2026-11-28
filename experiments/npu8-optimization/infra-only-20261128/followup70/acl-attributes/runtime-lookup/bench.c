#define _GNU_SOURCE
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sched.h>

extern void *original_lookup(void *,void *);
extern void *direct_lookup(void *,void *);
typedef void *(*lookup)(void *,void *);
static unsigned char streams[2];
void *runtime_word = streams;

static uint64_t now(clockid_t clock)
{
    struct timespec t;
    if (clock_gettime(clock,&t)) abort();
    return UINT64_C(1000000000)*t.tv_sec+t.tv_nsec;
}

static void run(char leg,unsigned round,unsigned n,lookup op)
{
    void *last=NULL;
    uint64_t wall=now(CLOCK_MONOTONIC_RAW), cpu=now(CLOCK_THREAD_CPUTIME_ID);
    for (unsigned i=0;i<n;++i) {
        last=op(NULL,&streams[i&1]);
    }
    cpu=now(CLOCK_THREAD_CPUTIME_ID)-cpu;
    wall=now(CLOCK_MONOTONIC_RAW)-wall;
    if (last!=&streams[(n-1)&1] || op(NULL,&streams[0])!=&streams[0]) abort();
    printf("%u,%u,%c,%u,%d,%" PRIu64 ",%" PRIu64 "\n",round,n,leg,n,sched_getcpu(),wall,cpu);
}

int main(void)
{
    for(unsigned i=0;i<4096;++i) {
        if (original_lookup(NULL,streams)!=streams || direct_lookup(NULL,streams)!=streams) return 2;
    }
    puts("round,batch,leg,calls,cpu,wall_ns,cpu_ns");
    const unsigned sizes[]={4096,16384,65536};
    for(unsigned r=0;r<51;++r) {
        for(unsigned shift=0;shift<3;++shift) {
            unsigned n=sizes[(r+shift)%3];
            const char *order=r%2==0?"ABBA":"BAAB";
            for(unsigned j=0;j<4;++j) {
                char l=order[j];
                run(l,r,n,l=='A'?original_lookup:direct_lookup);
            }
        }
    }
    return 0;
}
