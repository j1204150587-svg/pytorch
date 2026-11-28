#include <stdbool.h>
#include <stdint.h>
#include <time.h>

typedef bool (*context_query)(int32_t device);

int time_context_calls(context_query query, unsigned groups, unsigned per_group, uint64_t *durations)
{
    unsigned positives = 0;
    for (unsigned i = 0; i < groups; ++i) {
        struct timespec before, after;
        if (clock_gettime(CLOCK_MONOTONIC_RAW, &before)) return -1;
        for (unsigned j = 0; j < per_group; ++j) positives += query(0);
        if (clock_gettime(CLOCK_MONOTONIC_RAW, &after)) return -1;
        durations[i] = (uint64_t)(after.tv_sec - before.tv_sec) * 1000000000 +
            (uint64_t)(after.tv_nsec - before.tv_nsec);
    }
    return positives == groups * per_group ? 0 : -1;
}
