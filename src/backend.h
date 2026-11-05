#pragma once
#include <stdbool.h>

typedef struct {
    int ordinal;          // CUDA device ordinal
    char name[96];
    char busid[16];       // "01:00.0"
    int bus;
    int cc_major, cc_minor;
    int sms;
    unsigned long long mem;
} gpu_info;

typedef struct gpu_dev gpu_dev;

int gpu_list(gpu_info *out, int cap, char *err, int errlen);   // number of GPUs or -1
// Creates the context, JIT-loads the kernels and runs the self-test. kernel: "auto"|"mma"|"dp4a".
// rows: 0 = auto. Registers the geometry with the miner. NULL if the device can't mine.
gpu_dev *gpu_prepare(int stat_idx, const gpu_info *info, const char *kernel, int rows);
int gpu_launch(gpu_dev *d);       // starts the mining thread (after miner_start)
void gpu_join_all(void);

typedef struct cpu_dev cpu_dev;
cpu_dev *cpu_prepare(int stat_idx, int threads, int rows);
int cpu_launch(cpu_dev *d);

extern volatile int g_benchmark;
