// SIMT emulator: one pthread per CUDA thread, real block/warp barriers, and an
// mma.sync.m16n8k32.row.col.s32.s8.s8.s32 model following the PTX ISA fragment layout:
//   A (16x32): a0 = A[g][4q..4q+3], a1 = A[g+8][4q..], a2 = A[g][16+4q..], a3 = A[g+8][16+4q..]
//   B (32x8, col): b0 = B[4q..4q+3][g], b1 = B[16+4q..16+4q+3][g]
//   C/D (16x8): d0 = (g, 2q), d1 = (g, 2q+1), d2 = (g+8, 2q), d3 = (g+8, 2q+1)
#include "emu.h"
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <vector>

struct WarpScratch {
    u32 a[32][4], b[32][2];
    pthread_barrier_t bar;
};

static thread_local u32 t_tid, t_bx, t_by;
static thread_local u8 *t_smem;
static pthread_barrier_t g_block_bar;
static WarpScratch *g_warps;

u32 emu_tid() { return t_tid; }
u32 emu_bid_x() { return t_bx; }
u32 emu_bid_y() { return t_by; }
u8 *emu_smem() { return t_smem; }
void emu_bar_sync() { pthread_barrier_wait(&g_block_bar); }

void emu_mma_s8(i32 d[4], const u32 a[4], const u32 b[2]) {
    WarpScratch &w = g_warps[t_tid / 32];
    u32 lane = t_tid & 31;
    memcpy(w.a[lane], a, 16);
    memcpy(w.b[lane], b, 8);
    pthread_barrier_wait(&w.bar);
    int8_t A[16][32], B[32][8];
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, q = L & 3;
        for (int i = 0; i < 4; i++) {
            A[g][4 * q + i] = (int8_t)(w.a[L][0] >> (8 * i));
            A[g + 8][4 * q + i] = (int8_t)(w.a[L][1] >> (8 * i));
            A[g][16 + 4 * q + i] = (int8_t)(w.a[L][2] >> (8 * i));
            A[g + 8][16 + 4 * q + i] = (int8_t)(w.a[L][3] >> (8 * i));
            B[4 * q + i][g] = (int8_t)(w.b[L][0] >> (8 * i));
            B[16 + 4 * q + i][g] = (int8_t)(w.b[L][1] >> (8 * i));
        }
    }
    pthread_barrier_wait(&w.bar);   // everyone has read the scratch
    int g = lane >> 2, q = lane & 3;
    int rows[4] = {g, g, g + 8, g + 8}, cols[4] = {2 * q, 2 * q + 1, 2 * q, 2 * q + 1};
    for (int e = 0; e < 4; e++) {
        int32_t s = 0;
        for (int kk = 0; kk < 32; kk++) s += (int32_t)A[rows[e]][kk] * (int32_t)B[kk][cols[e]];
        d[e] += s;
    }
}

// mma.sync.m8n8k16.row.col.s32.s8.s8.s32 (PTX ISA): A 8x16 row: a = A[g][4q..4q+3];
// B 16x8 col: b = B[4q..4q+3][g]; C 8x8: d0 = (g, 2q), d1 = (g, 2q+1)
void emu_mma_m8n8k16(i32 *d, u32 a, u32 b) {
    WarpScratch &w = g_warps[t_tid / 32];
    u32 lane = t_tid & 31;
    w.a[lane][0] = a;
    w.b[lane][0] = b;
    pthread_barrier_wait(&w.bar);
    int8_t A[8][16], B[16][8];
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, q = L & 3;
        for (int i = 0; i < 4; i++) {
            A[g][4 * q + i] = (int8_t)(w.a[L][0] >> (8 * i));
            B[4 * q + i][g] = (int8_t)(w.b[L][0] >> (8 * i));
        }
    }
    pthread_barrier_wait(&w.bar);
    int g = lane >> 2, q = lane & 3;
    for (int e = 0; e < 2; e++) {
        int32_t s = 0;
        for (int kk = 0; kk < 16; kk++) s += (int32_t)A[g][kk] * (int32_t)B[kk][2 * q + e];
        d[e] += s;
    }
}

struct ThreadArg {
    u32 tid, bx, by;
    u8 *smem;
    const std::function<void()> *body;
};

static void *thread_main(void *p) {
    ThreadArg *a = (ThreadArg *)p;
    t_tid = a->tid; t_bx = a->bx; t_by = a->by; t_smem = a->smem;
    (*a->body)();
    return nullptr;
}

static pthread_mutex_t g_launch_mu = PTHREAD_MUTEX_INITIALIZER;   // emulator is not reentrant

void emu_launch(u32 gx, u32 gy, u32 block, size_t smem_bytes, const std::function<void()> &body) {
    pthread_mutex_lock(&g_launch_mu);
    std::vector<u8> smem(smem_bytes + 16);
    std::vector<WarpScratch> warps((block + 31) / 32);
    g_warps = warps.data();
    for (auto &w : warps) pthread_barrier_init(&w.bar, nullptr, 32);
    pthread_barrier_init(&g_block_bar, nullptr, block);
    std::vector<pthread_t> th(block);
    std::vector<ThreadArg> args(block);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 256 * 1024);
    for (u32 by = 0; by < gy; by++)
        for (u32 bx = 0; bx < gx; bx++) {
            memset(smem.data(), 0xcd, smem.size());   // garbage, like real shared memory
            for (u32 t = 0; t < block; t++) {
                args[t] = {t, bx, by, smem.data(), &body};
                pthread_create(&th[t], &attr, thread_main, &args[t]);
            }
            for (u32 t = 0; t < block; t++) pthread_join(th[t], nullptr);
        }
    pthread_attr_destroy(&attr);
    pthread_barrier_destroy(&g_block_bar);
    for (auto &w : warps) pthread_barrier_destroy(&w.bar);
    pthread_mutex_unlock(&g_launch_mu);
}

void emu_launch_serial(u32 gx, u32 gy, u32 block, const std::function<void()> &body) {
    for (u32 by = 0; by < gy; by++)
        for (u32 bx = 0; bx < gx; bx++)
            for (u32 t = 0; t < block; t++) {
                t_tid = t; t_bx = bx; t_by = by; t_smem = nullptr;
                body();
            }
}
