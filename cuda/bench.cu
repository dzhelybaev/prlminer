// ejjjminer --bench-mma: peak tensor-core throughput micro-benchmarks (no memory traffic).
// Every warp issues ACC independent mma chains on register operands; total MACs / time = ceiling
// the mining kernels can reach with that instruction.
#include "common.cuh"

#if PRL_MMA && PRL_TURING
DEVINL void mma_s4(i32 d[4], const u32 a[4], const u32 b[2]) {   // 4 x m8n8k32 = 8192 MAC
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[2]), "r"(b[1]));
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(d[2]), "+r"(d[3]) : "r"(a[1]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(d[2]), "+r"(d[3]) : "r"(a[3]), "r"(b[1]));
}
#elif PRL_MMA
DEVINL void mma_s4(i32 d[4], const u32 a[4], const u32 b[2]) {   // m16n8k64 = 8192 MAC
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#else
DEVINL void mma_s4(i32 d[4], const u32 a[4], const u32 b[2]) { __builtin_trap(); }
#endif

template <int KIND>
DEVINL void peak(u32 iters, u32 *out) {
    constexpr int ACC = 8;
    const u32 t = TID_X;
    u32 a[4] = {t * 0x01010101u, t + 7u, t ^ 0x55u, t * 3u}, b[2] = {t + 1u, t * 5u};
    i32 acc[ACC][4];
#pragma unroll
    for (int j = 0; j < ACC; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0;
    for (u32 i = 0; i < iters; i++) {
#pragma unroll
        for (int j = 0; j < ACC; j++) {
            if constexpr (KIND == 0) mma_s8(acc[j], a, b);
            else mma_s4(acc[j], a, b);
        }
    }
    i32 x = 0;
#pragma unroll
    for (int j = 0; j < ACC; j++) x ^= acc[j][0] ^ acc[j][1] ^ acc[j][2] ^ acc[j][3];
    if (x == 0x12345678) out[0] = (u32)x;
}

KERNEL void k_peak_s8(u32 iters, u32 *out) { peak<0>(iters, out); }   // 8 x 4096 MAC per iteration per warp
KERNEL void k_peak_s4(u32 iters, u32 *out) { peak<1>(iters, out); }   // 8 x 8192 MAC per iteration per warp
