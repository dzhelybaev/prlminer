// Freestanding CUDA definitions.
//  * device build: clang -x cuda -nocudainc (no CUDA toolkit needed), PTX via inline asm.
//  * PRL_EMU build: the same kernel source compiled as host C++ and executed by a small
//    SIMT emulator (tests/emu.cpp) — one OS thread per CUDA thread, real barriers, and a
//    faithful model of the mma.sync m16n8k32 fragment layout from the PTX ISA. Used to
//    test kernel indexing/pipelining logic on machines without a GPU.
#pragma once
typedef unsigned char u8;
typedef signed char i8;
typedef unsigned int u32;
typedef int i32;
typedef unsigned long long u64;

#ifndef PRL_EMU
// ============================================================================ device
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define DEVINL static __device__ __attribute__((always_inline)) inline
#define LAUNCH_BOUNDS(a, b) __attribute__((launch_bounds(a, b)))
#define KERNEL extern "C" __global__
#define CONSTANT __constant__

#define TID_X  ((u32)__nvvm_read_ptx_sreg_tid_x())
#define BID_X  ((u32)__nvvm_read_ptx_sreg_ctaid_x())
#define BID_Y  ((u32)__nvvm_read_ptx_sreg_ctaid_y())
#define DYN_SMEM(name) extern __shared__ __attribute__((aligned(16))) u8 name[]
#define BAR_SYNC() __syncthreads()

DEVINL u32 smem_addr(const void *p) {
    u32 r;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p));
    return r;
}
#if PRL_MMA && !PRL_TURING
DEVINL void cp_async16(void *smem_dst, const void *gsrc) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_addr(smem_dst)), "l"(gsrc));
}
DEVINL void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> DEVINL void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
#else
struct alignas(16) u32x4 { u32 x, y, z, w; };
DEVINL void cp_async16(void *smem_dst, const void *gsrc) { *(u32x4 *)smem_dst = *(const u32x4 *)gsrc; }
DEVINL void cp_async_commit() {}
template <int N> DEVINL void cp_async_wait() {}
#endif

DEVINL u32 atomic_add_u32(u32 *p, u32 v) { return (u32)__nvvm_atom_add_gen_i((int *)p, (int)v); }
DEVINL u32 byte_perm(u32 a, u32 b, u32 s) {
    u32 d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
    return d;
}

DEVINL i32 dp4a(u32 a, u32 b, i32 c) {
    i32 d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

#if PRL_MMA && PRL_TURING
// Turing (sm_75): m16n8k32 expressed as four m8n8k16 ops. Fragment layouts line up exactly:
//   m8n8k16 A = 1 reg (row g, k 4q..4q+3), B = 1 reg (k 4q..4q+3, col g), C = 2 regs (row g, cols 2q, 2q+1)
DEVINL void mma_m8n8k16(i32 *d, u32 a, u32 b) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a), "r"(b));
}
DEVINL void mma_s8(i32 d[4], const u32 a[4], const u32 b[2]) {
    mma_m8n8k16(d, a[0], b[0]);       // rows 0-7,  k 0-15
    mma_m8n8k16(d, a[2], b[1]);       // rows 0-7,  k 16-31
    mma_m8n8k16(d + 2, a[1], b[0]);   // rows 8-15, k 0-15
    mma_m8n8k16(d + 2, a[3], b[1]);   // rows 8-15, k 16-31
}
#elif PRL_MMA
// D = A(16x32 s8, row) * B(32x8 s8, col) + C (s32)
DEVINL void mma_s8(i32 d[4], const u32 a[4], const u32 b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#else
DEVINL void mma_s8(i32 d[4], const u32 a[4], const u32 b[2]) { __builtin_trap(); }
#endif

#else
// ============================================================================ emulator
#include "../tests/emu.h"
#define __device__
#define DEVINL static inline
#define LAUNCH_BOUNDS(a, b)
#define KERNEL extern "C"
#define CONSTANT static const
#define TID_X  (emu_tid())
#define BID_X  (emu_bid_x())
#define BID_Y  (emu_bid_y())
#define DYN_SMEM(name) u8 *name = emu_smem()
#define BAR_SYNC() emu_bar_sync()
DEVINL void cp_async16(void *dst, const void *src) { __builtin_memcpy(dst, src, 16); }
DEVINL void cp_async_commit() {}
template <int N> DEVINL void cp_async_wait() {}
DEVINL u32 atomic_add_u32(u32 *p, u32 v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
DEVINL u32 byte_perm(u32 a, u32 b, u32 s) {
    u64 x = (u64)a | ((u64)b << 32);
    u32 r = 0;
    for (int i = 0; i < 4; i++) r |= (u32)((x >> (8 * ((s >> (4 * i)) & 7))) & 0xff) << (8 * i);
    return r;
}
DEVINL i32 dp4a(u32 a, u32 b, i32 c) {
    for (int i = 0; i < 4; i++) c += (i32)(i8)(a >> (8 * i)) * (i32)(i8)(b >> (8 * i));
    return c;
}
#ifdef PRL_TURING
DEVINL void mma_s8(i32 d[4], const u32 a[4], const u32 b[2]) {
    emu_mma_m8n8k16(d, a[0], b[0]);
    emu_mma_m8n8k16(d, a[2], b[1]);
    emu_mma_m8n8k16(d + 2, a[1], b[0]);
    emu_mma_m8n8k16(d + 2, a[3], b[1]);
}
#else
DEVINL void mma_s8(i32 d[4], const u32 a[4], const u32 b[2]) { emu_mma_s8(d, a, b); }
#endif
#endif

// per-byte wrapping add / subtract of packed int8 (SWAR)
DEVINL u32 badd4(u32 a, u32 b) { return ((a & 0x7f7f7f7fu) + (b & 0x7f7f7f7fu)) ^ ((a ^ b) & 0x80808080u); }
DEVINL u32 bsub4(u32 a, u32 b) { return ((a | 0x80808080u) - (b & 0x7f7f7f7fu)) ^ ((a ^ ~b) & 0x80808080u); }
// 4x4 byte transpose: y[i].byte[j] = x[j].byte[i]
DEVINL void transpose4(u32 x0, u32 x1, u32 x2, u32 x3, u32 y[4]) {
    u32 t0 = byte_perm(x0, x1, 0x5140), t1 = byte_perm(x0, x1, 0x7362);
    u32 t2 = byte_perm(x2, x3, 0x5140), t3 = byte_perm(x2, x3, 0x7362);
    y[0] = byte_perm(t0, t2, 0x5410); y[1] = byte_perm(t0, t2, 0x7632);
    y[2] = byte_perm(t1, t3, 0x5410); y[3] = byte_perm(t1, t3, 0x7632);
}

DEVINL u32 rotl32(u32 x, u32 n) { return (x << n) | (x >> (32 - n)); }
DEVINL u32 rotr32(u32 x, u32 n) { return (x >> n) | (x << (32 - n)); }

// ------------------------------------------------------------------ BLAKE3 (single block)
#define B3_CHUNK_START 1u
#define B3_CHUNK_END   2u
#define B3_PARENT      4u
#define B3_ROOT        8u
#define B3_KEYED       16u

#define B3G(a, b, c, d, x, y)                     \
    do {                                          \
        a = a + b + (x); d = rotr32(d ^ a, 16);   \
        c = c + d;       b = rotr32(b ^ c, 12);   \
        a = a + b + (y); d = rotr32(d ^ a, 8);    \
        c = c + d;       b = rotr32(b ^ c, 7);    \
    } while (0)

// out[8] = first 8 words of compress(cv, m, counter=0, block_len=64, flags)
DEVINL void b3_compress(const u32 cv[8], const u32 m_in[16], u32 flags, u32 out[8]) {
    u32 m[16];
#pragma unroll
    for (int i = 0; i < 16; i++) m[i] = m_in[i];
    u32 v0 = cv[0], v1 = cv[1], v2 = cv[2], v3 = cv[3], v4 = cv[4], v5 = cv[5], v6 = cv[6], v7 = cv[7];
    u32 v8 = 0x6A09E667u, v9 = 0xBB67AE85u, v10 = 0x3C6EF372u, v11 = 0xA54FF53Au;
    u32 v12 = 0, v13 = 0, v14 = 64, v15 = flags;
#pragma unroll
    for (int r = 0; r < 7; r++) {
        B3G(v0, v4, v8, v12, m[0], m[1]);
        B3G(v1, v5, v9, v13, m[2], m[3]);
        B3G(v2, v6, v10, v14, m[4], m[5]);
        B3G(v3, v7, v11, v15, m[6], m[7]);
        B3G(v0, v5, v10, v15, m[8], m[9]);
        B3G(v1, v6, v11, v12, m[10], m[11]);
        B3G(v2, v7, v8, v13, m[12], m[13]);
        B3G(v3, v4, v9, v14, m[14], m[15]);
        if (r < 6) {  // message permutation [2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]
            u32 t0 = m[2], t1 = m[6], t2 = m[3], t3 = m[10], t4 = m[7], t5 = m[0], t6 = m[4], t7 = m[13];
            u32 t8 = m[1], t9 = m[11], t10 = m[12], t11 = m[5], t12 = m[9], t13 = m[14], t14 = m[15], t15 = m[8];
            m[0] = t0; m[1] = t1; m[2] = t2; m[3] = t3; m[4] = t4; m[5] = t5; m[6] = t6; m[7] = t7;
            m[8] = t8; m[9] = t9; m[10] = t10; m[11] = t11; m[12] = t12; m[13] = t13; m[14] = t14; m[15] = t15;
        }
    }
    out[0] = v0 ^ v8; out[1] = v1 ^ v9; out[2] = v2 ^ v10; out[3] = v3 ^ v11;
    out[4] = v4 ^ v12; out[5] = v5 ^ v13; out[6] = v6 ^ v14; out[7] = v7 ^ v15;
}

// keyed BLAKE3 of one 64-byte message (root output)
DEVINL void b3_keyed64(const u32 key[8], const u32 m[16], u32 out[8]) {
    b3_compress(key, m, B3_CHUNK_START | B3_CHUNK_END | B3_ROOT | B3_KEYED, out);
}
