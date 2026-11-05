// ejjjminer CUDA kernels: noise generation, noising, and the fused
// noised-int8-GEMM + jackpot transcript + keyed-BLAKE3 target check.
//
// Hash-tile geometry (matches the mma.sync m16n8k32 accumulator layout):
//   lane (g = lane/4, q = lane%4) of a warp owns, for each m16 sub-tile mt, the hash tile
//   rows {r0+16mt+g, r0+16mt+g+8} x cols {c0 + 8j + 2q + e : j < NT, e < 2}
//   i.e. rows_pattern [0,8], cols_pattern {0,1,8,9,...,8(NT-1),8(NT-1)+1}.
//   NT=32 is the pattern of the official Pearl miner (2x64 tile over 256 columns).
//
// Block tile 128 x 256, 8 warps. Warp tile (16*MT) x (8*NT) with MT*NT = 32:
//   NT=32,MT=1 (8x1 warps) | NT=16,MT=2 (4x2 warps) | NT=8,MT=4 (2x4 warps).
#include "common.cuh"

#define BM 128
#define BN 256
#define BK 32
#define SROW 48          // smem row stride in bytes (32 data + 16 pad: conflict-free 32-bit loads)
#define NTHREADS 256
#if defined(PRL_STAGES)
#define STAGES PRL_STAGES
#elif defined(PRL_EMU) || (PRL_MMA && !PRL_TURING)
#define STAGES 3
#else
#define STAGES 2         // sm_61/sm_75: synchronous copies, 48 KB shared memory
#endif
#define STAGE_BYTES ((BM + BN) * SROW)
#define NCAND_WORDS 4    // candidate record: t_rows, t_cols, hash word 7, hash word 6

// ------------------------------------------------------------------ noise generation
#if defined(PRL_EMU) || defined(K_NOISE)
// uniform: out[idx*32 + i] = (H(idx)[i] & 63) - 32, H(idx) = B3_keyed(key, [1+idx,0..0 | label])
KERNEL void k_noise_uniform(const u32 *key8, u32 label_b, u32 nblocks, i8 *out) {
    u32 idx = BID_X * NTHREADS + TID_X;
    if (idx >= nblocks) return;
    u32 key[8], m[16], h[8];
#pragma unroll
    for (int i = 0; i < 8; i++) key[i] = key8[i];
#pragma unroll
    for (int i = 0; i < 16; i++) m[i] = 0;
    m[0] = 1 + idx;
    m[8] = label_b ? 0x65745f42u : 0x65745f41u;   // "B_te" / "A_te" (LE)
    m[9] = 0x726f736eu;                           // "nsor"
    b3_keyed64(key, m, h);
    u32 *o = (u32 *)(out + (u64)idx * 32);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        u32 w = h[i], r = 0;
#pragma unroll
        for (int b = 0; b < 4; b++) r |= (u32)(u8)((((w >> (8 * b)) & 63) - 32)) << (8 * b);
        o[i] = r;
    }
}

// permutation pairs: line l = 8i + j uses word j of H(i) with slot 1
KERNEL void k_noise_perm(const u32 *key8, u32 label_b, u32 k, u8 *pairs) {
    u32 i = BID_X * NTHREADS + TID_X;
    if (i * 8 >= k) return;
    u32 key[8], m[16], h[8];
#pragma unroll
    for (int t = 0; t < 8; t++) key[t] = key8[t];
#pragma unroll
    for (int t = 0; t < 16; t++) m[t] = 0;
    m[1] = 1 + i;
    m[8] = label_b ? 0x65745f42u : 0x65745f41u;
    m[9] = 0x726f736eu;
    b3_keyed64(key, m, h);
#pragma unroll
    for (int j = 0; j < 8; j++) {
        u32 u = h[j];
        u32 first = u & 127u;
        u32 second = first ^ (1u + (u32)(((u64)127u * u) >> 32));
        if (i * 8 + j < k) {
            pairs[2 * (i * 8 + j)] = (u8)first;
            pairs[2 * (i * 8 + j) + 1] = (u8)second;
        }
    }
}

// out[r][l] = base[r][l] + uni[r][first_l] - uni[r][second_l].
// One block handles APPLY_ROWS rows; the permutation pairs (2k bytes, shared by all rows) and the
// rows' 128-byte uniform vectors are staged in shared memory. Dynamic smem: 2k + APPLY_ROWS*128.
// patch != 0: base[0][0..15] is replaced by the 16 nonce nibbles.
#define APPLY_ROWS 16
KERNEL void k_apply_noise(const i8 *base, const i8 *uni, const u8 *pairs, i8 *out, u32 rows, u32 k,
                          u32 nonce_lo, u32 nonce_hi, u32 patch) {
    DYN_SMEM(sm);
    u8 *sp = sm;                       // 2k bytes
    i8 *su = (i8 *)(sm + 2 * k);       // APPLY_ROWS x 128
    const u32 tid = TID_X, r0 = BID_X * APPLY_ROWS;
    for (u32 i = tid; i < 2 * k / 16; i += NTHREADS) cp_async16(sp + 16 * i, pairs + 16 * i);
    for (u32 i = tid; i < APPLY_ROWS * 128 / 16; i += NTHREADS) {
        u32 rr = r0 + (16 * i) / 128;
        if (rr < rows) cp_async16(su + 16 * i, uni + (u64)r0 * 128 + 16 * i);
    }
    cp_async_commit();
    cp_async_wait<0>();
    BAR_SYNC();
    const u32 segs = k / 16;
    for (u32 w = tid; w < APPLY_ROWS * segs; w += NTHREADS) {
        u32 rl = w / segs, s = w % segs, r = r0 + rl;
        if (r >= rows) break;
        const i8 *ur = su + rl * 128;
        const u32 *src = (const u32 *)(base + (u64)r * k + 16 * s);
        const u32 *pp = (const u32 *)(sp + 32 * s);
        u32 *dst = (u32 *)(out + (u64)r * k + 16 * s);
        bool nonce = patch && r == 0 && s == 0;
        u32 bw4[4] = {src[0], src[1], src[2], src[3]}, res4[4];
#pragma unroll
        for (int q = 0; q < 4; q++) {
            u32 bw = bw4[q], p0 = pp[2 * q], p1 = pp[2 * q + 1], res = 0;
#pragma unroll
            for (int b = 0; b < 4; b++) {
                int e = 4 * q + b;
                u32 pw = b < 2 ? p0 : p1;
                int f = (pw >> (16 * (b & 1))) & 0xff, sec = (pw >> (16 * (b & 1) + 8)) & 0xff;
                int v = (int)(i8)(bw >> (8 * b));
                if (nonce) v = (int)(((e < 8 ? (nonce_lo >> (4 * e)) : (nonce_hi >> (4 * (e - 8))))) & 15u);
                v += (int)ur[f] - (int)ur[sec];
                res |= (u32)(u8)(i8)v << (8 * b);
            }
            res4[q] = res;
        }
        dst[0] = res4[0]; dst[1] = res4[1]; dst[2] = res4[2]; dst[3] = res4[3];
    }
}

#endif  // K_NOISE

// ------------------------------------------------------------------ fused mining kernel
// grid: (n/BN, m/BM), block 256. A: m x k (noised), Bt: n x k (noised).
// For every hash tile: transcript over k/128 chunks, H = B3_keyed(seed_a, transcript),
// H (LE u256) <= bound -> candidate. dbg (optional): all transcripts, 16 words per tile.
//
// FUSED: A is not read from memory. The base A is all zero except the nonce nibbles in row 0
// (bytes 0..15), so A'[r][l] = nonce(r,l) + U[r][first_l] - U[r][second_l] is generated straight
// into shared memory from the block's 128 uniform rows U (staged transposed, 16 KB) and the
// permutation pairs P. Fused A tiles use a 32-byte row stride with an XOR word swizzle
// (conflict-free mma fragment loads). Extra dynamic smem: FUSED_SMEM bytes.
#define FUSED_SMEM (128 * 128)
DEVINL u32 a_swz(u32 row) { return (((row >> 2) & 1u) << 2) | ((row >> 3) & 3u); }

template <int MT, int NT, bool USE_MMA, bool FUSED>
DEVINL void mine_body(const i8 *A, const i8 *U, const u8 *P, u32 nlo, u32 nhi, const i8 *Bt, u32 m, u32 n,
                      u32 k, const u32 *seed8, const u32 *bound8, u32 *cand, u32 cand_cap, u32 *dbg) {
    constexpr int WARPS_N = BN / (8 * NT);
    DYN_SMEM(smem);
    const u32 tid = TID_X, lane = tid & 31, warp = tid >> 5;
    const u32 g = lane >> 2, q = lane & 3;
    const u32 wm = warp / WARPS_N, wn = warp % WARPS_N;
    const u32 row0 = BID_Y * BM, col0 = BID_X * BN;
    const u32 wrow = wm * 16 * MT, wcol = wn * 8 * NT;   // within block

    i32 acc[MT][NT][4];
    u32 jp[MT][16];
#pragma unroll
    for (int a = 0; a < MT; a++) {
#pragma unroll
        for (int b = 0; b < NT; b++) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0;
#pragma unroll
        for (int j = 0; j < 16; j++) jp[a][j] = 0;
    }

    const u32 nk = k / BK;
    u32 *ELt = (u32 *)(smem + STAGES * STAGE_BYTES);   // FUSED: ELt[f*32 + rg] = U[4rg+j][f], j = byte
    const u32 *P32 = (const u32 *)P;
    u32 pw0 = 0, pw1 = 0, gen_kt = 0;   // FUSED: pairs of the next k-step to generate
    if constexpr (FUSED) {
        const u32 fb = warp * 16;
        u32 x[4][4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const u32 *src = (const u32 *)(U + (u64)(row0 + 4 * lane + j) * 128 + fb);
#pragma unroll
            for (int w = 0; w < 4; w++) x[j][w] = src[w];
        }
#pragma unroll
        for (int w = 0; w < 4; w++) {
            u32 y[4];
            transpose4(x[0][w], x[1][w], x[2][w], x[3][w], y);
#pragma unroll
            for (int i = 0; i < 4; i++) ELt[(fb + 4 * w + i) * 32 + lane] = y[i];
        }
        pw0 = P32[2 * warp];
        pw1 = P32[2 * warp + 1];
        BAR_SYNC();
    }
    // FUSED: warp w generates lines 4w..4w+3 of the k-step for rows 4*lane..4*lane+3
    auto gen_stage = [&](u32 st) {
        u8 *As = smem + st * STAGE_BYTES;
        u32 wv[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            u32 pw = i < 2 ? pw0 : pw1;
            u32 f = (pw >> (16 * (i & 1))) & 0xffu, sc = (pw >> (16 * (i & 1) + 8)) & 0xffu;
            wv[i] = bsub4(ELt[f * 32 + lane], ELt[sc * 32 + lane]);
        }
        u32 r[4];
        transpose4(wv[0], wv[1], wv[2], wv[3], r);
        if (row0 == 0 && gen_kt == 0 && lane == 0 && warp < 4) {   // nonce nibbles 4w..4w+3 of row 0
            u32 v = ((warp < 2 ? nlo : nhi) >> (16 * (warp & 1))) & 0xffffu;
            u32 nw = (v & 15u) | ((v >> 4) & 15u) << 8 | ((v >> 8) & 15u) << 16 | ((v >> 12) & 15u) << 24;
            r[0] = badd4(r[0], nw);
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            u32 row = 4 * lane + j;
            ((u32 *)(As + row * 32))[warp ^ a_swz(row)] = r[j];
        }
        gen_kt++;
        if (gen_kt < nk) {
            pw0 = P32[gen_kt * 16 + 2 * warp];
            pw1 = P32[gen_kt * 16 + 2 * warp + 1];
        }
    };
    auto load_stage = [&](u32 st, u32 kt) {
        u8 *As = smem + st * STAGE_BYTES, *Bs = As + BM * SROW;
        if constexpr (FUSED) {
            gen_stage(st);
        } else {   // A: 128 rows x 2 segments
            u32 r = tid >> 1, s = tid & 1;
            cp_async16(As + r * SROW + 16 * s, A + (u64)(row0 + r) * k + kt * BK + 16 * s);
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {   // B: 256 rows x 2 segments
            u32 r = (tid >> 1) + 128 * i, s = tid & 1;
            cp_async16(Bs + r * SROW + 16 * s, Bt + (u64)(col0 + r) * k + kt * BK + 16 * s);
        }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if ((u32)s < nk) load_stage(s, s);
        cp_async_commit();
    }

    for (u32 kt = 0; kt < nk; kt++) {
        cp_async_wait<STAGES - 2>();
        BAR_SYNC();
        if (kt + STAGES - 1 < nk) load_stage((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        cp_async_commit();

        const u8 *As = smem + (kt % STAGES) * STAGE_BYTES, *Bs = As + BM * SROW;
        if constexpr (USE_MMA) {
            u32 af[MT][4];
#pragma unroll
            for (int a = 0; a < MT; a++) {
                if constexpr (FUSED) {
                    const u32 r0 = wrow + 16 * a + g, r8 = r0 + 8;
                    const u32 *p0 = (const u32 *)(As + r0 * 32), *p8 = (const u32 *)(As + r8 * 32);
                    af[a][0] = p0[q ^ a_swz(r0)];
                    af[a][1] = p8[q ^ a_swz(r8)];
                    af[a][2] = p0[(q + 4) ^ a_swz(r0)];
                    af[a][3] = p8[(q + 4) ^ a_swz(r8)];
                } else {
                    const u8 *ra = As + (wrow + 16 * a + g) * SROW + 4 * q;
                    af[a][0] = *(const u32 *)(ra);
                    af[a][1] = *(const u32 *)(ra + 8 * SROW);
                    af[a][2] = *(const u32 *)(ra + 16);
                    af[a][3] = *(const u32 *)(ra + 8 * SROW + 16);
                }
            }
#pragma unroll
            for (int b = 0; b < NT; b++) {
                const u8 *rb = Bs + (wcol + 8 * b + g) * SROW + 4 * q;
                u32 bf[2] = {*(const u32 *)(rb), *(const u32 *)(rb + 16)};
#pragma unroll
                for (int a = 0; a < MT; a++) mma_s8(acc[a][b], af[a], bf);
            }
        } else {
            // dp4a path: same accumulator layout, computed directly per lane
#pragma unroll
            for (int a = 0; a < MT; a++) {
                u32 r0w[8], r8w[8];
                const u32 *p0 = (const u32 *)(As + (wrow + 16 * a + g) * SROW);
                const u32 *p8 = (const u32 *)(As + (wrow + 16 * a + g + 8) * SROW);
#pragma unroll
                for (int i = 0; i < 8; i++) { r0w[i] = p0[i]; r8w[i] = p8[i]; }
#pragma unroll
                for (int b = 0; b < NT; b++) {
                    const u32 *c0 = (const u32 *)(Bs + (wcol + 8 * b + 2 * q) * SROW);
                    const u32 *c1 = (const u32 *)(Bs + (wcol + 8 * b + 2 * q + 1) * SROW);
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        u32 x = c0[i], y = c1[i];
                        acc[a][b][0] = dp4a(r0w[i], x, acc[a][b][0]);
                        acc[a][b][1] = dp4a(r0w[i], y, acc[a][b][1]);
                        acc[a][b][2] = dp4a(r8w[i], x, acc[a][b][2]);
                        acc[a][b][3] = dp4a(r8w[i], y, acc[a][b][3]);
                    }
                }
            }
        }

        if ((kt & 3) == 3) {   // end of a 128-wide k chunk: fold this lane's tiles into the transcript
#pragma unroll
            for (int a = 0; a < MT; a++) {
                u32 x = 0;
#pragma unroll
                for (int b = 0; b < NT; b++)
                    x ^= (u32)acc[a][b][0] ^ (u32)acc[a][b][1] ^ (u32)acc[a][b][2] ^ (u32)acc[a][b][3];
                // jp[a] holds transcript words rotated so that jp[a][0] is the current slot
                u32 head = rotl32(jp[a][0], 13) ^ x;
#pragma unroll
                for (int j = 0; j < 15; j++) jp[a][j] = jp[a][j + 1];
                jp[a][15] = head;
            }
        }
    }
    // nk/4 is a multiple of 16 (k % 2048 == 0), so jp[a][j] is transcript word j.

    u32 key[8], bound[8];
#pragma unroll
    for (int i = 0; i < 8; i++) { key[i] = seed8[i]; bound[i] = bound8[i]; }
    const u32 tiles_c = n / (2 * NT);
#pragma unroll
    for (int a = 0; a < MT; a++) {
        u32 t_rows = row0 + wrow + 16 * a + g;
        u32 t_cols = col0 + wcol + 2 * q;
        if (dbg) {
            u32 rid = (t_rows / 16) * 8 + (t_rows % 16);
            u32 cid = (t_cols / (8 * NT)) * 4 + (t_cols % (8 * NT)) / 2;
            u32 *d = dbg + ((u64)rid * tiles_c + cid) * 16;
#pragma unroll
            for (int j = 0; j < 16; j++) d[j] = jp[a][j];
        }
        u32 h[8];
        b3_keyed64(key, jp[a], h);
        // little-endian u256 compare: h <= bound
        int le = 1;   // equal so far
        bool ok = false;
#pragma unroll
        for (int i = 7; i >= 0; i--) {
            if (le) {
                if (h[i] < bound[i]) { ok = true; le = 0; }
                else if (h[i] > bound[i]) { ok = false; le = 0; }
            }
        }
        if (le) ok = true;
        if (ok) {
            u32 slot = atomic_add_u32(cand, 1u);
            if (slot < cand_cap) {
                u32 *c = cand + 4 + NCAND_WORDS * slot;
                c[0] = t_rows; c[1] = t_cols; c[2] = h[7]; c[3] = h[6];
            }
        }
    }
}

// ===================================================================== v2 mining kernel (fused only)
// Same hash-tile geometry and results as mine_body<..., FUSED=true>, restructured for throughput:
//  * 32-byte smem rows (A and B) with a 16-byte-chunk XOR swizzle: conflict-free ldmatrix,
//    cp.async and noise-generation stores, no padding;
//  * ldmatrix.x4 fragment loads (A: 1 per m16 sub-tile, B: 1 per pair of n8 sub-tiles);
//  * the transcript fold of chunk c is interleaved with the first mma's of chunk c+1
//    (no tensor-pipe drain every 128 k);
//  * JPS: transcript words in shared memory (frees 16*MT registers);
//  * BKS k32-slices per pipeline stage (fewer barriers), STG stages;
//  * without cp.async (Turing) the B tile is prefetched through registers one stage ahead.
#define V2_SLICE_A (BM * 32)
#define V2_SLICE_B (BN * 32)
#define V2_SLICE (V2_SLICE_A + V2_SLICE_B)
#if defined(PRL_EMU)
#if defined(PRL_TURING)
#define V2_ASYNC 0
#else
#define V2_ASYNC 1
#endif
#elif PRL_MMA && !PRL_TURING
#define V2_ASYNC 1
#else
#define V2_ASYNC 0
#endif
#if defined(PRL_EMU)
DEVINL void ldg128(u32 r[4], const void *p) { __builtin_memcpy(r, p, 16); }
DEVINL void sts128(void *p, const u32 r[4]) { __builtin_memcpy(p, r, 16); }
#else
DEVINL void ldg128(u32 r[4], const void *p) {
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "l"(p));
}
DEVINL void sts128(void *p, const u32 r[4]) {
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(smem_addr(p)), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3])
                 : "memory");
}
#endif

// byte offset of 16-byte chunk c (0/1) of row r. Groups of 4 rows form 8 chunks (128 bytes), XOR-swizzled
// by the low 2 bits of the group index: conflict-free ldmatrix / cp.async, 4-way generation stores, and
// linear in steps of 16 rows (+512 bytes), so per-sub-tile addresses become instruction immediates.
DEVINL u32 v2_off(u32 r, u32 c) { return (r >> 2) * 128u + 16u * ((2u * (r & 3u) + c) ^ ((r >> 2) & 3u)); }

// DIAG (benchmark only, results are wrong): 1 no noise generation, 2 no B loads, 4 no barriers, 8 no fold
template <int MT, int NT, int BKS, int STG, int JSM, bool PF, int DIAG = 0>
DEVINL void mine_v2(const i8 *U, const u8 *P, u32 nlo, u32 nhi, const i8 *Bt, u32 m, u32 n, u32 k,
                    const u32 *seed8, const u32 *bound8, u32 *cand, u32 cand_cap, u32 *dbg) {
    constexpr int WARPS_N = BN / (8 * NT);
    constexpr int NWARP = (BM / (16 * MT)) * WARPS_N, NTH = 32 * NWARP, CPI = 512 / NTH;   // threads, B chunks/thread
    constexpr int JS = JSM > MT ? MT : JSM;   // sub-tiles 0..JS-1 keep transcript words in smem
    constexpr u32 STAGE = BKS * V2_SLICE;
    constexpr u32 ELT = STG * STAGE, JPO = ELT + 128 * 128;   // byte offsets of ELt and the transcript words
    DYN_SMEM(smem);
    const saddr_t sb = sbase(smem);
    const u32 tid = TID_X, lane = tid & 31, warp = tid >> 5;
    const u32 g = lane >> 2, q = lane & 3;
    const u32 wm = warp / WARPS_N, wn = warp % WARPS_N;
    const u32 row0 = BID_Y * BM, col0 = BID_X * BN;
    const u32 wrow = wm * 16 * MT, wcol = wn * 8 * NT;
    // per-thread constant offsets (everything else is an immediate)
    const u32 a_ld = v2_off(wrow + (lane & 7) + 8 * ((lane >> 3) & 1), lane >> 4);                 // + 512*a
    const u32 b_ld = V2_SLICE_A + v2_off(wcol + 8 * (lane >> 4) + (lane & 7), (lane >> 3) & 1);    // + 512*bp
    const u32 g_st = v2_off(lane, warp >> 2) + 4 * (warp & 3);                                      // + 1024*j
    const u32 c_st = V2_SLICE_A + v2_off(tid >> 1, tid & 1);                                        // + 16*NTH*i
    const i8 *bsrc = Bt + (u64)(col0 + (tid >> 1)) * k + 16 * (tid & 1);                            // + 128k*i
    const saddr_t elt = sb + ELT + 4 * lane, jpa = sb + JPO + 4 * tid;

    i32 acc[MT][NT][4];
    u32 jpr[MT][16];
#pragma unroll
    for (int a = 0; a < MT; a++) {
#pragma unroll
        for (int b = 0; b < NT; b++) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            if (a < JS) sts32(jpa + 4 * NTH * (j * JS + a), 0);
            else jpr[a][j] = 0;
        }
    }
    const u32 *P32 = (const u32 *)P;
    u32 pw0, pw1, gen_kt = 0;
    if (warp < 8) {   // stage this block's 128 uniform rows transposed: word (f, rg) = bytes U[rg + 32j][f]
        const u32 fb = warp * 16;
        u32 x[4][4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const u32 *src = (const u32 *)(U + (u64)(row0 + lane + 32 * j) * 128 + fb);
#pragma unroll
            for (int w = 0; w < 4; w++) x[j][w] = src[w];
        }
#pragma unroll
        for (int w = 0; w < 4; w++) {
            u32 y[4];
            transpose4(x[0][w], x[1][w], x[2][w], x[3][w], y);
#pragma unroll
            for (int i = 0; i < 4; i++) sts32(elt + 128 * (fb + 4 * w + i), y[i]);
        }
        pw0 = P32[2 * warp];
        pw1 = P32[2 * warp + 1];
    }
    BAR_SYNC();
    const u32 nk = k / 32, ns = nk / BKS;

    // warp w generates lines 4w..4w+3 of a k32 slice for rows lane + 32j
    auto gen_slice = [&](saddr_t As) {
        u32 wv[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            u32 pw = i < 2 ? pw0 : pw1;
            u32 f = (pw >> (16 * (i & 1))) & 0xffu, sc = (pw >> (16 * (i & 1) + 8)) & 0xffu;
            wv[i] = bsub4(lds32(elt + 128 * f), lds32(elt + 128 * sc));
        }
        u32 r[4];
        transpose4(wv[0], wv[1], wv[2], wv[3], r);
        if (row0 == 0 && gen_kt == 0 && lane == 0 && warp < 4) {   // nonce nibbles 4w..4w+3 of row 0
            u32 v = ((warp < 2 ? nlo : nhi) >> (16 * (warp & 1))) & 0xffffu;
            u32 nw = (v & 15u) | ((v >> 4) & 15u) << 8 | ((v >> 8) & 15u) << 16 | ((v >> 12) & 15u) << 24;
            r[0] = badd4(r[0], nw);
        }
#pragma unroll
        for (int j = 0; j < 4; j++) sts32(As + g_st + 1024 * j, r[j]);
        gen_kt++;
        if (gen_kt < nk) {
            pw0 = P32[gen_kt * 16 + 2 * warp];
            pw1 = P32[gen_kt * 16 + 2 * warp + 1];
        }
    };
#if !V2_ASYNC
    u32 breg[BKS][CPI][4];
#endif
    auto issue = [&](u32 st, u32 s) {
#pragma unroll
        for (int j = 0; j < BKS; j++) {
            const saddr_t As = sb + st * STAGE + j * V2_SLICE;
            const u32 ks = s * BKS + j;
#pragma unroll
            for (int i = 0; i < CPI; i++) {
                const i8 *src = bsrc + (u64)(NTH / 2) * i * k + ks * 32;
                if constexpr (DIAG & 2) continue;
#if V2_ASYNC
                cp_async16s(As + c_st + 16 * NTH * i, src);
#else
                if constexpr (PF) ldg128(breg[j][i], src);
                else {   // synchronous copy
                    u32 t[4];
                    ldg128(t, src);
                    sts128s(As + c_st + 16 * NTH * i, t);
                }
#endif
            }
            if (!(DIAG & 1) && warp < 8) gen_slice(As);   // 8 warps generate (spread evenly over the 4 schedulers)
        }
    };
    auto commit = [&](u32 st) {
#if !V2_ASYNC
        if constexpr (PF) {
#pragma unroll
            for (int j = 0; j < BKS; j++)
#pragma unroll
                for (int i = 0; i < CPI; i++) sts128s(sb + st * STAGE + j * V2_SLICE + c_st + 16 * NTH * i, breg[j][i]);
        }
#endif
    };
    auto jp_update = [&](u32 chunk, const u32 *x) {
#pragma unroll
        for (int a = 0; a < MT; a++) {
            if (a < JS) {
                const saddr_t p = jpa + 4 * NTH * ((chunk & 15) * JS + a);
                sts32(p, rotl32(lds32(p), 13) ^ x[a]);
            } else {
                u32 head = rotl32(jpr[a][0], 13) ^ x[a];
#pragma unroll
                for (int j = 0; j < 15; j++) jpr[a][j] = jpr[a][j + 1];
                jpr[a][15] = head;
            }
        }
    };

#pragma unroll
    for (int s = 0; s < STG - 1; s++) {
        if ((u32)s < ns) { issue(s, s); commit(s); }
        cp_async_commit();
    }
    // Branch-free steady state: G stages (= one 128-wide chunk) per iteration, stage indices and fold
    // positions are compile-time, so the scheduler can interleave noise generation / copies with the mma's.
    // (The fold at k=0 is a no-op on all-zero state.)
    constexpr int G = 4 / BKS;
    auto step = [&](u32 s0, int g, bool do_issue) __attribute__((always_inline)) {
        const u32 s = s0 + g;
        if constexpr (!(DIAG & 4)) {
            cp_async_wait<STG - 2>();
            BAR_SYNC();
        }
        const u32 st_c = (G % STG == 0) ? (u32)(g % STG) : s % STG;
        const u32 st_i = (G % STG == 0) ? (u32)((g + STG - 1) % STG) : (s + STG - 1) % STG;
        if (do_issue) issue(st_i, s + STG - 1);
        cp_async_commit();
        const saddr_t S = sb + st_c * STAGE;
#pragma unroll
        for (int j = 0; j < BKS; j++) {
            const u32 ks = s * BKS + j;
            const saddr_t Sl = S + j * V2_SLICE;
            const bool fold = !(DIAG & 8) && ((g * BKS + j) & 3) == 0;
            u32 x[MT];
            u32 af[MT][4];
#pragma unroll
            for (int a = 0; a < MT; a++) {
                x[a] = 0;
                ldsm_x4(af[a], Sl + a_ld + 512 * a);
            }
#pragma unroll
            for (int bp = 0; bp < NT / 2; bp++) {
                u32 bf[4];
                ldsm_x4(bf, Sl + b_ld + 512 * bp);
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int b = 2 * bp + h;
                    const u32 bb[2] = {bf[2 * h], bf[2 * h + 1]};
#pragma unroll
                    for (int a = 0; a < MT; a++) {
                        if (fold) x[a] ^= (u32)acc[a][b][0] ^ (u32)acc[a][b][1] ^ (u32)acc[a][b][2] ^ (u32)acc[a][b][3];
                        mma_s8(acc[a][b], af[a], bb);
                    }
                }
            }
            if (fold) jp_update(ks / 4 - 1, x);
        }
        if (do_issue) commit(st_i);
    };
    const u32 ns_safe = ns >= (u32)(STG - 1) ? ns - (STG - 1) : 0;
    u32 s0 = 0;
    for (; s0 + G <= ns_safe; s0 += G) {
#pragma unroll
        for (int g = 0; g < G; g++) step(s0, g, true);
    }
    for (; s0 < ns; s0 += G) {
#pragma unroll
        for (int g = 0; g < G; g++) step(s0, g, s0 + g < ns_safe);
    }
    {   // last chunk
        u32 x[MT];
#pragma unroll
        for (int a = 0; a < MT; a++) {
            x[a] = 0;
#pragma unroll
            for (int b = 0; b < NT; b++)
                x[a] ^= (u32)acc[a][b][0] ^ (u32)acc[a][b][1] ^ (u32)acc[a][b][2] ^ (u32)acc[a][b][3];
        }
        jp_update(nk / 4 - 1, x);
    }

    u32 key[8], bound[8];
#pragma unroll
    for (int i = 0; i < 8; i++) { key[i] = seed8[i]; bound[i] = bound8[i]; }
    const u32 tiles_c = n / (2 * NT);
#pragma unroll
    for (int a = 0; a < MT; a++) {
        u32 words[16];
#pragma unroll
        for (int j = 0; j < 16; j++) {
            if (a < JS) words[j] = lds32(jpa + 4 * NTH * (j * JS + a));
            else words[j] = jpr[a][j];
        }
        u32 t_rows = row0 + wrow + 16 * a + g;
        u32 t_cols = col0 + wcol + 2 * q;
        if (dbg) {
            u32 rid = (t_rows / 16) * 8 + (t_rows % 16);
            u32 cid = (t_cols / (8 * NT)) * 4 + (t_cols % (8 * NT)) / 2;
            u32 *d = dbg + ((u64)rid * tiles_c + cid) * 16;
#pragma unroll
            for (int j = 0; j < 16; j++) d[j] = words[j];
        }
        u32 h[8];
        b3_keyed64(key, words, h);
        int le = 1;
        bool ok = false;
#pragma unroll
        for (int i = 7; i >= 0; i--) {
            if (le) {
                if (h[i] < bound[i]) { ok = true; le = 0; }
                else if (h[i] > bound[i]) { ok = false; le = 0; }
            }
        }
        if (le) ok = true;
        if (ok) {
            u32 slot = atomic_add_u32(cand, 1u);
            if (slot < cand_cap) {
                u32 *c = cand + 4 + NCAND_WORDS * slot;
                c[0] = t_rows; c[1] = t_cols; c[2] = h[7]; c[3] = h[6];
            }
        }
    }
}

// variants: BKS, STG, JSM (sub-tiles with transcript in smem, 9 = all), PF (Turing register prefetch)
// host computes smem = STG*BKS*V2_SLICE + 16384 + 16384*min(JSM, MT)*(threads/256)
#if V2_ASYNC
#define V2_A 1, 4, 9, false
#define V2_B 2, 2, 9, false
#define V2_C 1, 3, 0, false
#define V3_A 1, 3, 9, false
#define V3_B 2, 2, 9, false
#else
#define V2_A 1, 2, 1, true
#define V2_B 1, 2, 1, false
#define V2_C 1, 2, 0, true
#define V3_A 1, 2, 0, true
#define V3_B 1, 2, 0, false
#endif

#define MINE_ARGS const i8 *A, const i8 *Bt, u32 m, u32 n, u32 k, const u32 *seed8, const u32 *bound8, \
                  u32 *cand, u32 cand_cap, u32 *dbg
#define MINE_PASS A, nullptr, nullptr, 0u, 0u, Bt, m, n, k, seed8, bound8, cand, cand_cap, dbg
#define FMINE_ARGS const i8 *U, const u8 *P, u32 nlo, u32 nhi, const i8 *Bt, u32 m, u32 n, u32 k, \
                   const u32 *seed8, const u32 *bound8, u32 *cand, u32 cand_cap, u32 *dbg
#define FMINE_PASS nullptr, U, P, nlo, nhi, Bt, m, n, k, seed8, bound8, cand, cand_cap, dbg

#if defined(PRL_EMU) || (PRL_MMA && defined(K_MMA32))
KERNEL LAUNCH_BOUNDS(256, 1) void k_mine_mma_nt32(MINE_ARGS) { mine_body<1, 32, true, false>(MINE_PASS); }
#endif
#if defined(PRL_EMU) || (PRL_MMA && defined(K_MMA16))
KERNEL LAUNCH_BOUNDS(256, 1) void k_mine_mma_nt16(MINE_ARGS) { mine_body<2, 16, true, false>(MINE_PASS); }
#endif
#if defined(PRL_EMU) || (PRL_MMA && defined(K_MMA8))
KERNEL LAUNCH_BOUNDS(256, 1) void k_mine_mma_nt8(MINE_ARGS)  { mine_body<4, 8, true, false>(MINE_PASS); }
#endif
#if defined(PRL_EMU) || (PRL_MMA && defined(K_FMMA32))
KERNEL LAUNCH_BOUNDS(256, 1) void k_mine_fmma_nt32(FMINE_ARGS) { mine_body<1, 32, true, true>(FMINE_PASS); }
#endif
#if defined(PRL_EMU) || (PRL_MMA && defined(K_FMMA16))
KERNEL LAUNCH_BOUNDS(256, 1) void k_mine_fmma_nt16(FMINE_ARGS) { mine_body<2, 16, true, true>(FMINE_PASS); }
#endif
#if defined(PRL_EMU) || (PRL_MMA && defined(K_FMMA8))
KERNEL LAUNCH_BOUNDS(256, 1) void k_mine_fmma_nt8(FMINE_ARGS)  { mine_body<4, 8, true, true>(FMINE_PASS); }
#endif
#if defined(PRL_EMU) || defined(K_DP4A32)
KERNEL LAUNCH_BOUNDS(256, 1) void k_mine_dp4a_nt32(MINE_ARGS) { mine_body<1, 32, false, false>(MINE_PASS); }
#endif
#if defined(PRL_EMU) || defined(K_DP4A16)
KERNEL LAUNCH_BOUNDS(256, 1) void k_mine_dp4a_nt16(MINE_ARGS) { mine_body<2, 16, false, false>(MINE_PASS); }
#endif
#if defined(PRL_EMU) || defined(K_DP4A8)
KERNEL LAUNCH_BOUNDS(256, 1) void k_mine_dp4a_nt8(MINE_ARGS)  { mine_body<4, 8, false, false>(MINE_PASS); }
#endif
#define V2_PASS U, P, nlo, nhi, Bt, m, n, k, seed8, bound8, cand, cand_cap, dbg
#if defined(PRL_EMU) || (PRL_MMA && defined(K_V2_32))
KERNEL LAUNCH_BOUNDS(256, 1) void k_v2a_nt32(FMINE_ARGS) { mine_v2<1, 32, V2_A>(V2_PASS); }
KERNEL LAUNCH_BOUNDS(256, 1) void k_v2b_nt32(FMINE_ARGS) { mine_v2<1, 32, V2_B>(V2_PASS); }
KERNEL LAUNCH_BOUNDS(256, 1) void k_v2c_nt32(FMINE_ARGS) { mine_v2<1, 32, V2_C>(V2_PASS); }
#endif
#if defined(PRL_EMU) || (PRL_MMA && defined(K_V2_16))
// v3: 512 threads (16 warps, 4 per scheduler), warp tile 16x128, <= 128 registers
KERNEL LAUNCH_BOUNDS(512, 1) void k_v3a_nt16(FMINE_ARGS) { mine_v2<1, 16, V3_A>(V2_PASS); }
KERNEL LAUNCH_BOUNDS(512, 1) void k_v3b_nt16(FMINE_ARGS) { mine_v2<1, 16, V3_B>(V2_PASS); }
KERNEL LAUNCH_BOUNDS(256, 1) void k_v2a_nt16(FMINE_ARGS) { mine_v2<2, 16, V2_A>(V2_PASS); }
// --bench-mma breakdown of v2b (wrong results by design)
KERNEL LAUNCH_BOUNDS(256, 1) void k_diag1_nt16(FMINE_ARGS) { mine_v2<2, 16, V2_B, 1>(V2_PASS); }
KERNEL LAUNCH_BOUNDS(256, 1) void k_diag2_nt16(FMINE_ARGS) { mine_v2<2, 16, V2_B, 2>(V2_PASS); }
KERNEL LAUNCH_BOUNDS(256, 1) void k_diag3_nt16(FMINE_ARGS) { mine_v2<2, 16, V2_B, 3>(V2_PASS); }
KERNEL LAUNCH_BOUNDS(256, 1) void k_diag7_nt16(FMINE_ARGS) { mine_v2<2, 16, V2_B, 7>(V2_PASS); }
KERNEL LAUNCH_BOUNDS(256, 1) void k_diag15_nt16(FMINE_ARGS) { mine_v2<2, 16, V2_B, 15>(V2_PASS); }
KERNEL LAUNCH_BOUNDS(256, 1) void k_diag8_nt16(FMINE_ARGS) { mine_v2<2, 16, V2_B, 8>(V2_PASS); }
KERNEL LAUNCH_BOUNDS(256, 1) void k_v2b_nt16(FMINE_ARGS) { mine_v2<2, 16, V2_B>(V2_PASS); }
KERNEL LAUNCH_BOUNDS(256, 1) void k_v2c_nt16(FMINE_ARGS) { mine_v2<2, 16, V2_C>(V2_PASS); }
#endif
