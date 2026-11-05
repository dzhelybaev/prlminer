// prlminer CUDA kernels: noise generation, noising, and the fused
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
#elif defined(PRL_EMU) || PRL_MMA
#define STAGES 3
#else
#define STAGES 2         // sm_61: 48 KB shared memory limit
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

// out[r][l] = base[r][l] + uni[r][first_l] - uni[r][second_l]; 16 elements per thread.
// patch != 0: base[0][0..15] is replaced by the 16 nonce nibbles.
KERNEL void k_apply_noise(const i8 *base, const i8 *uni, const u8 *pairs, i8 *out, u32 rows, u32 k,
                          u32 nonce_lo, u32 nonce_hi, u32 patch) {
    u64 gid = (u64)BID_Y * 65535ull * NTHREADS + (u64)BID_X * NTHREADS + TID_X;
    u32 segs = k / 16;
    if (gid >= (u64)rows * segs) return;
    u32 r = (u32)(gid / segs), s = (u32)(gid % segs);
    const i8 *ur = uni + (u64)r * 128;
    const u32 *src = (const u32 *)(base + (u64)r * k + 16 * s);
    const u32 *pp = (const u32 *)(pairs + 32 * s);   // 16 pairs = 32 bytes
    u32 *dst = (u32 *)(out + (u64)r * k + 16 * s);
    bool nonce = patch && r == 0 && s == 0;
#pragma unroll
    for (int w = 0; w < 4; w++) {
        u32 bw = src[w], p0 = pp[2 * w], p1 = pp[2 * w + 1], res = 0;
#pragma unroll
        for (int b = 0; b < 4; b++) {
            int e = 4 * w + b;
            u32 pw = b < 2 ? p0 : p1;
            int f = (pw >> (16 * (b & 1))) & 0xff, sec = (pw >> (16 * (b & 1) + 8)) & 0xff;
            int v = (int)(i8)(bw >> (8 * b));
            if (nonce) v = (int)(((e < 8 ? (nonce_lo >> (4 * e)) : (nonce_hi >> (4 * (e - 8))))) & 15u);
            v += (int)ur[f] - (int)ur[sec];
            res |= (u32)(u8)(i8)v << (8 * b);
        }
        dst[w] = res;
    }
}

#endif  // K_NOISE

// ------------------------------------------------------------------ fused mining kernel
// grid: (n/BN, m/BM), block 256. A: m x k (noised), Bt: n x k (noised).
// For every hash tile: transcript over k/128 chunks, H = B3_keyed(seed_a, transcript),
// H (LE u256) <= bound -> candidate. dbg (optional): all transcripts, 16 words per tile.
template <int MT, int NT, bool USE_MMA>
DEVINL void mine_body(const i8 *A, const i8 *Bt, u32 m, u32 n, u32 k, const u32 *seed8, const u32 *bound8,
                      u32 *cand, u32 cand_cap, u32 *dbg) {
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
    auto load_stage = [&](u32 st, u32 kt) {
        u8 *As = smem + st * STAGE_BYTES, *Bs = As + BM * SROW;
        {   // A: 128 rows x 2 segments
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
                const u8 *ra = As + (wrow + 16 * a + g) * SROW + 4 * q;
                af[a][0] = *(const u32 *)(ra);
                af[a][1] = *(const u32 *)(ra + 8 * SROW);
                af[a][2] = *(const u32 *)(ra + 16);
                af[a][3] = *(const u32 *)(ra + 8 * SROW + 16);
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

#define MINE_ARGS const i8 *A, const i8 *Bt, u32 m, u32 n, u32 k, const u32 *seed8, const u32 *bound8, \
                  u32 *cand, u32 cand_cap, u32 *dbg
#define MINE_PASS A, Bt, m, n, k, seed8, bound8, cand, cand_cap, dbg

#if defined(PRL_EMU) || (PRL_MMA && defined(K_MMA32))
KERNEL LAUNCH_BOUNDS(256, 1) void k_mine_mma_nt32(MINE_ARGS) { mine_body<1, 32, true>(MINE_PASS); }
#endif
#if defined(PRL_EMU) || (PRL_MMA && defined(K_MMA16))
KERNEL LAUNCH_BOUNDS(256, 1) void k_mine_mma_nt16(MINE_ARGS) { mine_body<2, 16, true>(MINE_PASS); }
#endif
#if defined(PRL_EMU) || (PRL_MMA && defined(K_MMA8))
KERNEL LAUNCH_BOUNDS(256, 1) void k_mine_mma_nt8(MINE_ARGS)  { mine_body<4, 8, true>(MINE_PASS); }
#endif
#if defined(PRL_EMU) || defined(K_DP4A32)
KERNEL LAUNCH_BOUNDS(256, 1) void k_mine_dp4a_nt32(MINE_ARGS) { mine_body<1, 32, false>(MINE_PASS); }
#endif
#if defined(PRL_EMU) || defined(K_DP4A16)
KERNEL LAUNCH_BOUNDS(256, 1) void k_mine_dp4a_nt16(MINE_ARGS) { mine_body<2, 16, false>(MINE_PASS); }
#endif
#if defined(PRL_EMU) || defined(K_DP4A8)
KERNEL LAUNCH_BOUNDS(256, 1) void k_mine_dp4a_nt8(MINE_ARGS)  { mine_body<4, 8, false>(MINE_PASS); }
#endif
