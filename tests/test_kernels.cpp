// Runs the real CUDA kernel source (cuda/kernels.cu) in the SIMT emulator and checks
// every output against the CPU protocol core (src/pearl.c), which is itself
// cross-checked against the official Pearl code.
#define PRL_EMU 1
#include "../cuda/kernels.cu"

extern "C" {
#include "../src/pearl.h"
}
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void key_to_words(const uint8_t k[32], u32 w[8]) {
    for (int i = 0; i < 8; i++) w[i] = k[4 * i] | (k[4 * i + 1] << 8) | (k[4 * i + 2] << 16) | ((u32)k[4 * i + 3] << 24);
}

static uint32_t rng_state = 99;
static int8_t rnd_val() { rng_state = rng_state * 1664525u + 1013904223u; return (int8_t)((int)((rng_state >> 16) % 129) - 64); }

typedef void (*mine_fn)(const i8 *, const i8 *, u32, u32, u32, const u32 *, const u32 *, u32 *, u32, u32 *);

static void run_mine(mine_fn fn, const i8 *A, const i8 *Bt, u32 m, u32 n, u32 k, const u32 *seed, const u32 *bound,
                     u32 *cand, u32 cap, u32 *dbg) {
    emu_launch(n / BN, m / BM, NTHREADS, STAGES * STAGE_BYTES, [&]() { fn(A, Bt, m, n, k, seed, bound, cand, cap, dbg); });
}

static void test_noise() {
    uint8_t key[32];
    for (int i = 0; i < 32; i++) key[i] = (uint8_t)(i * 37 + 1);
    u32 kw[8];
    key_to_words(key, kw);
    const u32 rows = 40, k = 2048;
    for (int lb = 0; lb < 2; lb++) {
        std::vector<i8> uni(rows * 128);
        u32 nb = rows * 4;
        emu_launch((nb + 255) / 256, 1, 256, 0, [&]() { k_noise_uniform(kw, lb, nb, uni.data()); });
        for (u32 r = 0; r < rows; r++) {
            int8_t ref[128];
            prl_noise_uniform_row(lb ? PRL_LABEL_B : PRL_LABEL_A, key, r, ref);
            CHECK(!memcmp(ref, &uni[r * 128], 128), "uniform row %u label %d", r, lb);
        }
        std::vector<u8> pairs(2 * k), rp(2 * k);
        emu_launch((k / 8 + 255) / 256, 1, 256, 0, [&]() { k_noise_perm(kw, lb, k, pairs.data()); });
        prl_noise_perm(lb ? PRL_LABEL_B : PRL_LABEL_A, key, k, rp.data());
        CHECK(pairs == rp, "perm label %d", lb);
        // apply with nonce patch
        std::vector<i8> base(rows * k), out(rows * k);
        for (auto &v : base) v = rnd_val();
        uint64_t nonce = 0xfedcba9876543210ULL;
        u32 total = rows * (k / 16);
        emu_launch((total + 255) / 256, 1, 256, 0, [&]() {
            k_apply_noise(base.data(), uni.data(), pairs.data(), out.data(), rows, k, (u32)nonce, (u32)(nonce >> 32), 1);
        });
        for (u32 r = 0; r < rows; r++) {
            std::vector<i8> b(base.begin() + r * k, base.begin() + (r + 1) * k), nz(k);
            if (r == 0) prl_apply_nonce(b.data(), nonce);
            prl_noise_row(&uni[r * 128], rp.data(), k, nz.data());
            bool ok = true;
            for (u32 l = 0; l < k; l++) ok &= (i8)(b[l] + nz[l]) == out[r * k + l];
            CHECK(ok, "apply row %u", r);
        }
    }
    printf("noise kernels ok\n");
}

static void test_mine(const char *name, mine_fn fn, int nt, u32 m, u32 n, u32 k) {
    std::vector<i8> A(m * k), Bt(n * k);
    for (auto &v : A) v = (i8)(rnd_val() * 2 - (rnd_val() & 1));   // full noised range
    for (auto &v : Bt) v = (i8)(rnd_val() * 2 - (rnd_val() & 1));
    uint8_t seed[32];
    for (int i = 0; i < 32; i++) seed[i] = (uint8_t)(rnd_val() + 64);
    u32 sw[8];
    key_to_words(seed, sw);
    const u32 tiles_r = m / 2, tiles_c = n / (2 * nt), ntiles = tiles_r * tiles_c;
    // CPU reference for all tiles
    std::vector<uint32_t> ref(ntiles * 16);
    std::vector<std::pair<std::vector<uint8_t>, std::pair<u32, u32>>> hashes;
    prl_geom g;
    prl_geom_init(&g, m, n, k, nt);
    for (u32 rb = 0; rb < m; rb += 16)
        for (u32 gg = 0; gg < 8; gg++)
            for (u32 cb = 0; cb < n; cb += 8 * nt)
                for (u32 qq = 0; qq < 4; qq++) {
                    u32 tr = rb + gg, tc = cb + 2 * qq;
                    uint32_t rows[2], cols[64];
                    prl_tile_rows(&g, tr, rows);
                    prl_tile_cols(&g, tc, cols);
                    const int8_t *ap[2] = {&A[rows[0] * k], &A[rows[1] * k]}, *bp[64];
                    for (int j = 0; j < g.w; j++) bp[j] = &Bt[cols[j] * k];
                    u32 rid = (tr / 16) * 8 + tr % 16, cid = (tc / (8 * nt)) * 4 + (tc % (8 * nt)) / 2;
                    prl_jackpot(ap, 2, bp, g.w, k, 128, &ref[(rid * tiles_c + cid) * 16]);
                    uint8_t h[32];
                    prl_jackpot_hash(&ref[(rid * tiles_c + cid) * 16], seed, h);
                    std::vector<uint8_t> hv(h, h + 32);
                    std::reverse(hv.begin(), hv.end());   // big-endian for sorting
                    hashes.push_back({hv, {tr, tc}});
                }
    std::sort(hashes.begin(), hashes.end());
    // bound = 7th smallest hash -> exactly 7 candidates
    const int want = 7;
    std::vector<uint8_t> be = hashes[want - 1].first, le(be.rbegin(), be.rend());
    u256 bound;
    u256_from_le_bytes(&bound, le.data());
    std::vector<u32> dbg(ntiles * 16, 0xdeadbeef), cand(4 + 4 * 64, 0);
    run_mine(fn, A.data(), Bt.data(), m, n, k, sw, bound.w, cand.data(), 64, dbg.data());
    int bad = 0;
    for (u32 i = 0; i < ntiles * 16; i++) bad += dbg[i] != ref[i];
    CHECK(bad == 0, "%s m=%u n=%u: %d transcript words differ", name, m, n, bad);
    CHECK(cand[0] == (u32)want, "%s: %u candidates, want %d", name, cand[0], want);
    std::vector<std::pair<u32, u32>> got, exp;
    for (u32 i = 0; i < cand[0] && i < 64; i++) got.push_back({cand[4 + 4 * i], cand[5 + 4 * i]});
    for (int i = 0; i < want; i++) exp.push_back(hashes[i].second);
    std::sort(got.begin(), got.end());
    std::sort(exp.begin(), exp.end());
    CHECK(got == exp, "%s: candidate set differs", name);
    printf("%-18s m=%u n=%u k=%u: %u tiles, transcripts + %d candidates match CPU\n", name, m, n, k, ntiles, want);
}

int main() {
    test_noise();
    test_mine("k_mine_mma_nt32", k_mine_mma_nt32, 32, 128, 256, 2048);
    test_mine("k_mine_mma_nt16", k_mine_mma_nt16, 16, 128, 256, 2048);
    test_mine("k_mine_mma_nt8", k_mine_mma_nt8, 8, 128, 256, 2048);
    test_mine("k_mine_dp4a_nt32", k_mine_dp4a_nt32, 32, 128, 256, 2048);
    test_mine("k_mine_dp4a_nt16", k_mine_dp4a_nt16, 16, 128, 256, 2048);
    test_mine("k_mine_dp4a_nt8", k_mine_dp4a_nt8, 8, 128, 256, 2048);
    test_mine("k_mine_mma_nt32", k_mine_mma_nt32, 32, 256, 512, 4096);   // multi-block, 2 transcript cycles
    test_mine("k_mine_mma_nt8", k_mine_mma_nt8, 8, 256, 512, 2048);
    if (fails) { printf("%d FAILURES\n", fails); return 1; }
    printf("ALL KERNEL EMULATION TESTS PASSED\n");
    return 0;
}
