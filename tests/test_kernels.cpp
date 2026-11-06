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

// ELt layout of uniform rows (see k_noise_uniform label bit 2 / mine_v2 EG)
static std::vector<i8> to_elt(const std::vector<i8> &U, u32 m) {
    std::vector<i8> E(m * 128);
    for (u32 r = 0; r < m; r++)
        for (u32 f = 0; f < 128; f++) {
            u32 b = r >> 7, rl = r & 127;
            E[(size_t)b * 16384 + (f * 32 + (rl & 31)) * 4 + (rl >> 5)] = U[(size_t)r * 128 + f];
        }
    return E;
}

static uint32_t rng_state = 99;
static int8_t rnd_val() { rng_state = rng_state * 1664525u + 1013904223u; return (int8_t)((int)((rng_state >> 16) % 129) - 64); }

typedef void (*mine_fn)(const i8 *, const i8 *, u32, u32, u32, const u32 *, const u32 *, u32 *, u32, u32 *);
typedef void (*fmine_fn)(const i8 *, const u8 *, u32, u32, const i8 *, u32, u32, u32, const u32 *, const u32 *, u32 *,
                         u32, u32 *);

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
        {   // ELt layout output (label bit 2) must equal the transposition of the plain rows
            const u32 rows2 = 256, nb2 = rows2 * 4;
            std::vector<i8> plain(rows2 * 128), elt(rows2 * 128);
            emu_launch_serial((nb2 + 255) / 256, 1, 256, [&]() { k_noise_uniform(kw, lb, nb2, plain.data()); });
            emu_launch_serial((nb2 + 255) / 256, 1, 256, [&]() { k_noise_uniform(kw, lb | 2, nb2, elt.data()); });
            CHECK(to_elt(plain, rows2) == elt, "uniform ELt layout label %d", lb);
        }
        std::vector<u8> pairs(2 * k), rp(2 * k);
        emu_launch((k / 8 + 255) / 256, 1, 256, 0, [&]() { k_noise_perm(kw, lb, k, pairs.data()); });
        prl_noise_perm(lb ? PRL_LABEL_B : PRL_LABEL_A, key, k, rp.data());
        CHECK(pairs == rp, "perm label %d", lb);
        // apply with nonce patch
        std::vector<i8> base(rows * k), out(rows * k);
        for (auto &v : base) v = rnd_val();
        uint64_t nonce = 0xfedcba9876543210ULL;
        emu_launch((rows + APPLY_ROWS - 1) / APPLY_ROWS, 1, 256, 2 * k + APPLY_ROWS * 128, [&]() {
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

static void test_mine(const char *name, mine_fn fn, int nt, u32 m, u32 n, u32 k, fmine_fn ffn = nullptr, u32 fsmem = 0,
                      u32 threads = NTHREADS, bool l2 = false, bool eg = false) {
    std::vector<i8> A(m * k), Bt(n * k), U(m * 128);
    std::vector<u8> P(2 * k);
    const uint64_t nonce = 0x0123456789abcdefULL ^ ((uint64_t)m << 40);
    for (auto &v : A) v = (i8)(rnd_val() * 2 - (rnd_val() & 1));   // full noised range
    if (ffn) {   // fused: A' = nonce patch (row 0) + U[r][first] - U[r][second], base A = 0
        for (auto &v : U) v = (i8)((rnd_val() & 63) - 32);
        for (u32 l = 0; l < k; l++) {
            u32 f = (u32)(rnd_val() & 127), sc = f ^ (1 + (u32)(rnd_val() & 63) % 127);
            P[2 * l] = (u8)f; P[2 * l + 1] = (u8)(sc & 127);
        }
        std::vector<i8> nz(k);
        for (u32 r = 0; r < m; r++) {
            std::vector<i8> b(k, 0);
            if (r == 0) prl_apply_nonce((int8_t *)b.data(), nonce);
            prl_noise_row((const int8_t *)&U[r * 128], P.data(), k, (int8_t *)nz.data());
            for (u32 l = 0; l < k; l++) A[r * k + l] = (i8)(b[l] + nz[l]);
        }
    }
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
    std::vector<i8> UE = to_elt(U, m);
    if (ffn)
        emu_launch(n / BN, m / BM, threads, fsmem ? fsmem : STAGES * STAGE_BYTES + FUSED_SMEM, [&]() {
            ffn(l2 ? A.data() : (eg ? UE.data() : U.data()), P.data(), (u32)nonce, (u32)(nonce >> 32), Bt.data(), m, n, k, sw, bound.w, cand.data(), 64,
                dbg.data());
        });
    else
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

static void test_seeds() {
    const size_t m = 128, k = 2048;
    std::vector<uint8_t> A(m * k, 0);
    uint8_t key[32], bs[32];
    for (int i = 0; i < 32; i++) { key[i] = (uint8_t)(i * 7 + 3); bs[i] = (uint8_t)(200 - i); }
    prl_tree t;
    prl_tree_build(&t, A.data(), A.size(), key, 1);
    for (int salted = 0; salted < 2; salted++) {
        std::vector<u32> ctx(PRL_SEEDCTX_WORDS), out(8 * 37);
        CHECK(prl_seed_ctx(&t, A.data(), bs, (uint32_t)m, salted, ctx.data()) == 0, "seed ctx");
        const uint64_t base = 0x00000005fffffff0ULL;   // crosses a 32-bit boundary inside the batch
        emu_launch_serial(1, 1, 256, [&]() { k_seeds(ctx.data(), (u32)base, (u32)(base >> 32), 37, out.data()); });
        int bad = 0;
        for (int z = 0; z < 37; z++) {
            uint8_t c0[1024];
            memcpy(c0, A.data(), 1024);
            prl_apply_nonce((int8_t *)c0, base + z);
            prl_path0 p;
            prl_tree_path0(&t, c0, &p);
            uint8_t sd[32];
            prl_seed_a(bs, p.node[p.nlayers - 1], (uint32_t)m, salted, sd);
            bad += memcmp(sd, &out[8 * z], 32) != 0;
        }
        CHECK(bad == 0, "k_seeds salted=%d: %d of 37 seeds differ", salted, bad);
    }
    prl_tree_free(&t);
    printf("seed kernel ok (37 nonces, salted + legacy)\n");
}

// batch of Z nonces in one launch (grid z): candidate set (t_rows, t_cols, z) must match the CPU
static void test_batch(const char *name, fmine_fn ffn, int nt, u32 m, u32 n, u32 k, u32 smem, u32 threads, bool l2) {
    const u32 Z = 3;
    const uint64_t nonce0 = 0x1122334455667788ULL;
    std::vector<i8> U(Z * m * 128), Ap(Z * m * k), Bt(n * k);
    std::vector<u8> P(Z * 2 * k);
    for (auto &v : U) v = (i8)((rnd_val() & 63) - 32);
    for (auto &v : Bt) v = (i8)(rnd_val() * 2 - (rnd_val() & 1));
    for (u32 l = 0; l < Z * k; l++) {
        u32 f = (u32)(rnd_val() & 127), sc = f ^ (1 + (u32)(rnd_val() & 63) % 127);
        P[2 * l] = (u8)f; P[2 * l + 1] = (u8)(sc & 127);
    }
    uint8_t seeds[3][32];
    std::vector<u32> sw(8 * Z);
    for (u32 z = 0; z < Z; z++) {
        for (int i = 0; i < 32; i++) seeds[z][i] = (uint8_t)(rnd_val() + 64 + z);
        key_to_words(seeds[z], &sw[8 * z]);
        std::vector<i8> nz(k);
        for (u32 r = 0; r < m; r++) {
            std::vector<i8> b(k, 0);
            if (r == 0) prl_apply_nonce((int8_t *)b.data(), nonce0 + z);
            prl_noise_row((const int8_t *)&U[(z * m + r) * 128], &P[z * 2 * k], k, (int8_t *)nz.data());
            for (u32 l = 0; l < k; l++) Ap[(size_t)z * m * k + (size_t)r * k + l] = (i8)(b[l] + nz[l]);
        }
    }
    prl_geom g;
    prl_geom_init(&g, m, n, k, nt);
    std::vector<std::pair<std::vector<uint8_t>, std::vector<u32>>> hs;
    for (u32 z = 0; z < Z; z++)
        for (u32 rb = 0; rb < m; rb += 16)
            for (u32 gg = 0; gg < 8; gg++)
                for (u32 cb = 0; cb < n; cb += 8 * nt)
                    for (u32 qq = 0; qq < 4; qq++) {
                        u32 tr = rb + gg, tc = cb + 2 * qq, rows[2], cols[64], jp[16];
                        prl_tile_rows(&g, tr, rows);
                        prl_tile_cols(&g, tc, cols);
                        const int8_t *ap[2] = {&Ap[(size_t)z * m * k + rows[0] * k], &Ap[(size_t)z * m * k + rows[1] * k]};
                        const int8_t *bp[64];
                        for (int j = 0; j < g.w; j++) bp[j] = &Bt[cols[j] * k];
                        prl_jackpot(ap, 2, bp, g.w, k, 128, jp);
                        uint8_t h[32];
                        prl_jackpot_hash(jp, seeds[z], h);
                        std::vector<uint8_t> be(h, h + 32);
                        std::reverse(be.begin(), be.end());
                        hs.push_back({be, {tr, tc, z}});
                    }
    std::sort(hs.begin(), hs.end());
    const int want = 9;
    std::vector<uint8_t> le(hs[want - 1].first.rbegin(), hs[want - 1].first.rend());
    u256 bound;
    u256_from_le_bytes(&bound, le.data());
    std::vector<u32> cand(4 + 4 * 64, 0);
    emu_launch3(n / BN, m / BM, Z, threads, smem, [&]() {
        ffn(l2 ? Ap.data() : U.data(), P.data(), (u32)nonce0, (u32)(nonce0 >> 32), Bt.data(), m, n, k, sw.data(),
            bound.w, cand.data(), 64, nullptr);
    });
    std::vector<std::vector<u32>> got, exp;
    for (u32 i = 0; i < cand[0] && i < 64; i++) got.push_back({cand[4 + 4 * i], cand[5 + 4 * i], cand[7 + 4 * i]});
    for (int i = 0; i < want; i++) exp.push_back(hs[i].second);
    std::sort(got.begin(), got.end());
    std::sort(exp.begin(), exp.end());
    CHECK(cand[0] == (u32)want && got == exp, "%s batch: candidate set differs (%u found)", name, cand[0]);
    printf("%-18s batch of %u nonces: %d candidates (rows, cols, nonce index) match CPU\n", name, Z, want);
}

int main() {
    test_batch("k_mine_fmma_nt16", k_mine_fmma_nt16, 16, 128, 256, 2048, STAGES * STAGE_BYTES + FUSED_SMEM, 256, false);
    test_batch("k_v4c_nt16", k_v4c_nt16, 16, 128, 256, 2048, 131072, 512, false);
    test_batch("k_l2b_nt16", k_l2b_nt16, 16, 128, 256, 2048, 131072, 256, true);
    test_seeds();
    test_noise();
    test_mine("k_mine_mma_nt32", k_mine_mma_nt32, 32, 128, 256, 2048);
    test_mine("k_mine_mma_nt16", k_mine_mma_nt16, 16, 128, 256, 2048);
    test_mine("k_mine_mma_nt8", k_mine_mma_nt8, 8, 128, 256, 2048);
    test_mine("k_mine_dp4a_nt32", k_mine_dp4a_nt32, 32, 128, 256, 2048);
    test_mine("k_mine_dp4a_nt16", k_mine_dp4a_nt16, 16, 128, 256, 2048);
    test_mine("k_mine_dp4a_nt8", k_mine_dp4a_nt8, 8, 128, 256, 2048);
    test_mine("k_mine_mma_nt32", k_mine_mma_nt32, 32, 256, 512, 4096);   // multi-block, 2 transcript cycles
    test_mine("k_mine_mma_nt8", k_mine_mma_nt8, 8, 256, 512, 2048);
    test_mine("k_mine_fmma_nt32", nullptr, 32, 128, 256, 2048, k_mine_fmma_nt32);
    test_mine("k_mine_fmma_nt16", nullptr, 16, 256, 256, 2048, k_mine_fmma_nt16);
    test_mine("k_mine_fmma_nt8", nullptr, 8, 128, 512, 4096, k_mine_fmma_nt8);
    {
        struct V { const char *name; fmine_fn f; int nt, mt, bks, stg, jsm, pf; } vs[] = {
#define VV(nm, f, nt, mt, ...) {nm, f, nt, mt, __VA_ARGS__}
            VV("k_v2a_nt16", k_v2a_nt16, 16, 2, V2_A), VV("k_v2b_nt16", k_v2b_nt16, 16, 2, V2_B),
            VV("k_v2c_nt16", k_v2c_nt16, 16, 2, V2_C), VV("k_v2a_nt32", k_v2a_nt32, 32, 1, V2_A),
            VV("k_v2b_nt32", k_v2b_nt32, 32, 1, V2_B), VV("k_v2c_nt32", k_v2c_nt32, 32, 1, V2_C)};
        for (auto &v : vs) {
            u32 sm = v.stg * v.bks * V2_SLICE + 16384 + 16384 * (v.jsm < v.mt ? v.jsm : v.mt);
            test_mine(v.name, nullptr, v.nt, 128, 256, 2048, v.f, sm);
        }
        test_mine("k_v3a_nt16", nullptr, 16, 128, 256, 2048, k_v3a_nt16, 131072, 512);
        test_mine("k_v3b_nt16", nullptr, 16, 128, 256, 2048, k_v3b_nt16, 131072, 512);
        test_mine("k_t2b_nt16", nullptr, 16, 256, 512, 4096, k_t2b_nt16, 131072, 256, false, true);
        test_mine("k_t2p_nt16", nullptr, 16, 128, 256, 2048, k_t2p_nt16, 131072, 256, false, true);
        test_mine("k_t3b_nt16", nullptr, 16, 256, 256, 2048, k_t3b_nt16, 131072, 512, false, true);
        test_mine("k_t4b_nt16", nullptr, 16, 256, 512, 4096, k_t4b_nt16, 131072, 512, false, true);
        test_mine("k_t4p_nt16", nullptr, 16, 128, 256, 2048, k_t4p_nt16, 131072, 512, false, true);
        test_mine("k_l2a_nt16", nullptr, 16, 128, 256, 2048, k_l2a_nt16, 131072, 256, true);
        test_mine("k_l2b_nt16", nullptr, 16, 256, 512, 4096, k_l2b_nt16, 131072, 256, true);
        test_mine("k_l3b_nt16", nullptr, 16, 256, 256, 2048, k_l3b_nt16, 131072, 512, true);
        test_mine("k_l4c_nt16", nullptr, 16, 256, 512, 4096, k_l4c_nt16, 131072, 512, true);
        test_mine("k_v4a_nt16", nullptr, 16, 128, 256, 2048, k_v4a_nt16, 131072, 512);
        test_mine("k_v4b_nt16", nullptr, 16, 128, 256, 2048, k_v4b_nt16, 131072, 512);
        test_mine("k_v4c_nt16", nullptr, 16, 128, 256, 2048, k_v4c_nt16, 131072, 512);
        test_mine("k_v4c_nt16 multi", nullptr, 16, 256, 512, 4096, k_v4c_nt16, 131072, 512);
        test_mine("k_v3a_nt16 multi", nullptr, 16, 256, 512, 4096, k_v3a_nt16, 131072, 512);
        test_mine("k_v2a_nt16 multi", nullptr, 16, 256, 512, 4096, k_v2a_nt16, 131072);
    }
    if (fails) { printf("%d FAILURES\n", fails); return 1; }
    printf("ALL KERNEL EMULATION TESTS PASSED\n");
    return 0;
}
