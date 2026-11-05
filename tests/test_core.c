// Unit tests for the CPU protocol core.
#include "../src/pearl.h"
#include "b3_vectors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void hex(const uint8_t *b, int n, char *out) {
    for (int i = 0; i < n; i++) sprintf(out + 2 * i, "%02x", b[i]);
}

static void test_blake3(void) {
    const char *key = "whats the Elvish word for friend";
    uint8_t in[102400];
    for (int i = 0; i < 102400; i++) in[i] = (uint8_t)(i % 251);
    int ntree = 0;
    for (size_t c = 0; c < sizeof(B3V) / sizeof(B3V[0]); c++) {
        uint8_t out[32];
        char hx[65];
        prl_keyed_hash((const uint8_t *)key, in, B3V[c].len, out);
        hex(out, 32, hx);
        CHECK(!strcmp(hx, B3V[c].keyed), "keyed len %d", B3V[c].len);
        prl_hash(in, B3V[c].len, out);
        hex(out, 32, hx);
        CHECK(!strcmp(hx, B3V[c].hash), "hash len %d", B3V[c].len);
        if (B3V[c].len >= 2048 && B3V[c].len % 1024 == 0) {
            for (int th = 1; th <= 3; th += 2) {
                prl_tree t;
                CHECK(prl_tree_build(&t, in, B3V[c].len, (const uint8_t *)key, th) == 0, "tree build");
                hex(prl_tree_root(&t), 32, hx);
                CHECK(!strcmp(hx, B3V[c].keyed), "tree root len %d threads %d", B3V[c].len, th);
                // path0 with unchanged chunk 0 must reproduce the root
                prl_path0 p;
                prl_tree_path0(&t, in, &p);
                CHECK(!memcmp(p.node[t.nlayers - 1], prl_tree_root(&t), 32), "path0 root len %d", B3V[c].len);
                // path0 with modified chunk 0 must equal a full rebuild
                uint8_t *mod = malloc(B3V[c].len);
                memcpy(mod, in, B3V[c].len);
                mod[5] ^= 0x5a;
                prl_tree_path0(&t, mod, &p);
                prl_tree t2;
                prl_tree_build(&t2, mod, B3V[c].len, (const uint8_t *)key, 1);
                CHECK(!memcmp(p.node[t.nlayers - 1], prl_tree_root(&t2), 32), "path0 modified len %d", B3V[c].len);
                prl_tree_free(&t2);
                free(mod);
                prl_tree_free(&t);
                ntree++;
            }
        }
    }
    printf("blake3: %zu vectors, %d tree checks\n", sizeof(B3V) / sizeof(B3V[0]), ntree);
}

static void test_commitment_pinned(void) {
    // zk-pow api/seed.rs commitment_hash_pinned_vectors: m=192, n=320, job_key=0x11.., hash_a=0xAA.., hash_b=0xBB..
    uint8_t jk[32], ha[32], hb[32], b[32], a[32];
    memset(jk, 0x11, 32); memset(ha, 0xAA, 32); memset(hb, 0xBB, 32);
    const uint8_t LB[32] = {0xad,0xd6,0xf7,0xea,0x5f,0xee,0xbf,0x89,0xc8,0xa7,0x7e,0x2e,0xbf,0xa0,0xd8,0x24,0x42,0xe7,0xdb,0xb0,0x04,0x6d,0xbd,0x48,0x97,0x18,0x61,0xd1,0x2f,0xcb,0x01,0x77};
    const uint8_t LA[32] = {0x48,0x3b,0x07,0xb6,0xf7,0x31,0x05,0x03,0x0b,0x94,0x82,0x25,0x5f,0x37,0x72,0x3f,0x3f,0xed,0x69,0xae,0x91,0x67,0x24,0xee,0x82,0x91,0x84,0x8b,0x8c,0x28,0x79,0x4b};
    const uint8_t SB[32] = {0x60,0xed,0x9b,0x73,0xc5,0xa9,0x59,0x9b,0x20,0x0b,0x6c,0xd5,0x63,0xe7,0xf0,0xd5,0xd9,0xa6,0x7d,0x24,0x02,0xd8,0x5f,0xd4,0xef,0x96,0x6c,0x58,0x00,0x80,0xd0,0xe5};
    const uint8_t SA[32] = {0x30,0x17,0x84,0x16,0x80,0x05,0xec,0x83,0x3a,0xb0,0xaa,0x60,0x00,0x6f,0x7f,0xe7,0xfa,0xaa,0x95,0x30,0x7d,0x8c,0x1f,0xc6,0x81,0x9b,0x2f,0xfd,0xd7,0x17,0xec,0xcf};
    prl_seed_b(jk, hb, 320, false, b); prl_seed_a(b, ha, 192, false, a);
    CHECK(!memcmp(b, LB, 32) && !memcmp(a, LA, 32), "legacy seeds");
    prl_seed_b(jk, hb, 320, true, b); prl_seed_a(b, ha, 192, true, a);
    CHECK(!memcmp(b, SB, 32) && !memcmp(a, SA, 32), "salted seeds");
    // salts are blake3 of context strings
    uint8_t s[32];
    prl_hash("pearl/cert-v3/noise-seed/A", 26, s);
    CHECK(s[0] == 0x82 && s[31] == 0x7b, "salt A");
    printf("commitment pinned vectors ok\n");
}

static void test_patterns(void) {
    prl_geom g;
    for (int nt = 8; nt <= 32; nt *= 2) {
        CHECK(prl_geom_init(&g, 128, 256, 2048, nt) == 0, "geom nt=%d", nt);
        uint32_t lst[256];
        int n = prl_pattern_to_list(&g.cols, lst, 256);
        CHECK(n == 2 * nt, "cols size");
        for (int j = 0; j < nt; j++) CHECK(lst[2 * j] == 8u * j && lst[2 * j + 1] == 8u * j + 1, "cols list");
        CHECK(prl_pattern_period(&g.cols) == 8u * nt, "cols period");
        int valid = 0;
        for (uint32_t o = 0; o < 8u * nt; o++) valid += prl_pattern_offset_valid(&g.cols, o);
        CHECK(valid == 4, "4 col offsets per period (got %d)", valid);
        CHECK(prl_pattern_offset_valid(&g.cols, 2) && !prl_pattern_offset_valid(&g.cols, 1), "col offsets");
        CHECK(g.factor == (uint32_t)(2 * 2 * nt * 2048), "factor");
    }
    uint32_t r[2];
    int n = prl_pattern_to_list(&g.rows, r, 2);
    CHECK(n == 2 && r[0] == 0 && r[1] == 8, "rows list");
    int valid = 0;
    for (uint32_t o = 0; o < 16; o++) valid += prl_pattern_offset_valid(&g.rows, o);
    CHECK(valid == 8, "row offsets");
    // config bytes for the official 2x64 pattern
    prl_geom_init(&g, 128, 256, 8192, 32);
    const uint8_t want[20] = {0x00, 0x20, 0, 0, 128, 0, 0, 0, 7, 1, 0, 0, 0, 0, 0, 1, 3, 31, 0, 0};
    CHECK(!memcmp(g.config_bytes, want, 20), "config bytes");
    // pattern used in the official tests
    uint32_t tr[4] = {0, 8, 64, 72};
    prl_pattern p;
    CHECK(prl_pattern_from_list(&p, tr, 4) == 0, "test pattern");
    uint32_t tl[4];
    CHECK(prl_pattern_to_list(&p, tl, 4) == 4 && tl[2] == 64 && tl[3] == 72, "test pattern list");
    uint32_t bad[3] = {0, 1, 3};
    CHECK(prl_pattern_from_list(&p, bad, 3) != 0, "non periodic rejected");
    printf("patterns ok\n");
}

static void test_u256(void) {
    u256 a, b;
    u256_from_nbits(&a, 0x1d00ffff);
    char hx[65];
    u256_to_be_hex(&a, hx);
    CHECK(!strcmp(hx, "00000000ffff0000000000000000000000000000000000000000000000000000"), "nbits %s", hx);
    u256_from_be_hex(&b, "00000000ffff0000000000000000000000000000000000000000000000000000");
    CHECK(u256_cmp(&a, &b) == 0, "hex parse");
    u256_mul_small(&b, &a, 2);
    u256_to_be_hex(&b, hx);
    CHECK(!strcmp(hx, "00000001fffe0000000000000000000000000000000000000000000000000000"), "mul %s", hx);
    printf("u256 ok\n");
}

// Mine a tiny problem on the CPU with an easy target, build a proof, verify it.
static void test_proof_roundtrip(void) {
    for (int nt = 8; nt <= 32; nt *= 2) {
        prl_geom g;
        int m = 32, n = 8 * nt * 2, k = 2048;
        CHECK(prl_geom_init(&g, m, n, k, nt) == 0, "geom");
        uint8_t header[76];
        for (int i = 0; i < 76; i++) header[i] = (uint8_t)(i * 7 + 3);
        uint8_t *A = malloc((size_t)m * k), *Bt = malloc((size_t)n * k);
        uint32_t s = 12345;
        for (int i = 0; i < m * k; i++) { s = s * 1103515245 + 12345; A[i] = (uint8_t)(int8_t)((int)((s >> 16) % 129) - 64); }
        for (int i = 0; i < n * k; i++) { s = s * 1103515245 + 12345; Bt[i] = (uint8_t)(int8_t)((int)((s >> 16) % 129) - 64); }
        uint8_t jk[32];
        prl_job_key(header, &g, jk);
        prl_tree ta, tb;
        prl_tree_build(&ta, A, (size_t)m * k, jk, 2);
        prl_tree_build(&tb, Bt, (size_t)n * k, jk, 2);
        uint8_t chunk0[1024];
        memcpy(chunk0, A, 1024);
        prl_apply_nonce((int8_t *)chunk0, 0x0123456789abcdefULL);
        prl_path0 pa;
        prl_tree_path0(&ta, chunk0, &pa);
        uint8_t bseed[32], aseed[32];
        prl_seed_b(jk, prl_tree_root(&tb), n, true, bseed);
        prl_seed_a(bseed, pa.node[ta.nlayers - 1], m, true, aseed);
        // pick tile rows 8+3, cols offset 2 of the second period
        uint32_t t_rows = 16 + 3, t_cols = 8 * nt + 2;
        uint32_t rows[2], cols[64];
        prl_tile_rows(&g, t_rows, rows);
        prl_tile_cols(&g, t_cols, cols);
        int8_t *sa = malloc(2 * k), *sb = malloc(64 * k);
        uint8_t *perm = malloc(2 * k);
        int8_t uni[128], nz[65536];
        prl_noise_perm(PRL_LABEL_A, aseed, k, perm);
        for (int i = 0; i < 2; i++) {
            const uint8_t *src = rows[i] == 0 ? chunk0 : A + (size_t)rows[i] * k;
            prl_noise_uniform_row(PRL_LABEL_A, aseed, rows[i], uni);
            prl_noise_row(uni, perm, k, nz);
            for (int l = 0; l < k; l++) sa[i * k + l] = (int8_t)((int8_t)src[l] + nz[l]);
        }
        prl_noise_perm(PRL_LABEL_B, bseed, k, perm);
        for (int i = 0; i < g.w; i++) {
            prl_noise_uniform_row(PRL_LABEL_B, bseed, cols[i], uni);
            prl_noise_row(uni, perm, k, nz);
            for (int l = 0; l < k; l++) sb[i * k + l] = (int8_t)((int8_t)Bt[(size_t)cols[i] * k + l] + nz[l]);
        }
        const int8_t *ap[2] = {sa, sa + k}, *bp[64];
        for (int i = 0; i < g.w; i++) bp[i] = sb + i * k;
        uint32_t jp[16];
        prl_jackpot(ap, 2, bp, g.w, k, 128, jp);
        uint8_t hh[32];
        prl_jackpot_hash(jp, aseed, hh);
        prl_bytes pr;
        CHECK(prl_build_proof(&g, &ta, &pa, A, chunk0, &tb, Bt, t_rows, t_cols, &pr) == 0, "build");
        u256 easy;
        memset(&easy, 0xff, sizeof(easy));
        char err[256];
        uint8_t vh[32];
        int rc = prl_verify_proof(pr.buf, pr.len, header, true, &easy, vh, err, sizeof err);
        CHECK(rc == 0, "verify nt=%d: %s", nt, err);
        CHECK(!memcmp(vh, hh, 32), "verifier jackpot hash == miner jackpot hash");
        // wrong derivation must change the hash (legacy) and hard target must fail
        rc = prl_verify_proof(pr.buf, pr.len, header, false, &easy, vh, err, sizeof err);
        CHECK(rc == 0 && memcmp(vh, hh, 32) != 0, "legacy differs");
        u256 zero;
        memset(&zero, 0, sizeof zero);
        CHECK(prl_verify_proof(pr.buf, pr.len, header, true, &zero, NULL, err, sizeof err) != 0, "hard target fails");
        // tampering with a leaf must break the merkle root
        pr.buf[8 * 4 + 8 + 8 + 100] ^= 1;
        CHECK(prl_verify_proof(pr.buf, pr.len, header, true, &easy, NULL, err, sizeof err) != 0, "tamper detected");
        pr.buf[8 * 4 + 8 + 8 + 100] ^= 1;
        // tampering with header must change job key -> roots mismatch
        uint8_t h2[76];
        memcpy(h2, header, 76);
        h2[70] ^= 1;
        CHECK(prl_verify_proof(pr.buf, pr.len, h2, true, &easy, NULL, err, sizeof err) != 0, "header bound");
        printf("proof nt=%d: %zu bytes, roundtrip ok\n", nt, pr.len);
        prl_bytes_free(&pr);
        prl_tree_free(&ta); prl_tree_free(&tb);
        free(A); free(Bt); free(sa); free(sb); free(perm);
    }
}

int main(void) {
    test_blake3();
    test_commitment_pinned();
    test_patterns();
    test_u256();
    test_proof_roundtrip();
    if (fails) { printf("%d FAILURES\n", fails); return 1; }
    printf("ALL CORE TESTS PASSED\n");
    return 0;
}
