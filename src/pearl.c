// ejjjminer — Pearl protocol core. See pearl.h.
#include "pearl.h"
#include "../third_party/blake3/blake3.h"
#include "../third_party/blake3/blake3_impl.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// =====================================================================================
// u256
// =====================================================================================
static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

void u256_from_be_hex(u256 *out, const char *hex) {
    memset(out, 0, sizeof(*out));
    if (hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X')) hex += 2;
    size_t n = strlen(hex);
    // read from the end (least significant nibble first)
    for (size_t i = 0; i < n && i < 64; i++) {
        int v = hexval(hex[n - 1 - i]);
        if (v < 0) v = 0;
        out->w[i / 8] |= (uint32_t)v << (4 * (i % 8));
    }
}

void u256_from_le_bytes(u256 *out, const uint8_t b[32]) {
    for (int i = 0; i < 8; i++)
        out->w[i] = (uint32_t)b[4 * i] | ((uint32_t)b[4 * i + 1] << 8) | ((uint32_t)b[4 * i + 2] << 16) |
                    ((uint32_t)b[4 * i + 3] << 24);
}

void u256_to_be_hex(const u256 *v, char out[65]) {
    static const char *H = "0123456789abcdef";
    for (int i = 0; i < 64; i++) {
        int nib = 63 - i;
        out[i] = H[(v->w[nib / 8] >> (4 * (nib % 8))) & 15];
    }
    out[64] = 0;
}

int u256_cmp(const u256 *a, const u256 *b) {
    for (int i = 7; i >= 0; i--) {
        if (a->w[i] < b->w[i]) return -1;
        if (a->w[i] > b->w[i]) return 1;
    }
    return 0;
}

bool u256_mul_small(u256 *out, const u256 *a, uint32_t f) {
    uint64_t carry = 0;
    u256 r;
    for (int i = 0; i < 8; i++) {
        uint64_t t = (uint64_t)a->w[i] * f + carry;
        r.w[i] = (uint32_t)t;
        carry = t >> 32;
    }
    if (carry) {
        for (int i = 0; i < 8; i++) out->w[i] = 0xffffffffu;
        return false;
    }
    *out = r;
    return true;
}

void u256_from_nbits(u256 *out, uint32_t nbits) {
    memset(out, 0, sizeof(*out));
    uint32_t exponent = nbits >> 24, mantissa = nbits & 0x00ffffff;
    if (mantissa == 0 || exponent == 0 || (mantissa & 0x00800000)) return;
    if (exponent <= 3) {
        out->w[0] = mantissa >> (8 * (3 - exponent));
        return;
    }
    // mantissa << 8*(exponent-3)
    int shift = 8 * (int)(exponent - 3);
    if (shift >= 256) {
        for (int i = 0; i < 8; i++) out->w[i] = 0xffffffffu;
        return;
    }
    int limb = shift / 32, bit = shift % 32;
    uint64_t v = (uint64_t)mantissa << bit;
    if (limb < 8) out->w[limb] = (uint32_t)v;
    if (limb + 1 < 8) out->w[limb + 1] = (uint32_t)(v >> 32);
}

void u256_from_double(u256 *out, double v) {
    memset(out, 0, sizeof(*out));
    if (!(v >= 1)) return;
    int e;
    double f = frexp(v, &e);                 // v = f * 2^e, f in [0.5, 1)
    uint64_t mant = (uint64_t)ldexp(f, 53);  // 53-bit mantissa
    int shift = e - 53;
    if (shift >= 256 - 53) { for (int i = 0; i < 8; i++) out->w[i] = 0xffffffffu; return; }
    if (shift < 0) { mant >>= -shift; shift = 0; }
    for (int b = 0; b < 64; b++)
        if (mant >> b & 1) { int p = b + shift; if (p < 256) out->w[p / 32] |= 1u << (p % 32); }
}

double u256_to_double(const u256 *v) {
    double d = 0;
    for (int i = 7; i >= 0; i--) d = d * 4294967296.0 + (double)v->w[i];
    return d;
}

// =====================================================================================
// periodic patterns (port of zk-pow PeriodicPattern)
// =====================================================================================
int prl_pattern_to_list(const prl_pattern *p, uint32_t *out, int cap) {
    uint32_t tmp[4096];
    int n = 1;
    tmp[0] = 0;
    for (int d = 0; d < 3; d++) {
        uint32_t nw[4096];
        int nn = 0;
        for (uint32_t i = 0; i < p->ln[d]; i++)
            for (int r = 0; r < n; r++) {
                if (nn >= 4096) return -1;
                nw[nn++] = tmp[r] + i * p->st[d];
            }
        memcpy(tmp, nw, sizeof(uint32_t) * nn);
        n = nn;
    }
    if (n > cap) return -1;
    memcpy(out, tmp, sizeof(uint32_t) * n);
    return n;
}

void prl_pattern_to_bytes(const prl_pattern *p, uint8_t out[6]) {
    uint32_t min_stride = 1;
    for (int i = 0; i < 3; i++) {
        uint32_t factor = p->st[i] / min_stride;
        out[2 * i] = (uint8_t)(factor - 1);
        out[2 * i + 1] = (uint8_t)(p->ln[i] - 1);
        min_stride = p->st[i] * p->ln[i];
    }
}

static int pattern_from_bytes(prl_pattern *p, const uint8_t data[6]) {
    uint32_t min_stride = 1;
    bool done = false;
    for (int i = 0; i < 3; i++) {
        uint32_t factor = 1 + data[2 * i], length = 1 + data[2 * i + 1];
        if (length == 1 || done) {
            if (!(factor == 1 && length == 1)) return -1;
            done = true;
        } else if (factor <= 1 && min_stride != 1) {
            return -1;
        }
        if (!(min_stride <= (1u << 24) / (factor * length))) return -1;
        uint32_t stride = factor * min_stride;
        p->st[i] = stride;
        p->ln[i] = length;
        min_stride = stride * length;
    }
    uint8_t rt[6];
    prl_pattern_to_bytes(p, rt);
    return memcmp(rt, data, 6) == 0 ? 0 : -1;
}

int prl_pattern_from_list(prl_pattern *pat, const uint32_t *idx, int n) {
    if (n <= 0 || n > 4096) return -1;
    if (idx[0] != 0) return -1;
    for (int i = 1; i < n; i++)
        if (!(idx[i - 1] < idx[i])) return -1;
    uint32_t p[4096];
    memcpy(p, idx, sizeof(uint32_t) * n);
    int len = n;
    uint32_t sv_s[8], sv_l[8];
    int ns = 0;
    while (len > 1) {
        bool found = false;
        for (int period = 1; period < len; period++) {
            if (len % period) continue;
            uint32_t s = p[period];
            bool per = true;
            for (int i = 0; i < len - period; i++)
                if (p[i] + s != p[i + period]) { per = false; break; }
            if (per) {
                if (ns >= 8) return -1;
                sv_s[ns] = s;
                sv_l[ns] = (uint32_t)(len / period);
                ns++;
                len = period;
                found = true;
                break;
            }
        }
        if (!found) return -1;
    }
    if (ns > 3) return -1;
    // reverse
    uint32_t rs[3], rl[3];
    for (int i = 0; i < ns; i++) { rs[i] = sv_s[ns - 1 - i]; rl[i] = sv_l[ns - 1 - i]; }
    uint32_t period = ns ? rs[ns - 1] * rl[ns - 1] : 1;
    for (int i = ns; i < 3; i++) { rs[i] = period; rl[i] = 1; }
    for (int i = 0; i < 3; i++) { pat->st[i] = rs[i]; pat->ln[i] = rl[i]; }
    // is_valid: roundtrip through bytes
    uint8_t b[6];
    prl_pattern_to_bytes(pat, b);
    prl_pattern rt;
    if (pattern_from_bytes(&rt, b) != 0) return -1;
    for (int i = 0; i < 3; i++)
        if (rt.st[i] != pat->st[i] || rt.ln[i] != pat->ln[i]) return -1;
    return 0;
}

uint32_t prl_pattern_period(const prl_pattern *p) { return p->st[2] * p->ln[2]; }
uint32_t prl_pattern_size(const prl_pattern *p) { return p->ln[0] * p->ln[1] * p->ln[2]; }
uint32_t prl_pattern_max(const prl_pattern *p) {
    uint32_t mx = 0;
    for (int i = 0; i < 3; i++) mx += (p->ln[i] - 1) * p->st[i];
    return mx;
}
bool prl_pattern_offset_valid(const prl_pattern *p, uint32_t off) {
    for (int i = 2; i >= 0; i--) {
        off %= p->st[i] * p->ln[i];
        if (off >= p->st[i]) return false;
    }
    return true;
}

// =====================================================================================
// geometry / mining configuration
// =====================================================================================
static void config_bytes(uint32_t k, uint16_t rank, const prl_pattern *rows, const prl_pattern *cols,
                         uint8_t out[PRL_CONFIG_LEN]) {
    memset(out, 0, PRL_CONFIG_LEN);
    out[0] = k & 0xff; out[1] = (k >> 8) & 0xff; out[2] = (k >> 16) & 0xff; out[3] = (k >> 24) & 0xff;
    out[4] = rank & 0xff; out[5] = rank >> 8;
    out[6] = 0; out[7] = 0;                      // MMAType::Int7xInt7ToInt32
    prl_pattern_to_bytes(rows, out + 8);
    prl_pattern_to_bytes(cols, out + 14);
    // 32-byte trailer: zero (dense job, no MoE)
}

int prl_geom_init(prl_geom *g, int m, int n, int k, int nt) {
    memset(g, 0, sizeof(*g));
    if (nt != 8 && nt != 16 && nt != 32) return -1;
    if (k < 16 * PRL_RANK || k % (PRL_RANK * PRL_JACKPOT_WORDS) || k > 65536) return -2;
    if (m < 16 || m % 16 || n % (8 * nt) || n < 8 * nt) return -3;
    if (m > (1 << 24) || n > (1 << 24)) return -3;
    g->m = m; g->n = n; g->k = k; g->nt = nt;
    g->h = 2; g->w = 2 * nt;
    uint32_t rl[2] = {0, 8};
    uint32_t cl[64];
    for (int j = 0; j < nt; j++) { cl[2 * j] = 8 * j; cl[2 * j + 1] = 8 * j + 1; }
    if (prl_pattern_from_list(&g->rows, rl, 2)) return -4;
    if (prl_pattern_from_list(&g->cols, cl, 2 * nt)) return -4;
    config_bytes((uint32_t)k, PRL_RANK, &g->rows, &g->cols, g->config_bytes);
    // penalized factor: tile * (dpl / rank) * 128, dpl = k - k%rank
    uint32_t dpl = (uint32_t)(k - k % PRL_RANK);
    g->factor = (uint32_t)(g->h * g->w) * (dpl / PRL_RANK) * 128u;
    return 0;
}

void prl_tile_rows(const prl_geom *g, uint32_t t, uint32_t *rows) { (void)g; rows[0] = t; rows[1] = t + 8; }
void prl_tile_cols(const prl_geom *g, uint32_t c, uint32_t *cols) {
    for (int j = 0; j < g->nt; j++) { cols[2 * j] = c + 8 * j; cols[2 * j + 1] = c + 8 * j + 1; }
}

// =====================================================================================
// hashing
// =====================================================================================
void prl_hash(const void *data, size_t len, uint8_t out[32]) {
    blake3_hasher h;
    blake3_hasher_init(&h);
    blake3_hasher_update(&h, data, len);
    blake3_hasher_finalize(&h, out, 32);
}

void prl_keyed_hash(const uint8_t key[32], const void *data, size_t len, uint8_t out[32]) {
    blake3_hasher h;
    blake3_hasher_init_keyed(&h, key);
    blake3_hasher_update(&h, data, len);
    blake3_hasher_finalize(&h, out, 32);
}

static void key_words(const uint8_t key[32], uint32_t kw[8]) {
    for (int i = 0; i < 8; i++)
        kw[i] = (uint32_t)key[4 * i] | ((uint32_t)key[4 * i + 1] << 8) | ((uint32_t)key[4 * i + 2] << 16) |
                ((uint32_t)key[4 * i + 3] << 24);
}
static void words_to_bytes(const uint32_t w[8], uint8_t out[32]) {
    for (int i = 0; i < 8; i++) {
        out[4 * i] = w[i] & 0xff; out[4 * i + 1] = (w[i] >> 8) & 0xff;
        out[4 * i + 2] = (w[i] >> 16) & 0xff; out[4 * i + 3] = w[i] >> 24;
    }
}

static void chunk_cv(const uint8_t key[32], const uint8_t chunk[PRL_CHUNK], uint64_t index, uint8_t out[32]) {
    uint32_t kw[8];
    key_words(key, kw);
    const uint8_t *in[1] = {chunk};
    blake3_hash_many(in, 1, PRL_CHUNK / BLAKE3_BLOCK_LEN, kw, index, true, KEYED_HASH, CHUNK_START, CHUNK_END, out);
}

static void parent_cv(const uint8_t key[32], const uint8_t l[32], const uint8_t r[32], bool root, uint8_t out[32]) {
    uint32_t cv[8];
    key_words(key, cv);
    uint8_t block[64];
    memcpy(block, l, 32);
    memcpy(block + 32, r, 32);
    blake3_compress_in_place(cv, block, 64, 0, KEYED_HASH | PARENT | (root ? ROOT : 0));
    words_to_bytes(cv, out);
}

// =====================================================================================
// Merkle tree (port of pearl-blake3 MerkleTree)
// =====================================================================================
typedef struct {
    const uint8_t *data;
    size_t first, count;
    const uint8_t *key;
    uint8_t *out;
} leaf_job;

static void *leaf_worker(void *arg) {
    leaf_job *j = (leaf_job *)arg;
    uint32_t kw[8];
    key_words(j->key, kw);
    const size_t B = 64;
    const uint8_t *ptrs[64];
    for (size_t i = 0; i < j->count; i += B) {
        size_t c = j->count - i < B ? j->count - i : B;
        for (size_t t = 0; t < c; t++) ptrs[t] = j->data + (j->first + i + t) * PRL_CHUNK;
        blake3_hash_many(ptrs, c, PRL_CHUNK / BLAKE3_BLOCK_LEN, kw, j->first + i, true, KEYED_HASH, CHUNK_START,
                         CHUNK_END, j->out + (j->first + i) * 32);
    }
    return NULL;
}

int prl_tree_build(prl_tree *t, const uint8_t *data, size_t len, const uint8_t key[32], int nthreads) {
    memset(t, 0, sizeof(*t));
    memcpy(t->key, key, 32);
    if (len == 0 || len % PRL_CHUNK) return -1;   // callers always pass chunk-aligned data
    size_t nch = len / PRL_CHUNK;
    t->data_len = len;
    if (nch < 2) return -1;
    t->len[0] = nch;
    t->cv[0] = (uint8_t *)malloc(nch * 32);
    if (!t->cv[0]) return -1;
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 64) nthreads = 64;
    if ((size_t)nthreads > nch / 64 + 1) nthreads = (int)(nch / 64 + 1);
    pthread_t th[64];
    leaf_job jobs[64];
    size_t per = (nch + nthreads - 1) / nthreads;
    int started = 0;
    for (int i = 0; i < nthreads; i++) {
        size_t f = (size_t)i * per;
        if (f >= nch) break;
        jobs[i].data = data; jobs[i].first = f; jobs[i].count = (f + per > nch) ? nch - f : per;
        jobs[i].key = key; jobs[i].out = t->cv[0];
        if (i == nthreads - 1 || pthread_create(&th[i], NULL, leaf_worker, &jobs[i]) != 0) {
            leaf_worker(&jobs[i]);
            th[i] = 0;
        }
        started = i + 1;
    }
    for (int i = 0; i < started; i++)
        if (th[i]) pthread_join(th[i], NULL);
    int L = 1;
    while (t->len[L - 1] > 2) {
        size_t pl = t->len[L - 1], nl = (pl + 1) / 2;
        t->cv[L] = (uint8_t *)malloc(nl * 32);
        if (!t->cv[L]) return -1;
        for (size_t i = 0; i < nl; i++) {
            if (2 * i + 1 < pl) parent_cv(key, t->cv[L - 1] + 64 * i, t->cv[L - 1] + 64 * i + 32, false, t->cv[L] + 32 * i);
            else memcpy(t->cv[L] + 32 * i, t->cv[L - 1] + 64 * i, 32);
        }
        t->len[L] = nl;
        L++;
    }
    // last layer has 2 entries -> root layer
    t->cv[L] = (uint8_t *)malloc(32);
    parent_cv(key, t->cv[L - 1], t->cv[L - 1] + 32, true, t->cv[L]);
    t->len[L] = 1;
    t->nlayers = L + 1;
    return 0;
}

void prl_tree_free(prl_tree *t) {
    for (int i = 0; i < t->nlayers; i++) free(t->cv[i]);
    memset(t, 0, sizeof(*t));
}

const uint8_t *prl_tree_root(const prl_tree *t) { return t->cv[t->nlayers - 1]; }

void prl_tree_path0(const prl_tree *t, const uint8_t chunk0[PRL_CHUNK], prl_path0 *out) {
    out->nlayers = t->nlayers;
    chunk_cv(t->key, chunk0, 0, out->node[0]);
    for (int l = 1; l < t->nlayers; l++) {
        bool root = (l == t->nlayers - 1);
        parent_cv(t->key, out->node[l - 1], t->cv[l - 1] + 32, root, out->node[l]);
    }
}

static const uint8_t *tree_node(const prl_tree *t, const prl_path0 *p, int level, size_t i) {
    if (p && i == 0) return p->node[level];
    return t->cv[level] + 32 * i;
}

// =====================================================================================
// job key / seeds
// =====================================================================================
static const uint8_t SEED_SALT_A[32] = {
    0x82, 0x49, 0x40, 0x6c, 0xa0, 0xed, 0x15, 0x16, 0x96, 0x16, 0xf6, 0x92, 0xfc, 0xf0, 0x76, 0xf8,
    0x92, 0xdb, 0xdb, 0x2a, 0x70, 0x23, 0xb8, 0x52, 0xf0, 0xd4, 0x77, 0x19, 0xc3, 0x90, 0x01, 0x7b};
static const uint8_t SEED_SALT_B[32] = {
    0x11, 0x30, 0x06, 0x32, 0xec, 0x63, 0x01, 0xca, 0x2b, 0xe2, 0xaf, 0x71, 0x8b, 0x3f, 0x4d, 0x4f,
    0x1a, 0xe9, 0xc6, 0x39, 0x88, 0xe8, 0xcc, 0x04, 0x48, 0x44, 0x30, 0x1d, 0x71, 0xb8, 0x9a, 0xa9};

void prl_job_key(const uint8_t header[PRL_HEADER_LEN], const prl_geom *g, uint8_t out[32]) {
    uint8_t buf[PRL_HEADER_LEN + PRL_CONFIG_LEN];
    memcpy(buf, header, PRL_HEADER_LEN);
    memcpy(buf + PRL_HEADER_LEN, g->config_bytes, PRL_CONFIG_LEN);
    prl_hash(buf, sizeof(buf), out);
}

static void bind_root(const uint8_t root[32], uint32_t dim, const uint8_t salt[32], uint8_t out[32]) {
    uint8_t msg[64] = {0};
    memcpy(msg, root, 32);
    msg[32] = dim & 0xff; msg[33] = (dim >> 8) & 0xff; msg[34] = (dim >> 16) & 0xff; msg[35] = dim >> 24;
    prl_keyed_hash(salt, msg, 64, out);
}

void prl_seed_b(const uint8_t job_key[32], const uint8_t hash_b[32], uint32_t n, bool salted, uint8_t out[32]) {
    uint8_t hb[32], buf[64];
    if (salted) bind_root(hash_b, n, SEED_SALT_B, hb); else memcpy(hb, hash_b, 32);
    memcpy(buf, job_key, 32);
    memcpy(buf + 32, hb, 32);
    prl_hash(buf, 64, out);
}

void prl_seed_a(const uint8_t b_seed[32], const uint8_t hash_a[32], uint32_t m, bool salted, uint8_t out[32]) {
    uint8_t ha[32], buf[64];
    if (salted) bind_root(hash_a, m, SEED_SALT_A, ha); else memcpy(ha, hash_a, 32);
    memcpy(buf, b_seed, 32);
    memcpy(buf + 32, ha, 32);
    prl_hash(buf, 64, out);
}

// =====================================================================================
// noise (port of zk-pow circuit/pearl_noise.rs)
// =====================================================================================
const uint8_t PRL_LABEL_A[32] = {'A', '_', 't', 'e', 'n', 's', 'o', 'r'};
const uint8_t PRL_LABEL_B[32] = {'B', '_', 't', 'e', 'n', 's', 'o', 'r'};

void prl_noise_hash(uint32_t index, const uint8_t label[32], const uint8_t key[32], int slot, uint8_t out[32]) {
    uint8_t msg[64] = {0};
    uint32_t v = 1 + index;
    msg[slot * 4] = v & 0xff; msg[slot * 4 + 1] = (v >> 8) & 0xff;
    msg[slot * 4 + 2] = (v >> 16) & 0xff; msg[slot * 4 + 3] = v >> 24;
    memcpy(msg + 32, label, 32);
    prl_keyed_hash(key, msg, 64, out);
}

void prl_noise_uniform_row(const uint8_t label[32], const uint8_t key[32], uint32_t row, int8_t *out) {
    // element idx = row*rank + c drawn from hash block idx/32, byte idx%32
    for (int b = 0; b < PRL_RANK / 32; b++) {
        uint8_t h[32];
        prl_noise_hash(row * (PRL_RANK / 32) + b, label, key, 0, h);
        for (int i = 0; i < 32; i++) out[b * 32 + i] = (int8_t)((h[i] & 63) - 32);
    }
}

void prl_noise_perm(const uint8_t label[32], const uint8_t key[32], int k, uint8_t *out) {
    for (int i = 0; i < (k + 7) / 8; i++) {
        uint8_t h[32];
        prl_noise_hash((uint32_t)i, label, key, 1, h);
        for (int j = 0; j < 8 && i * 8 + j < k; j++) {
            uint32_t u = (uint32_t)h[4 * j] | ((uint32_t)h[4 * j + 1] << 8) | ((uint32_t)h[4 * j + 2] << 16) |
                         ((uint32_t)h[4 * j + 3] << 24);
            uint32_t first = u & (PRL_RANK - 1);
            uint32_t second = first ^ (1u + (uint32_t)(((uint64_t)(PRL_RANK - 1) * u) >> 32));
            out[2 * (i * 8 + j)] = (uint8_t)first;
            out[2 * (i * 8 + j) + 1] = (uint8_t)second;
        }
    }
}

void prl_noise_row(const int8_t *uni, const uint8_t *perm, int k, int8_t *out) {
    for (int l = 0; l < k; l++) out[l] = (int8_t)((int)uni[perm[2 * l]] - (int)uni[perm[2 * l + 1]]);
}

void prl_apply_nonce(int8_t *row0, uint64_t nonce) {
    for (int i = 0; i < PRL_NONCE_BYTES; i++) row0[i] = (int8_t)((nonce >> (4 * i)) & 15);
}

// =====================================================================================
// jackpot
// =====================================================================================
static inline uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

void prl_jackpot(const int8_t *const *a, int h, const int8_t *const *b, int w, int k, int rank,
                 uint32_t jp[PRL_JACKPOT_WORDS]) {
    int32_t acc[2 * 64 * 4];
    memset(acc, 0, sizeof(int32_t) * h * w);
    memset(jp, 0, sizeof(uint32_t) * PRL_JACKPOT_WORDS);
    for (int c = 0; c < k / rank; c++) {
        uint32_t x = 0;
        for (int u = 0; u < h; u++)
            for (int v = 0; v < w; v++) {
                int32_t s = 0;
                const int8_t *pa = a[u] + c * rank, *pb = b[v] + c * rank;
                for (int l = 0; l < rank; l++) s += (int32_t)pa[l] * (int32_t)pb[l];
                acc[u * w + v] += s;
                x ^= (uint32_t)acc[u * w + v];
            }
        int t = c % PRL_JACKPOT_WORDS;
        jp[t] = rotl32(jp[t], PRL_LROT) ^ x;
    }
}

void prl_jackpot_hash(const uint32_t jp[PRL_JACKPOT_WORDS], const uint8_t a_seed[32], uint8_t out[32]) {
    uint8_t msg[64];
    for (int i = 0; i < 16; i++) {
        msg[4 * i] = jp[i] & 0xff; msg[4 * i + 1] = (jp[i] >> 8) & 0xff;
        msg[4 * i + 2] = (jp[i] >> 16) & 0xff; msg[4 * i + 3] = jp[i] >> 24;
    }
    prl_keyed_hash(a_seed, msg, 64, out);
}

// =====================================================================================
// bincode writer
// =====================================================================================
void prl_bytes_free(prl_bytes *b) { free(b->buf); memset(b, 0, sizeof(*b)); }

static int bput(prl_bytes *b, const void *d, size_t n) {
    if (b->len + n > b->cap) {
        size_t nc = b->cap ? b->cap * 2 : 65536;
        while (nc < b->len + n) nc *= 2;
        uint8_t *nb = (uint8_t *)realloc(b->buf, nc);
        if (!nb) return -1;
        b->buf = nb; b->cap = nc;
    }
    memcpy(b->buf + b->len, d, n);
    b->len += n;
    return 0;
}
static int bput_u64(prl_bytes *b, uint64_t v) {
    uint8_t t[8];
    for (int i = 0; i < 8; i++) t[i] = (uint8_t)(v >> (8 * i));
    return bput(b, t, 8);
}

// ---- index set helpers (small sorted arrays)
typedef struct { size_t *v; size_t n, cap; } idxset;
static void is_add(idxset *s, size_t x) {
    // insert keeping sorted unique
    size_t lo = 0, hi = s->n;
    while (lo < hi) { size_t mid = (lo + hi) / 2; if (s->v[mid] < x) lo = mid + 1; else hi = mid; }
    if (lo < s->n && s->v[lo] == x) return;
    if (s->n == s->cap) { s->cap = s->cap ? s->cap * 2 : 64; s->v = (size_t *)realloc(s->v, s->cap * sizeof(size_t)); }
    memmove(s->v + lo + 1, s->v + lo, (s->n - lo) * sizeof(size_t));
    s->v[lo] = x;
    s->n++;
}
static bool is_has(const idxset *s, size_t x) {
    size_t lo = 0, hi = s->n;
    while (lo < hi) { size_t mid = (lo + hi) / 2; if (s->v[mid] < x) lo = mid + 1; else hi = mid; }
    return lo < s->n && s->v[lo] == x;
}

// Serialize one MatrixMerkleProof for the given matrix rows.
static int write_matrix_proof(prl_bytes *b, const prl_tree *t, const prl_path0 *p, const uint8_t *data,
                              const uint8_t *chunk0, size_t ncols, const uint32_t *rows, int nrows) {
    idxset leaves = {0};
    for (int r = 0; r < nrows; r++) {
        size_t first = ((size_t)rows[r] * ncols) / PRL_CHUNK;
        size_t last = (((size_t)rows[r] + 1) * ncols - 1) / PRL_CHUNK;
        for (size_t i = first; i <= last; i++) is_add(&leaves, i);
    }
    size_t total = t->len[0];
    int rc = 0;
    // leaf_data: Vec<Vec<u8>> (each 1024)
    rc |= bput_u64(b, leaves.n);
    for (size_t i = 0; i < leaves.n; i++) {
        rc |= bput_u64(b, PRL_CHUNK);
        const uint8_t *src = (leaves.v[i] == 0 && chunk0) ? chunk0 : data + leaves.v[i] * PRL_CHUNK;
        rc |= bput(b, src, PRL_CHUNK);
    }
    // leaf_indices
    rc |= bput_u64(b, leaves.n);
    for (size_t i = 0; i < leaves.n; i++) rc |= bput_u64(b, leaves.v[i]);
    rc |= bput_u64(b, total);
    // root
    const uint8_t *root = p ? p->node[t->nlayers - 1] : prl_tree_root(t);
    rc |= bput(b, root, 32);
    // siblings: walk levels exactly like MerkleTree::get_multileaf_proof
    prl_bytes sib = {0};
    size_t nsib = 0;
    idxset cur = {0};
    for (size_t i = 0; i < leaves.n; i++) is_add(&cur, leaves.v[i]);
    size_t level_len = total;
    int level = 0;
    while (level_len > 1 && cur.n) {
        for (size_t q = 0; q < cur.n; q++) {
            size_t i = cur.v[q];
            if (i % 2 == 1) {
                if (!is_has(&cur, i - 1)) { rc |= bput(&sib, tree_node(t, p, level, i - 1), 32); nsib++; }
            } else if (!is_has(&cur, i + 1) && i + 1 < level_len) {
                rc |= bput(&sib, tree_node(t, p, level, i + 1), 32); nsib++;
            }
        }
        idxset nx = {0};
        for (size_t q = 0; q < cur.n; q++) is_add(&nx, cur.v[q] / 2);
        free(cur.v);
        cur = nx;
        level_len = (level_len + 1) / 2;
        level++;
    }
    free(cur.v);
    rc |= bput_u64(b, nsib);
    if (nsib) rc |= bput(b, sib.buf, sib.len);
    prl_bytes_free(&sib);
    // row_indices
    rc |= bput_u64(b, (uint64_t)nrows);
    for (int r = 0; r < nrows; r++) rc |= bput_u64(b, rows[r]);
    free(leaves.v);
    return rc ? -1 : 0;
}

int prl_build_proof(const prl_geom *g, const prl_tree *ta, const prl_path0 *pa, const uint8_t *A_data,
                    const uint8_t *a_chunk0, const prl_tree *tb, const uint8_t *Bt_data, uint32_t t_rows,
                    uint32_t t_cols, prl_bytes *out) {
    memset(out, 0, sizeof(*out));
    uint32_t rows[2], cols[64];
    prl_tile_rows(g, t_rows, rows);
    prl_tile_cols(g, t_cols, cols);
    int rc = 0;
    rc |= bput_u64(out, (uint64_t)g->m);
    rc |= bput_u64(out, (uint64_t)g->n);
    rc |= bput_u64(out, (uint64_t)g->k);
    rc |= bput_u64(out, PRL_RANK);
    rc |= write_matrix_proof(out, ta, pa, A_data, a_chunk0, (size_t)g->k, rows, g->h);
    rc |= write_matrix_proof(out, tb, NULL, Bt_data, NULL, (size_t)g->k, cols, g->w);
    uint8_t none = 0;
    rc |= bput(out, &none, 1);   // moe: None
    return rc ? -1 : 0;
}

// =====================================================================================
// verifier (independent re-computation from the serialized bytes)
// =====================================================================================
typedef struct { const uint8_t *p; size_t n, off; bool bad; } rd;
static uint64_t rd_u64(rd *r) {
    if (r->off + 8 > r->n) { r->bad = true; return 0; }
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)r->p[r->off + i] << (8 * i);
    r->off += 8;
    return v;
}
static const uint8_t *rd_bytes(rd *r, size_t n) {
    if (r->off + n > r->n) { r->bad = true; return NULL; }
    const uint8_t *p = r->p + r->off;
    r->off += n;
    return p;
}

typedef struct {
    size_t nleaves;
    const uint8_t **leaf;      // 1024 each
    uint64_t *leaf_idx;
    uint64_t total;
    const uint8_t *root;
    size_t nsib;
    const uint8_t *sib;        // 32 each
    size_t nrows;
    uint64_t *rows;
} mproof;

static void mproof_free(mproof *m) { free(m->leaf); free(m->leaf_idx); free(m->rows); }

static int parse_mproof(rd *r, mproof *m) {
    memset(m, 0, sizeof(*m));
    uint64_t nl = rd_u64(r);
    if (r->bad || nl == 0 || nl > 100000) return -1;
    m->nleaves = nl;
    m->leaf = (const uint8_t **)calloc(nl, sizeof(*m->leaf));
    for (uint64_t i = 0; i < nl; i++) {
        uint64_t l = rd_u64(r);
        if (l != PRL_CHUNK) return -1;
        m->leaf[i] = rd_bytes(r, PRL_CHUNK);
        if (r->bad) return -1;
    }
    uint64_t ni = rd_u64(r);
    if (ni != nl) return -1;
    m->leaf_idx = (uint64_t *)calloc(nl, sizeof(uint64_t));
    for (uint64_t i = 0; i < nl; i++) m->leaf_idx[i] = rd_u64(r);
    m->total = rd_u64(r);
    m->root = rd_bytes(r, 32);
    m->nsib = rd_u64(r);
    if (r->bad || m->nsib > 100000) return -1;
    m->sib = rd_bytes(r, 32 * m->nsib);
    m->nrows = rd_u64(r);
    if (r->bad || m->nrows == 0 || m->nrows > 4096) return -1;
    m->rows = (uint64_t *)calloc(m->nrows, sizeof(uint64_t));
    for (size_t i = 0; i < m->nrows; i++) m->rows[i] = rd_u64(r);
    if (r->bad) return -1;
    for (size_t i = 1; i < nl; i++) if (m->leaf_idx[i - 1] >= m->leaf_idx[i]) return -1;
    return 0;
}

// map index -> cv (sorted array of pairs)
typedef struct { size_t idx; uint8_t cv[32]; } node_t;

static int mproof_root(const mproof *m, const uint8_t key[32], uint8_t out[32]) {
    size_t n = m->nleaves;
    node_t *cur = (node_t *)malloc(n * sizeof(node_t));
    for (size_t i = 0; i < n; i++) {
        cur[i].idx = m->leaf_idx[i];
        chunk_cv(key, m->leaf[i], m->leaf_idx[i], cur[i].cv);
    }
    size_t level_len = m->total, si = 0;
    if (level_len == 1) { free(cur); return -1; }
    while (level_len > 2) {
        node_t *nx = (node_t *)malloc(n * sizeof(node_t));
        size_t nn = 0;
        for (size_t q = 0; q < n; q++) {
            size_t i = cur[q].idx;
            if (i % 2 == 0) {
                bool has_r = (q + 1 < n && cur[q + 1].idx == i + 1);
                nx[nn].idx = i / 2;
                if (has_r) parent_cv(key, cur[q].cv, cur[q + 1].cv, false, nx[nn].cv);
                else if (i + 1 < level_len) {
                    if (si >= m->nsib) { free(nx); free(cur); return -1; }
                    parent_cv(key, cur[q].cv, m->sib + 32 * si++, false, nx[nn].cv);
                } else memcpy(nx[nn].cv, cur[q].cv, 32);
                nn++;
            } else {
                if (q > 0 && cur[q - 1].idx == i - 1) continue;
                if (si >= m->nsib) { free(nx); free(cur); return -1; }
                nx[nn].idx = i / 2;
                parent_cv(key, m->sib + 32 * si++, cur[q].cv, false, nx[nn].cv);
                nn++;
            }
        }
        free(cur);
        cur = nx;
        n = nn;
        level_len = (level_len + 1) / 2;
    }
    uint8_t l[32], r[32];
    size_t q = 0;
    if (q < n && cur[q].idx == 0) memcpy(l, cur[q++].cv, 32);
    else { if (si >= m->nsib) { free(cur); return -1; } memcpy(l, m->sib + 32 * si++, 32); }
    if (q < n && cur[q].idx == 1) memcpy(r, cur[q++].cv, 32);
    else { if (si >= m->nsib) { free(cur); return -1; } memcpy(r, m->sib + 32 * si++, 32); }
    free(cur);
    if (si != m->nsib) return -1;
    parent_cv(key, l, r, true, out);
    return 0;
}

static int mproof_extract(const mproof *m, size_t start, size_t len, int8_t *out) {
    size_t copied = 0, end = start + len;
    for (size_t i = 0; i < m->nleaves; i++) {
        size_t ls = m->leaf_idx[i] * PRL_CHUNK, le = ls + PRL_CHUNK;
        if (ls < end && start < le) {
            size_t cs = start > ls ? start : ls, ce = end < le ? end : le;
            memcpy(out + (cs - start), m->leaf[i] + (cs - ls), ce - cs);
            copied += ce - cs;
        }
    }
    return copied == len ? 0 : -1;
}

int prl_verify_raw_target;

#define VFAIL(...) do { snprintf(err, errlen, __VA_ARGS__); goto done; } while (0)

int prl_verify_proof(const uint8_t *proof, size_t len, const uint8_t header[PRL_HEADER_LEN], bool salted,
                     const u256 *base_target, uint8_t hash_out[32], char *err, size_t errlen) {
    int ret = -1;
    rd r = {proof, len, 0, false};
    mproof pa, pb;
    memset(&pa, 0, sizeof(pa));
    memset(&pb, 0, sizeof(pb));
    int8_t *strips = NULL;
    uint64_t m = rd_u64(&r), n = rd_u64(&r), k = rd_u64(&r), rank = rd_u64(&r);
    if (r.bad) VFAIL("truncated header");
    if (parse_mproof(&r, &pa)) VFAIL("bad A proof encoding");
    if (parse_mproof(&r, &pb)) VFAIL("bad B proof encoding");
    {
        const uint8_t *tag = rd_bytes(&r, 1);
        if (!tag || tag[0] != 0) VFAIL("unexpected moe tag");
        if (r.off != r.n) VFAIL("trailing bytes");
    }
    if (rank != PRL_RANK) VFAIL("rank %llu unsupported", (unsigned long long)rank);
    if (k % 64 || k < 1024 || k < 16 * rank || k > 4 * rank * rank || k > 65536) VFAIL("bad k");
    if (m > (1u << 24) || n > (1u << 24)) VFAIL("bad m/n");
    if (pa.total != (m * k + PRL_CHUNK - 1) / PRL_CHUNK) VFAIL("A tree size mismatch");
    if (pb.total != (n * k + PRL_CHUNK - 1) / PRL_CHUNK) VFAIL("B tree size mismatch");
    size_t h = pa.nrows, w = pb.nrows;
    if (h * w < 32 || h * w > 256 || h % 2 || w % 2) VFAIL("bad tile %zux%zu", h, w);
    // patterns from row indices
    uint32_t rel[4096];
    prl_pattern prow, pcol;
    uint64_t t_rows = pa.rows[0], t_cols = pb.rows[0];
    for (size_t i = 0; i < h; i++) { if (pa.rows[i] < t_rows) VFAIL("rows unsorted"); rel[i] = (uint32_t)(pa.rows[i] - t_rows); }
    if (prl_pattern_from_list(&prow, rel, (int)h)) VFAIL("rows not periodic");
    for (size_t i = 0; i < w; i++) { if (pb.rows[i] < t_cols) VFAIL("cols unsorted"); rel[i] = (uint32_t)(pb.rows[i] - t_cols); }
    if (prl_pattern_from_list(&pcol, rel, (int)w)) VFAIL("cols not periodic");
    if (!(t_rows + prl_pattern_max(&prow) < m)) VFAIL("rows out of range");
    if (!(t_cols + prl_pattern_max(&pcol) < n)) VFAIL("cols out of range");
    uint8_t cfg[PRL_CONFIG_LEN];
    config_bytes((uint32_t)k, (uint16_t)rank, &prow, &pcol, cfg);
    uint8_t buf[PRL_HEADER_LEN + PRL_CONFIG_LEN], job_key[32];
    memcpy(buf, header, PRL_HEADER_LEN);
    memcpy(buf + PRL_HEADER_LEN, cfg, PRL_CONFIG_LEN);
    prl_hash(buf, sizeof(buf), job_key);
    // merkle roots
    uint8_t ra[32], rb[32];
    if (mproof_root(&pa, job_key, ra) || memcmp(ra, pa.root, 32)) VFAIL("A merkle root mismatch");
    if (mproof_root(&pb, job_key, rb) || memcmp(rb, pb.root, 32)) VFAIL("B merkle root mismatch");
    // strips
    strips = (int8_t *)malloc((h + w) * 2 * k);
    int8_t *sa = strips, *sb = strips + h * k, *noise = strips + (h + w) * k;
    for (size_t i = 0; i < h; i++) if (mproof_extract(&pa, pa.rows[i] * k, k, sa + i * k)) VFAIL("A strip missing");
    for (size_t i = 0; i < w; i++) if (mproof_extract(&pb, pb.rows[i] * k, k, sb + i * k)) VFAIL("B strip missing");
    for (size_t i = 0; i < (h + w) * k; i++) if (strips[i] < -64 || strips[i] > 64) VFAIL("value out of range");
    // seeds + noise
    uint8_t bseed[32], aseed[32];
    prl_seed_b(job_key, pb.root, (uint32_t)n, salted, bseed);
    prl_seed_a(bseed, pa.root, (uint32_t)m, salted, aseed);
    {
        uint8_t *perm = (uint8_t *)malloc(2 * k);
        int8_t uni[PRL_RANK];
        prl_noise_perm(PRL_LABEL_A, aseed, (int)k, perm);
        for (size_t i = 0; i < h; i++) {
            prl_noise_uniform_row(PRL_LABEL_A, aseed, (uint32_t)pa.rows[i], uni);
            prl_noise_row(uni, perm, (int)k, noise + i * k);
            for (size_t l = 0; l < k; l++) sa[i * k + l] = (int8_t)(sa[i * k + l] + noise[i * k + l]);
        }
        prl_noise_perm(PRL_LABEL_B, bseed, (int)k, perm);
        for (size_t i = 0; i < w; i++) {
            prl_noise_uniform_row(PRL_LABEL_B, bseed, (uint32_t)pb.rows[i], uni);
            prl_noise_row(uni, perm, (int)k, noise + i * k);
            for (size_t l = 0; l < k; l++) sb[i * k + l] = (int8_t)(sb[i * k + l] + noise[i * k + l]);
        }
        free(perm);
    }
    {
        const int8_t *ap[2048], *bp[2048];
        for (size_t i = 0; i < h; i++) ap[i] = sa + i * k;
        for (size_t i = 0; i < w; i++) bp[i] = sb + i * k;
        uint32_t jp[16];
        prl_jackpot(ap, (int)h, bp, (int)w, (int)k, (int)rank, jp);
        uint8_t hh[32];
        prl_jackpot_hash(jp, aseed, hh);
        if (hash_out) memcpy(hash_out, hh, 32);
        uint32_t dpl = (uint32_t)(k - k % rank);
        uint32_t factor = (uint32_t)(h * w) * (dpl / (uint32_t)rank) * 128u;
        u256 bound, hv;
        if (prl_verify_raw_target) bound = *base_target;
        else u256_mul_small(&bound, base_target, factor);
        u256_from_le_bytes(&hv, hh);
        if (u256_cmp(&hv, &bound) > 0) VFAIL("jackpot above target");
    }
    ret = 0;
    if (errlen) err[0] = 0;
done:
    mproof_free(&pa);
    mproof_free(&pb);
    free(strips);
    return ret;
}

// =====================================================================================
// base64
// =====================================================================================
char *prl_base64(const uint8_t *d, size_t n) {
    static const char *T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t ol = 4 * ((n + 2) / 3);
    char *o = (char *)malloc(ol + 1);
    if (!o) return NULL;
    size_t j = 0;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t)d[i] << 16;
        if (i + 1 < n) v |= (uint32_t)d[i + 1] << 8;
        if (i + 2 < n) v |= d[i + 2];
        o[j++] = T[(v >> 18) & 63];
        o[j++] = T[(v >> 12) & 63];
        o[j++] = (i + 1 < n) ? T[(v >> 6) & 63] : '=';
        o[j++] = (i + 2 < n) ? T[v & 63] : '=';
    }
    o[j] = 0;
    return o;
}
