// ejjjminer — Pearl (PRL) protocol core: patterns, commitments, Merkle trees,
// noise, jackpot reference, PlainProof build/serialize/verify.
// Ported from the open-source reference implementation
// (github.com/pearl-research-labs/pearl, zk-pow + pearl-blake3 + miner-base).
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define PRL_RANK          128   // noise rank (>=128 required by rank-penalty fork)
#define PRL_JACKPOT_WORDS 16
#define PRL_LROT          13
#define PRL_CHUNK         1024
#define PRL_HEADER_LEN    76
#define PRL_CONFIG_LEN    52

// ---------------------------------------------------------------- u256 (LE limbs)
typedef struct { uint32_t w[8]; } u256;
void u256_from_be_hex(u256 *out, const char *hex);           // 64 hex chars, big-endian
void u256_from_le_bytes(u256 *out, const uint8_t b[32]);
void u256_to_be_hex(const u256 *v, char out[65]);
int  u256_cmp(const u256 *a, const u256 *b);
bool u256_mul_small(u256 *out, const u256 *a, uint32_t f);    // false on overflow (saturates)
void u256_from_nbits(u256 *out, uint32_t nbits);
double u256_to_double(const u256 *v);
void u256_from_double(u256 *out, double v);   // floor(v), v >= 0

// ---------------------------------------------------------------- patterns
typedef struct { uint32_t st[3], ln[3]; } prl_pattern;
int      prl_pattern_from_list(prl_pattern *p, const uint32_t *idx, int n);  // 0 ok
void     prl_pattern_to_bytes(const prl_pattern *p, uint8_t out[6]);
int      prl_pattern_to_list(const prl_pattern *p, uint32_t *out, int cap);
uint32_t prl_pattern_period(const prl_pattern *p);
uint32_t prl_pattern_size(const prl_pattern *p);
uint32_t prl_pattern_max(const prl_pattern *p);
bool     prl_pattern_offset_valid(const prl_pattern *p, uint32_t off);

// ---------------------------------------------------------------- mining geometry
// Hash tile owned by one mma lane: rows {t, t+8}, cols {c + 8j + e : j<nt, e<2}.
// nt=32 is the pattern used by the official miner (2x64 tile over 256 columns).
typedef struct {
    int m, n, k;          // matrix dims; A is m x k, B^T is n x k
    int nt;               // number of n8 sub-tiles per hash tile (8, 16 or 32)
    prl_pattern rows, cols;
    uint8_t config_bytes[PRL_CONFIG_LEN];
    int h, w;             // hash tile size (2, 2*nt)
    uint32_t factor;      // difficulty adjustment factor h*w*(k/r)*128
} prl_geom;

int  prl_geom_init(prl_geom *g, int m, int n, int k, int nt);       // 0 ok, else error
void prl_tile_rows(const prl_geom *g, uint32_t t_rows, uint32_t *rows); // h entries
void prl_tile_cols(const prl_geom *g, uint32_t t_cols, uint32_t *cols); // w entries

// ---------------------------------------------------------------- hashing helpers
void prl_hash(const void *data, size_t len, uint8_t out[32]);
void prl_keyed_hash(const uint8_t key[32], const void *data, size_t len, uint8_t out[32]);

// ---------------------------------------------------------------- Merkle tree
typedef struct {
    uint8_t key[32];
    int nlayers;               // layer 0 = chunk CVs ... last = [root]
    size_t len[48];
    uint8_t *cv[48];           // len[i]*32 bytes each
    size_t data_len;           // padded length in bytes
} prl_tree;

int  prl_tree_build(prl_tree *t, const uint8_t *data, size_t len, const uint8_t key[32], int nthreads);
void prl_tree_free(prl_tree *t);
const uint8_t *prl_tree_root(const prl_tree *t);

// Path of chunk 0 when its contents change (the nonce lives in chunk 0 of A).
typedef struct {
    int nlayers;
    uint8_t node[48][32];      // node[l] = new value of layer l index 0 (last = root)
} prl_path0;
void prl_tree_path0(const prl_tree *t, const uint8_t chunk0[PRL_CHUNK], prl_path0 *out);

// ---------------------------------------------------------------- job
typedef struct {
    uint8_t header[PRL_HEADER_LEN];
    uint32_t nbits;
    bool salted;               // certificate V3 (salted noise seeds)
    uint8_t job_key[32];
} prl_job;
void prl_job_key(const uint8_t header[PRL_HEADER_LEN], const prl_geom *g, uint8_t out[32]);

// seeds: b_seed = H(job_key||bind_b(hash_b)), a_seed = H(b_seed||bind_a(hash_a))
void prl_seed_b(const uint8_t job_key[32], const uint8_t hash_b[32], uint32_t n, bool salted, uint8_t out[32]);
void prl_seed_a(const uint8_t b_seed[32], const uint8_t hash_a[32], uint32_t m, bool salted, uint8_t out[32]);

// ---------------------------------------------------------------- noise (CPU)
extern const uint8_t PRL_LABEL_A[32], PRL_LABEL_B[32];
void prl_noise_hash(uint32_t index, const uint8_t label[32], const uint8_t key[32], int slot, uint8_t out[32]);
// uniform rows (row-major rows x rank), only rows listed
void prl_noise_uniform_row(const uint8_t label[32], const uint8_t key[32], uint32_t row, int8_t *out /*rank*/);
// permutation pairs for k lines: out[2*l] = first, out[2*l+1] = second
void prl_noise_perm(const uint8_t label[32], const uint8_t key[32], int k, uint8_t *out);
// noise rows: A'[row] = A[row] + E_AL[row] (x) perm_A
void prl_noise_row(const int8_t *uni_row, const uint8_t *perm, int k, int8_t *out);

// ---------------------------------------------------------------- matrices + nonce
// A row 0, bytes [0,16) hold the nonce as 16 nibbles.
#define PRL_NONCE_BYTES 16
void prl_apply_nonce(int8_t *row0, uint64_t nonce);

// ---------------------------------------------------------------- jackpot
// strips: h rows of A' (noised, k each), w rows of B^T' (noised, k each)
void prl_jackpot(const int8_t *const *a_rows, int h, const int8_t *const *b_cols, int w,
                 int k, int rank, uint32_t jp[PRL_JACKPOT_WORDS]);
void prl_jackpot_hash(const uint32_t jp[PRL_JACKPOT_WORDS], const uint8_t a_seed[32], uint8_t out[32]);

// ---------------------------------------------------------------- PlainProof
typedef struct { uint8_t *buf; size_t len, cap; } prl_bytes;
void prl_bytes_free(prl_bytes *b);

// Build serialized (bincode) PlainProof for tile (t_rows, t_cols).
// A_data/Bt_data: full unnoised base matrices; a_chunk0 (optional) replaces chunk 0
// of A (the chunk holding the nonce) and pa is the matching Merkle path.
int prl_build_proof(const prl_geom *g, const prl_tree *ta, const prl_path0 *pa,
                    const uint8_t *A_data, const uint8_t *a_chunk0, const prl_tree *tb,
                    const uint8_t *Bt_data, uint32_t t_rows, uint32_t t_cols, prl_bytes *out);

// Independent verifier: parses bytes, recomputes Merkle roots from leaves and
// siblings, re-derives seeds/noise/jackpot, and checks hash <= target*factor.
// Returns 0 if valid; writes reason into err. hash_out optional.
int prl_verify_proof(const uint8_t *proof, size_t len, const uint8_t header[PRL_HEADER_LEN],
                     bool salted, const u256 *base_target, uint8_t hash_out[32], char *err, size_t errlen);
// raw_target: compare the jackpot hash with the target directly (no tile factor)
extern int prl_verify_raw_target;

// base64
char *prl_base64(const uint8_t *data, size_t len);
