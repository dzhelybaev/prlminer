// Thin C ABI over the core for Python cross-checks (tests only).
#include "../src/pearl.h"
#include <string.h>

void py_keyed_hash(const uint8_t *key, const uint8_t *data, size_t len, uint8_t *out) { prl_keyed_hash(key, data, len, out); }
void py_hash(const uint8_t *data, size_t len, uint8_t *out) { prl_hash(data, len, out); }

// noise rows for A (label A) or B (label B): out is nrows x k
void py_noise_rows(int which, const uint8_t *key, int k, const uint32_t *rows, int nrows, int8_t *out) {
    const uint8_t *label = which ? PRL_LABEL_B : PRL_LABEL_A;
    uint8_t perm[2 * 65536];
    int8_t uni[PRL_RANK];
    prl_noise_perm(label, key, k, perm);
    for (int i = 0; i < nrows; i++) {
        prl_noise_uniform_row(label, key, rows[i], uni);
        prl_noise_row(uni, perm, k, out + (size_t)i * k);
    }
}

void py_uniform_row(int which, const uint8_t *key, uint32_t row, int8_t *out) {
    prl_noise_uniform_row(which ? PRL_LABEL_B : PRL_LABEL_A, key, row, out);
}

// jackpot for h rows (h x k) and w cols (w x k), already noised
void py_jackpot(const int8_t *a, int h, const int8_t *b, int w, int k, uint32_t *jp) {
    const int8_t *ap[256], *bp[256];
    for (int i = 0; i < h; i++) ap[i] = a + (size_t)i * k;
    for (int i = 0; i < w; i++) bp[i] = b + (size_t)i * k;
    prl_jackpot(ap, h, bp, w, k, PRL_RANK, jp);
}
