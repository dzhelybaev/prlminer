// ejjjminer — CPU reference backend. Very slow; used for testing the full pipeline
// (pool protocol, proofs) on machines without a GPU (--cpu).
#include "backend.h"
#include "log.h"
#include "miner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct cpu_dev {
    int idx, m, geom, threads;
    pthread_t th;
};

cpu_dev *cpu_prepare(int stat_idx, int threads, int rows) {
    cpu_dev *d = calloc(1, sizeof(cpu_dev));
    d->idx = stat_idx;
    d->threads = threads;
    d->m = rows > 0 ? (rows + 15) / 16 * 16 : 16;
    d->geom = miner_add_geometry(d->m);
    if (d->geom < 0) { free(d); return NULL; }
    dev_stats *ds = &g_miner.dev[stat_idx];
    snprintf(ds->tag, sizeof ds->tag, "CPU%d", stat_idx);
    snprintf(ds->name, sizeof ds->name, "CPU");
    snprintf(ds->busid, sizeof ds->busid, "cpu");
    snprintf(ds->kernel, sizeof ds->kernel, "cpu-nt%d", g_miner.nt);
    LOGI("CPU%d: reference backend, matrix %dx%dx%d (slow, for testing)", stat_idx, d->m, g_miner.n, g_miner.k);
    return d;
}

static void noise_matrix(const uint8_t *base, int rows, int k, const uint8_t *label, const uint8_t seed[32],
                         const uint8_t *chunk0, int8_t *out) {
    uint8_t *perm = malloc(2 * (size_t)k);
    int8_t uni[PRL_RANK], *nz = malloc(k);
    prl_noise_perm(label, seed, k, perm);
    for (int r = 0; r < rows; r++) {
        prl_noise_uniform_row(label, seed, (uint32_t)r, uni);
        prl_noise_row(uni, perm, k, nz);
        const int8_t *src = (const int8_t *)base + (size_t)r * k;
        for (int l = 0; l < k; l++) {
            int8_t b = (r == 0 && chunk0 && l < PRL_CHUNK) ? (int8_t)chunk0[l] : src[l];
            out[(size_t)r * k + l] = (int8_t)(b + nz[l]);
        }
    }
    free(perm);
    free(nz);
}

static void *cpu_main(void *arg) {
    cpu_dev *d = arg;
    miner_t *M = &g_miner;
    const int m = d->m, n = M->n, k = M->k;
    int8_t *An = malloc((size_t)m * k), *Bn = malloc((size_t)n * k);
    uint8_t last_bseed[32];
    int have_b = 0;
    prl_geom g;
    prl_geom_init(&g, m, n, k, M->nt);
    M->dev[d->idx].active = 1;
    while (!M->quit) {
        job_t *job = miner_wait_job(1.0);
        if (!job) continue;
        uint64_t seq = job->seq;
        if (!have_b || memcmp(last_bseed, job->work->b_seed, 32)) {
            noise_matrix(M->Bt_base, n, k, PRL_LABEL_B, job->work->b_seed, NULL, Bn);
            memcpy(last_bseed, job->work->b_seed, 32);
            have_b = 1;
        }
        while (!M->quit && miner_seq() == seq) {
            uint64_t nonce = miner_next_nonce();
            uint8_t chunk0[PRL_CHUNK], aseed[32];
            miner_attempt_chunk0(job->work, d->geom, nonce, chunk0, aseed);
            noise_matrix(M->A_base, m, k, PRL_LABEL_A, aseed, chunk0, An);
            for (uint32_t tr0 = 0; tr0 < (uint32_t)m && miner_seq() == seq; tr0 += 16)
                for (uint32_t gg = 0; gg < 8; gg++)
                    for (uint32_t c0 = 0; c0 < (uint32_t)n; c0 += 8 * M->nt)
                        for (uint32_t qq = 0; qq < 4; qq++) {
                            uint32_t tr = tr0 + gg, tc = c0 + 2 * qq, rows[2], cols[64], jp[16];
                            prl_tile_rows(&g, tr, rows);
                            prl_tile_cols(&g, tc, cols);
                            const int8_t *ap[2] = {An + (size_t)rows[0] * k, An + (size_t)rows[1] * k}, *bp[64];
                            for (int j = 0; j < g.w; j++) bp[j] = Bn + (size_t)cols[j] * k;
                            prl_jackpot(ap, 2, bp, g.w, k, PRL_RANK, jp);
                            uint8_t hh[32];
                            prl_jackpot_hash(jp, aseed, hh);
                            u256 hv;
                            u256_from_le_bytes(&hv, hh);
                            if (u256_cmp(&hv, &job->bound) <= 0) miner_candidate(job, d->geom, d->idx, nonce, tr, tc);
                        }
            atomic_fetch_add(&M->dev[d->idx].work, (uint64_t)m * n * k);
        }
        job_release(job);
    }
    free(An);
    free(Bn);
    return NULL;
}

int cpu_launch(cpu_dev *d) { return pthread_create(&d->th, NULL, cpu_main, d); }
