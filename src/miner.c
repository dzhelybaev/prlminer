// prlminer — job preparation, nonce allocation, share building/verification, stats.
#include "miner.h"
#include "log.h"
#include "stratum.h"
#include "backend.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

miner_t g_miner;

typedef struct {
    int valid;
    char id[256];
    uint8_t header[PRL_HEADER_LEN];
    u256 target;
    uint32_t height;
    uint64_t conn_gen;
} pending_t;

static pending_t g_pending;
static work_t *g_last_work;          // most recent prepared work (for reuse)
static pthread_t g_prep_th, g_share_th;
static int g_threads_started;

// share queue
typedef struct {
    job_t *job;
    int geom, dev;
    uint64_t nonce;
    uint32_t tr, tc;
} cand_t;
#define CQ_CAP 512
static cand_t g_cq[CQ_CAP];
static int g_cq_head, g_cq_len;
static pthread_mutex_t g_cq_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cq_cv = PTHREAD_COND_INITIALIZER;

// ------------------------------------------------------------------ helpers
static uint64_t rand64(void) {
    uint64_t v = 0;
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        if (read(fd, &v, sizeof v) != sizeof v) v = 0;
        close(fd);
    }
    if (!v) v = (uint64_t)time(NULL) * 0x9E3779B97F4A7C15ULL ^ (uint64_t)getpid();
    return v;
}

static work_t *work_ref(work_t *w) {
    if (w) atomic_fetch_add(&w->refs, 1);
    return w;
}

static void work_release(work_t *w) {
    if (!w) return;
    if (atomic_fetch_sub(&w->refs, 1) != 1) return;
    prl_tree_free(&w->tree_b);
    for (int i = 0; i < w->ngeom; i++) prl_tree_free(&w->g[i].tree_a);
    free(w);
}

void job_release(job_t *j) {
    if (!j) return;
    if (atomic_fetch_sub(&j->refs, 1) != 1) return;
    work_release(j->work);
    free(j);
}

static job_t *job_ref(job_t *j) {
    if (j) atomic_fetch_add(&j->refs, 1);
    return j;
}

const char *miner_cert_name(void) {
    miner_t *M = &g_miner;
    if (M->cert == 2) return "v2 (legacy seeds)";
    if (M->cert == 3) return "v3 (salted seeds)";
    return "auto";
}

// ------------------------------------------------------------------ init
int miner_init(int k, int n, int nt) {
    miner_t *M = &g_miner;
    memset(M, 0, sizeof(*M));
    pthread_mutex_init(&M->mu, NULL);
    pthread_cond_init(&M->cv, NULL);
    M->k = k; M->n = n; M->nt = nt;
    M->salted_height = 99000;
    M->prep_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (M->prep_threads < 1) M->prep_threads = 1;
    M->nonce_prefix = rand64() & 0xffffffffull;
    M->start_t = now_sec();
    prl_geom g;
    int rc = prl_geom_init(&g, 16, n, k, nt);
    if (rc) { LOGE("invalid geometry n=%d k=%d nt=%d (%d)", n, k, nt, rc); return -1; }
    return 0;
}

int miner_add_geometry(int m) {
    miner_t *M = &g_miner;
    for (int i = 0; i < M->ngeoms; i++)
        if (M->geom_m[i] == m) return i;
    if (M->ngeoms >= MAX_DEV) return -1;
    prl_geom g;
    if (prl_geom_init(&g, m, M->n, M->k, M->nt)) return -1;
    M->geom_m[M->ngeoms] = m;
    if (m > M->max_m) M->max_m = m;
    return M->ngeoms++;
}

// ------------------------------------------------------------------ work preparation
static work_t *build_work(const uint8_t header[PRL_HEADER_LEN], bool salted) {
    miner_t *M = &g_miner;
    double t0 = now_sec();
    work_t *w = calloc(1, sizeof(work_t));
    if (!w) return NULL;
    atomic_init(&w->refs, 1);
    memcpy(w->header, header, PRL_HEADER_LEN);
    w->salted = salted;
    prl_geom g0;
    prl_geom_init(&g0, M->geom_m[0], M->n, M->k, M->nt);
    prl_job_key(header, &g0, w->job_key);
    if (prl_tree_build(&w->tree_b, M->Bt_base, (size_t)M->n * M->k, w->job_key, M->prep_threads)) goto fail;
    prl_seed_b(w->job_key, prl_tree_root(&w->tree_b), (uint32_t)M->n, salted, w->b_seed);
    for (int i = 0; i < M->ngeoms; i++) {
        w->g[i].m = M->geom_m[i];
        prl_geom_init(&w->g[i].geom, M->geom_m[i], M->n, M->k, M->nt);
        if (prl_tree_build(&w->g[i].tree_a, M->A_base, (size_t)M->geom_m[i] * M->k, w->job_key, M->prep_threads))
            goto fail;
        w->ngeom = i + 1;
    }
    LOGD("work prepared in %.0f ms", (now_sec() - t0) * 1e3);
    return w;
fail:
    LOGE("failed to build Merkle trees (out of memory?)");
    work_release(w);
    return NULL;
}

static void compute_bound(job_t *j) {
    miner_t *M = &g_miner;
    if (prl_verify_raw_target) { j->bound = j->target; return; }
    prl_geom g;
    prl_geom_init(&g, M->geom_m[0], M->n, M->k, M->nt);
    if (!u256_mul_small(&j->bound, &j->target, g.factor)) LOGW("share target too easy, saturated");
}

static bool want_salted(uint32_t height) {
    miner_t *M = &g_miner;
    if (M->cert == 2) return false;
    if (M->cert == 3) return true;
    return height >= (uint32_t)M->salted_height;
}

static void *prep_main(void *arg) {
    (void)arg;
    miner_t *M = &g_miner;
    pthread_mutex_lock(&M->mu);
    while (!M->quit) {
        if (!g_pending.valid) { pthread_cond_wait(&M->cv, &M->mu); continue; }
        pending_t p = g_pending;
        g_pending.valid = 0;
        pthread_mutex_unlock(&M->mu);

        bool salted = want_salted(p.height);
        work_t *w = NULL;
        if (g_last_work && g_last_work->salted == salted && !memcmp(g_last_work->header, p.header, PRL_HEADER_LEN))
            w = work_ref(g_last_work);
        else {
            w = build_work(p.header, salted);
            if (w) { work_release(g_last_work); g_last_work = work_ref(w); }
        }
        job_t *j = NULL;
        if (w) {
            j = calloc(1, sizeof(job_t));
            atomic_init(&j->refs, 1);
            snprintf(j->id, sizeof j->id, "%s", p.id);
            j->height = p.height;
            j->target = p.target;
            j->conn_gen = p.conn_gen;
            j->work = w;
            compute_bound(j);
        }
        pthread_mutex_lock(&M->mu);
        if (j) {
            if (stratum_conn_gen() != j->conn_gen) {   // connection dropped meanwhile
                job_release(j);
                continue;
            }
            j->seq = __atomic_add_fetch(&M->seq, 1, __ATOMIC_RELEASE);
            job_t *old = M->cur;
            M->cur = j;
            M->height = j->height;
            M->diff = 26959946667150639794667015087019630673637144422540572481103610249216.0 /
                      (u256_to_double(&j->target) + 1);   // 2^224 / target
            pthread_cond_broadcast(&M->cv);
            pthread_mutex_unlock(&M->mu);
            job_release(old);
            char d[32];
            LOGI("job %s height %u diff %s  seeds %s", j->id, j->height, fmt_rate(M->diff, d, sizeof d),
                 salted ? "v3" : "v2");
            pthread_mutex_lock(&M->mu);
        }
    }
    pthread_mutex_unlock(&M->mu);
    return NULL;
}

void miner_on_notify(const char *job_id, const uint8_t header[PRL_HEADER_LEN], const u256 *target, uint32_t height,
                     uint64_t conn_gen) {
    miner_t *M = &g_miner;
    pthread_mutex_lock(&M->mu);
    // If only the target / id changed (same header), publish immediately without preparation.
    bool salted = want_salted(height);
    if (M->cur && M->cur->work->salted == salted && !memcmp(M->cur->work->header, header, PRL_HEADER_LEN) &&
        !g_pending.valid) {
        job_t *j = calloc(1, sizeof(job_t));
        atomic_init(&j->refs, 1);
        snprintf(j->id, sizeof j->id, "%s", job_id);
        j->height = height;
        j->target = *target;
        j->conn_gen = conn_gen;
        j->work = work_ref(M->cur->work);
        compute_bound(j);
        j->seq = __atomic_add_fetch(&M->seq, 1, __ATOMIC_RELEASE);
        job_t *old = M->cur;
        M->cur = j;
        M->diff = 26959946667150639794667015087019630673637144422540572481103610249216.0 /
                  (u256_to_double(target) + 1);
        pthread_cond_broadcast(&M->cv);
        pthread_mutex_unlock(&M->mu);
        job_release(old);
        LOGD("job %s (same header)", job_id);
        return;
    }
    g_pending.valid = 1;
    snprintf(g_pending.id, sizeof g_pending.id, "%s", job_id);
    memcpy(g_pending.header, header, PRL_HEADER_LEN);
    g_pending.target = *target;
    g_pending.height = height;
    g_pending.conn_gen = conn_gen;
    // a new block height invalidates the current job right away
    if (M->cur && M->cur->height != height) {
        job_t *old = M->cur;
        M->cur = NULL;
        __atomic_add_fetch(&M->seq, 1, __ATOMIC_RELEASE);
        pthread_cond_broadcast(&M->cv);
        pthread_mutex_unlock(&M->mu);
        job_release(old);
        return;
    }
    pthread_cond_broadcast(&M->cv);
    pthread_mutex_unlock(&M->mu);
}

void miner_on_disconnect(void) {
    miner_t *M = &g_miner;
    pthread_mutex_lock(&M->mu);
    job_t *old = M->cur;
    M->cur = NULL;
    g_pending.valid = 0;
    __atomic_add_fetch(&M->seq, 1, __ATOMIC_RELEASE);
    pthread_cond_broadcast(&M->cv);
    pthread_mutex_unlock(&M->mu);
    job_release(old);
}

uint64_t miner_seq(void) { return __atomic_load_n(&g_miner.seq, __ATOMIC_ACQUIRE); }

job_t *miner_wait_job(double timeout_s) {
    miner_t *M = &g_miner;
    pthread_mutex_lock(&M->mu);
    double until = now_sec() + timeout_s;
    while (!M->quit && M->cur == NULL) {
        double left = until - now_sec();
        if (left <= 0) break;
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        double t = ts.tv_sec + ts.tv_nsec * 1e-9 + (left > 0.5 ? 0.5 : left);
        ts.tv_sec = (time_t)t;
        ts.tv_nsec = (long)((t - ts.tv_sec) * 1e9);
        pthread_cond_timedwait(&M->cv, &M->mu, &ts);
    }
    job_t *j = M->quit ? NULL : job_ref(M->cur);
    pthread_mutex_unlock(&M->mu);
    return j;
}

uint64_t miner_next_nonce(void) {
    uint64_t c = atomic_fetch_add(&g_miner.nonce_ctr, 1);
    return (g_miner.nonce_prefix << 32) + c;
}

void miner_attempt_chunk0(const work_t *w, int geom, uint64_t nonce, uint8_t chunk0[PRL_CHUNK], uint8_t a_seed[32]) {
    miner_t *M = &g_miner;
    memcpy(chunk0, M->A_base, PRL_CHUNK);
    prl_apply_nonce((int8_t *)chunk0, nonce);
    prl_path0 p;
    prl_tree_path0(&w->g[geom].tree_a, chunk0, &p);
    prl_seed_a(w->b_seed, p.node[p.nlayers - 1], (uint32_t)w->g[geom].m, w->salted, a_seed);
}

void miner_candidate(job_t *j, int geom, int dev, uint64_t nonce, uint32_t tr, uint32_t tc) {
    pthread_mutex_lock(&g_cq_mu);
    if (g_cq_len < CQ_CAP) {
        cand_t *c = &g_cq[(g_cq_head + g_cq_len) % CQ_CAP];
        c->job = job_ref(j);
        c->geom = geom; c->dev = dev; c->nonce = nonce; c->tr = tr; c->tc = tc;
        g_cq_len++;
        pthread_cond_signal(&g_cq_cv);
    } else LOGW("share queue full, dropping a share");
    pthread_mutex_unlock(&g_cq_mu);
}

// ------------------------------------------------------------------ share thread
static void process_candidate(cand_t *c) {
    miner_t *M = &g_miner;
    job_t *j = c->job;
    work_t *w = j->work;
    work_geom *wg = &w->g[c->geom];
    atomic_fetch_add(&M->dev[c->dev].found, 1);
    uint8_t chunk0[PRL_CHUNK];
    memcpy(chunk0, M->A_base, PRL_CHUNK);
    prl_apply_nonce((int8_t *)chunk0, c->nonce);
    prl_path0 p;
    prl_tree_path0(&wg->tree_a, chunk0, &p);
    prl_bytes pr;
    if (prl_build_proof(&wg->geom, &wg->tree_a, &p, M->A_base, chunk0, &w->tree_b, M->Bt_base, c->tr, c->tc, &pr)) {
        LOGE("%s: failed to build proof", M->dev[c->dev].tag);
        return;
    }
    char err[256];
    uint8_t hh[32];
    int rc = prl_verify_proof(pr.buf, pr.len, w->header, w->salted, &j->target, hh, err, sizeof err);
    if (rc) {
        atomic_fetch_add(&M->dev[c->dev].invalid, 1);
        atomic_fetch_add(&M->shares_invalid, 1);
        LOGW("%s: share failed local verification (%s) — not submitted", M->dev[c->dev].tag, err);
        prl_bytes_free(&pr);
        return;
    }
    size_t blen = pr.len;
    if (g_benchmark) {
        atomic_fetch_add(&M->dev[c->dev].accepted, 1);
        unsigned a = atomic_fetch_add(&M->shares_accepted, 1) + 1;
        LOGI("%s: share verified locally (benchmark, not submitted) [%u]", M->dev[c->dev].tag, a);
        prl_bytes_free(&pr);
        return;
    }
    if (stratum_conn_gen() != j->conn_gen) {
        atomic_fetch_add(&M->dev[c->dev].stale, 1);
        atomic_fetch_add(&M->shares_stale, 1);
        LOGW("%s: share for a previous connection dropped", M->dev[c->dev].tag);
        prl_bytes_free(&pr);
        return;
    }
    pthread_mutex_lock(&M->mu);
    bool stale = !M->cur || M->cur->height != j->height;
    pthread_mutex_unlock(&M->mu);
    if (stale) {
        atomic_fetch_add(&M->dev[c->dev].stale, 1);
        atomic_fetch_add(&M->shares_stale, 1);
        LOGI("%s: share for old block %u dropped", M->dev[c->dev].tag, j->height);
        prl_bytes_free(&pr);
        return;
    }
    LOGI("%s: share found (job %s, tile %u/%u, %zu bytes), submitting", M->dev[c->dev].tag, j->id, c->tr, c->tc, blen);
    stratum_submit(j->id, pr.buf, pr.len, c->dev);
    prl_bytes_free(&pr);
}

static void *share_main(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_cq_mu);
        while (!g_cq_len && !g_miner.quit) pthread_cond_wait(&g_cq_cv, &g_cq_mu);
        if (!g_cq_len && g_miner.quit) { pthread_mutex_unlock(&g_cq_mu); break; }
        cand_t c = g_cq[g_cq_head];
        g_cq_head = (g_cq_head + 1) % CQ_CAP;
        g_cq_len--;
        pthread_mutex_unlock(&g_cq_mu);
        process_candidate(&c);
        job_release(c.job);
    }
    return NULL;
}

void miner_on_share_result(int dev, int accepted, int code, const char *msg) {
    miner_t *M = &g_miner;
    if (dev < 0 || dev >= MAX_DEV) dev = 0;
    if (accepted) {
        atomic_fetch_add(&M->dev[dev].accepted, 1);
        unsigned a = atomic_fetch_add(&M->shares_accepted, 1) + 1;
        LOGI("%s: share accepted (%u/%u)", M->dev[dev].tag, a, a + atomic_load(&M->shares_rejected));
    } else if (code == 21) {
        atomic_fetch_add(&M->dev[dev].stale, 1);
        atomic_fetch_add(&M->shares_stale, 1);
        LOGW("%s: share stale (%s)", M->dev[dev].tag, msg ? msg : "");
    } else {
        atomic_fetch_add(&M->dev[dev].rejected, 1);
        unsigned rej = atomic_fetch_add(&M->shares_rejected, 1) + 1;
        if (rej == 5 && atomic_load(&M->shares_accepted) == 0 && (code == 23 || code == 26))
            LOGE("the pool rejected the first %u shares (code %d). Likely causes: wrong noise-seed version for "
                 "this height (try --cert 2 or --cert 3) or a pool that compares the target without the tile "
                 "factor (try --share-target raw). Please report the pool name.", rej, code);
        LOGW("%s: share rejected: code %d %s", M->dev[dev].tag, code, msg ? msg : "");
    }
}

// ------------------------------------------------------------------ start / stop
int miner_start(void) {
    miner_t *M = &g_miner;
    if (!M->ngeoms) { LOGE("no mining devices"); return -1; }
    size_t asz = (size_t)M->max_m * M->k, bsz = (size_t)M->n * M->k;
    M->A_base = malloc(asz);
    M->Bt_base = malloc(bsz);
    if (!M->A_base || !M->Bt_base) { LOGE("out of memory for base matrices"); return -1; }
    // Base matrices are all zero (only the nonce nibbles in A differ). The noise makes the
    // GEMM operands random anyway, and zero leaves make PlainProofs compress ~100x.
    memset(M->A_base, 0, asz);
    memset(M->Bt_base, 0, bsz);
    if (pthread_create(&g_prep_th, NULL, prep_main, NULL)) return -1;
    if (pthread_create(&g_share_th, NULL, share_main, NULL)) return -1;
    g_threads_started = 1;
    return 0;
}

void miner_stop(void) {
    miner_t *M = &g_miner;
    pthread_mutex_lock(&M->mu);
    M->quit = 1;
    pthread_cond_broadcast(&M->cv);
    pthread_mutex_unlock(&M->mu);
    pthread_mutex_lock(&g_cq_mu);
    pthread_cond_broadcast(&g_cq_cv);
    pthread_mutex_unlock(&g_cq_mu);
    if (g_threads_started) {
        pthread_join(g_prep_th, NULL);
        pthread_join(g_share_th, NULL);
        g_threads_started = 0;
    }
}
