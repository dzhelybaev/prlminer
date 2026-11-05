// ejjjminer — shared mining state: jobs, work preparation, nonces, shares, stats.
#pragma once
#include "pearl.h"
#include <pthread.h>
#include <stdatomic.h>

#define MAX_DEV   32
#define MAX_POOLS 8
#define PRL_VERSION "1.0.9"
#define PRL_AGENT   "ejjjminer/" PRL_VERSION

typedef struct {
    int m;
    prl_geom geom;
    prl_tree tree_a;      // A_base[0 : m*k], chunk 0 = base (nonce path recomputed per attempt)
} work_geom;

// Expensive per-header context (Merkle trees). Shared by jobs with the same header.
typedef struct work {
    atomic_int refs;
    uint8_t header[PRL_HEADER_LEN];
    bool salted;
    uint8_t job_key[32];
    prl_tree tree_b;
    uint8_t b_seed[32];
    int ngeom;
    work_geom g[MAX_DEV];
} work_t;

typedef struct job {
    atomic_int refs;
    uint64_t seq;
    char id[256];
    uint32_t height;
    u256 target;          // as sent by the pool
    u256 bound;           // jackpot hash bound used by the GPU
    uint64_t conn_gen;    // stratum connection generation
    work_t *work;
} job_t;

typedef struct {
    char tag[8];          // "GPU0", "CPU1"
    char name[96];
    char busid[16];
    char kernel[32];
    atomic_uint_fast64_t work;      // MACs done (hash-tile work units)
    atomic_uint accepted, rejected, stale, invalid, found;
    double rate;                    // smoothed work/s
    double last_t;
    uint64_t last_work;
    int active;
} dev_stats;

typedef struct {
    // configuration
    int k, n, nt;
    int cert;             // 0 auto (by height), 2 legacy seeds, 3 salted seeds
    int salted_height;    // fork height for salted seeds (auto mode)
    int prep_threads;
    // base matrices (unnoised, values in [-64, 64])
    int max_m;
    uint8_t *A_base;      // max_m x k
    uint8_t *Bt_base;     // n x k
    int ngeoms;
    int geom_m[MAX_DEV];
    // job state
    pthread_mutex_t mu;
    pthread_cond_t cv;
    job_t *cur;
    job_t *pending;       // waiting for preparation
    uint64_t seq;
    atomic_uint_fast64_t nonce_ctr;
    uint64_t nonce_prefix;
    volatile int quit;
    // stats
    int ndev;
    dev_stats dev[MAX_DEV];
    double start_t;
    atomic_uint shares_accepted, shares_rejected, shares_stale, shares_invalid;
    atomic_int consecutive_lowdiff;
    char pool_url[256];
    uint32_t height;
    double diff;
} miner_t;

extern miner_t g_miner;

int miner_init(int k, int n, int nt);
int miner_add_geometry(int m);            // returns geometry index
int miner_start(void);                    // starts prep + share threads
void miner_stop(void);

// stratum -> miner
void miner_on_notify(const char *job_id, const uint8_t header[PRL_HEADER_LEN], const u256 *target,
                     uint32_t height, uint64_t conn_gen);
void miner_on_disconnect(void);
void miner_on_share_result(int dev, int accepted, int code, const char *msg);

// workers
uint64_t miner_seq(void);                 // changes whenever the current job changes or is withdrawn
job_t *miner_wait_job(double timeout_s);  // ref'd current job, or NULL on timeout/quit
void job_release(job_t *j);
uint64_t miner_next_nonce(void);
void miner_candidate(job_t *j, int geom, int dev, uint64_t nonce, uint32_t t_rows, uint32_t t_cols);
void miner_attempt_chunk0(const work_t *w, int geom, uint64_t nonce, uint8_t chunk0[PRL_CHUNK],
                          uint8_t a_seed[32]);
const char *miner_cert_name(void);
