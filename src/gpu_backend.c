// ejjjminer — NVIDIA GPU backend (CUDA driver API, PTX JIT-compiled by the driver).
#include "backend.h"
#include "cuda_drv.h"
#include "log.h"
#include "miner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern const char prl_ptx_sm80_noise[], prl_ptx_sm80_mma32[], prl_ptx_sm80_mma16[], prl_ptx_sm80_mma8[],
    prl_ptx_sm80_dp4a32[], prl_ptx_sm80_dp4a16[], prl_ptx_sm80_dp4a8[];
extern const char prl_ptx_sm61_noise[], prl_ptx_sm61_dp4a32[], prl_ptx_sm61_dp4a16[], prl_ptx_sm61_dp4a8[];
extern const char prl_ptx_sm75_mma32[], prl_ptx_sm75_mma16[], prl_ptx_sm75_mma8[];
extern const char prl_ptx_sm80_fmma32[], prl_ptx_sm80_fmma16[], prl_ptx_sm80_fmma8[];
extern const char prl_ptx_sm75_fmma32[], prl_ptx_sm75_fmma16[], prl_ptx_sm75_fmma8[];
extern const char prl_ptx_sm80_v2_16[], prl_ptx_sm80_v2_32[], prl_ptx_sm75_v2_16[], prl_ptx_sm75_v2_32[];

// v2 kernel variants: must match V2_A/V2_B/V2_C in cuda/kernels.cu
typedef struct { const char *name; int mt, threads, bks, stg, jsm, hs, l2, eg; } v2_variant;   // must match kernels.cu
// l2 = 1: A' comes precomputed from k_apply_noise (matrix sized to stay in L2), no generation in the GEMM
static const v2_variant V2_SM80[] = {
    {"v2a", 0, 256, 1, 4, 9, 1, 0, 0}, {"v2b", 0, 256, 2, 2, 9, 1, 0, 0}, {"v2c", 0, 256, 1, 3, 0, 1, 0, 0},
    {"v3a", 1, 512, 1, 3, 9, 1, 0, 0}, {"v3b", 1, 512, 2, 2, 9, 1, 0, 0},
    {"v4a", 2, 512, 1, 3, 9, 2, 0, 0}, {"v4b", 2, 512, 1, 2, 9, 2, 0, 0}, {"v4c", 2, 512, 2, 2, 9, 2, 0, 0},
    {"l2a", 0, 256, 1, 4, 9, 1, 1, 0}, {"l2b", 0, 256, 2, 2, 9, 1, 1, 0}, {"l3b", 1, 512, 2, 2, 9, 1, 1, 0},
    {"l4c", 2, 512, 2, 2, 9, 2, 1, 0},
    {"t2b", 0, 256, 1, 2, 9, 1, 0, 1}, {"t2p", 0, 256, 1, 2, 9, 1, 0, 1}, {"t3b", 1, 512, 1, 2, 9, 1, 0, 1},
    {"t4b", 2, 512, 1, 2, 9, 2, 0, 1}, {"t4p", 2, 512, 1, 2, 9, 2, 0, 1}};
static const v2_variant V2_SM75[] = {
    {"v2a", 0, 256, 1, 2, 1, 1, 0, 0}, {"v2b", 0, 256, 1, 2, 1, 1, 0, 0}, {"v2c", 0, 256, 1, 2, 0, 1, 0, 0},
    {"v3a", 1, 512, 1, 2, 0, 1, 0, 0}, {"v3b", 1, 512, 1, 2, 0, 1, 0, 0},
    {"v4a", 2, 512, 1, 2, 1, 2, 0, 0}, {"v4b", 2, 512, 1, 2, 1, 2, 0, 0}, {"v4c", 2, 512, 1, 2, 0, 2, 0, 0},
    {"l2a", 0, 256, 1, 2, 1, 1, 1, 0}, {"l2b", 0, 256, 1, 2, 1, 1, 1, 0}, {"l3b", 1, 512, 1, 2, 0, 1, 1, 0},
    {"l4c", 2, 512, 1, 2, 0, 2, 1, 0},
    {"t2b", 0, 256, 1, 2, 9, 1, 0, 1}, {"t2p", 0, 256, 1, 2, 9, 1, 0, 1}, {"t3b", 1, 512, 1, 2, 9, 1, 0, 1},
    {"t4b", 2, 512, 1, 2, 9, 2, 0, 1}, {"t4p", 2, 512, 1, 2, 9, 2, 0, 1}};
#define NV2 17
static unsigned v2_smem(const v2_variant *v, int nt) {
    int mt = v->mt ? v->mt : 32 / nt, js = v->jsm < mt ? v->jsm : mt, owners = v->threads / v->hs;
    return (unsigned)(v->stg * v->bks * 12288 + (v->eg ? 0 : 16384) + 64 * js * owners +
                      (v->hs == 2 ? 4 * mt * owners : 0));
}

static const char *ptx_turing_mma(int nt) {
    return nt == 32 ? prl_ptx_sm75_mma32 : nt == 16 ? prl_ptx_sm75_mma16 : prl_ptx_sm75_mma8;
}
static const char *ptx_fused(bool sm80, int nt) {
    if (sm80) return nt == 32 ? prl_ptx_sm80_fmma32 : nt == 16 ? prl_ptx_sm80_fmma16 : prl_ptx_sm80_fmma8;
    return nt == 32 ? prl_ptx_sm75_fmma32 : nt == 16 ? prl_ptx_sm75_fmma16 : prl_ptx_sm75_fmma8;
}

int g_fused_mode = 0;
const char *g_variant = NULL;

static const char *ptx_for(bool sm80, const char *kind) {
    static const struct { const char *k; const char *p80, *p61; } T[] = {
        {"noise", prl_ptx_sm80_noise, prl_ptx_sm61_noise},   {"mma32", prl_ptx_sm80_mma32, NULL},
        {"mma16", prl_ptx_sm80_mma16, NULL},                 {"mma8", prl_ptx_sm80_mma8, NULL},
        {"dp4a32", prl_ptx_sm80_dp4a32, prl_ptx_sm61_dp4a32}, {"dp4a16", prl_ptx_sm80_dp4a16, prl_ptx_sm61_dp4a16},
        {"dp4a8", prl_ptx_sm80_dp4a8, prl_ptx_sm61_dp4a8}};
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++)
        if (!strcmp(T[i].k, kind)) return sm80 ? T[i].p80 : T[i].p61;
    return NULL;
}

#define BM 128
#define BN 256
#define SMEM_MMA (3 * (BM + BN) * 48)
#define SMEM_DP4A_SM61 (2 * (BM + BN) * 48)
#define CAND_CAP 64
#define APPLY_ROWS 16
#define APPLY_SMEM(k) (2 * (k) + APPLY_ROWS * 128)
#define CAND_WORDS (4 + 4 * CAND_CAP)
#define FUSED_SMEM (128 * 128)

volatile int g_benchmark;

struct gpu_dev {
    int idx;
    gpu_info info;
    CUcontext ctx;
    CUstream st, st2, lst;   // st: GEMM, st2: noise (L2 pipeline), lst: launch-stream override
    CUfunction f_uni, f_perm, f_apply, f_mine, f_fused, f_seeds;
    int gpu_seeds;   // k_seeds passed its self-test: batches with seeds derived on the GPU
    unsigned smem, smem_fused;
    bool mma, use_fused;
    struct { CUfunction f; unsigned smem, threads; int l2, eg; char name[16]; } fv[24];   // self-tested variants
    int eg_fused;   // selected kernel reads the ELt table from global memory (uniform rows in ELt layout)
    unsigned threads_fused;
    int m_l2, geom_l2, use_l2;   // L2 kernels: rows sized so A' + B' stay in L2
    int nfv;
    int m, geom;
    pthread_t th;
};

static gpu_dev *g_devs[MAX_DEV];
static int g_ndevs;

#define CK(call)                                                                              \
    do {                                                                                      \
        CUresult _r = (call);                                                                 \
        if (_r) { LOGE("GPU%d: %s failed: %s", d->idx, #call, cu_errstr(_r)); goto fail; }    \
    } while (0)

int gpu_list(gpu_info *out, int cap, char *err, int errlen) {
    if (cuda_load(err, errlen)) return -1;
    int n = 0;
    if (cu.DeviceGetCount(&n)) { snprintf(err, errlen, "cuDeviceGetCount failed"); return -1; }
    if (n > cap) n = cap;
    for (int i = 0; i < n; i++) {
        gpu_info *g = &out[i];
        memset(g, 0, sizeof(*g));
        CUdevice dv;
        cu.DeviceGet(&dv, i);
        g->ordinal = i;
        cu.DeviceGetName(g->name, sizeof g->name, dv);
        int bus = 0, dev = 0, dom = 0;
        cu.DeviceGetAttribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dv);
        cu.DeviceGetAttribute(&dev, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dv);
        cu.DeviceGetAttribute(&dom, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dv);
        g->bus = bus;
        snprintf(g->busid, sizeof g->busid, "%02x:%02x.0", bus, dev);
        cu.DeviceGetAttribute(&g->cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dv);
        cu.DeviceGetAttribute(&g->cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dv);
        cu.DeviceGetAttribute(&g->sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dv);
        size_t mem = 0;
        cu.DeviceTotalMem(&mem, dv);
        g->mem = mem;
    }
    return n;
}

static int load_module(gpu_dev *d, CUmodule *mod, const char *ptx, const char *what) {
    static char elog[8192];
    elog[0] = 0;
    if (!ptx) { LOGE("GPU%d: no PTX for %s", d->idx, what); return -1; }
    int opts[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void *vals[] = {elog, (void *)(size_t)sizeof elog};
    double t0 = now_sec();
    CUresult r = cu.ModuleLoadDataEx(mod, ptx, 2, opts, vals);
    if (r) { LOGE("GPU%d: PTX JIT of %s failed: %s %s", d->idx, what, cu_errstr(r), elog); return -1; }
    LOGD("GPU%d: %s kernels JIT-compiled in %.1f s", d->idx, what, now_sec() - t0);
    return 0;
}

static CUresult launch_t(gpu_dev *d, CUfunction f, unsigned gx, unsigned gy, unsigned threads, unsigned smem,
                         void **args) {
    return cu.LaunchKernel(f, gx, gy, 1, threads, 1, 1, smem, d->lst ? d->lst : d->st, args, NULL);
}
static CUresult launch3(gpu_dev *d, CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned threads,
                       unsigned smem, void **args) {
    return cu.LaunchKernel(f, gx, gy, gz, threads, 1, 1, smem, d->lst ? d->lst : d->st, args, NULL);
}
static CUresult launch(gpu_dev *d, CUfunction f, unsigned gx, unsigned gy, unsigned smem, void **args) {
    return launch_t(d, f, gx, gy, 256, smem, args);
}

// ------------------------------------------------------------------ device-side primitives
typedef struct {
    CUdeviceptr base, uni, pairs, out, seed;
} noise_bufs;

// out = base (+ nonce patch) + noise(label, seed) for `rows` rows of length k
static CUresult run_noise(gpu_dev *d, noise_bufs *b, unsigned label, unsigned rows, unsigned k, uint64_t nonce,
                          unsigned patch, bool apply) {
    CUresult r;
    unsigned nblocks = rows * 4;
    void *a1[] = {&b->seed, &label, &nblocks, &b->uni};
    if ((r = launch(d, d->f_uni, (nblocks + 255) / 256, 1, 0, a1))) return r;
    void *a2[] = {&b->seed, &label, &k, &b->pairs};
    if ((r = launch(d, d->f_perm, (k / 8 + 255) / 256, 1, 0, a2))) return r;
    if (!apply) return 0;
    unsigned lo = (unsigned)nonce, hi = (unsigned)(nonce >> 32);
    void *a3[] = {&b->base, &b->uni, &b->pairs, &b->out, &rows, &k, &lo, &hi, &patch};
    return launch(d, d->f_apply, (rows + APPLY_ROWS - 1) / APPLY_ROWS, 1, APPLY_SMEM(k), a3);
}

// batch of `count` nonces: seeds (8 words each) -> uniform rows (count x rows x 128) + pairs (count x 2k)
static CUresult run_noise_batch(gpu_dev *d, CUdeviceptr seeds, CUdeviceptr uni, CUdeviceptr pairs, unsigned rows,
                                unsigned k, unsigned count) {
    CUresult r;
    unsigned label = d->eg_fused ? 2 : 0, nblocks = rows * 4;
    void *a1[] = {&seeds, &label, &nblocks, &uni};
    if ((r = launch3(d, d->f_uni, (nblocks + 255) / 256, 1, count, 256, 0, a1))) return r;
    void *a2[] = {&seeds, &label, &k, &pairs};
    return launch3(d, d->f_perm, (k / 8 + 255) / 256, 1, count, 256, 0, a2);
}

static CUresult run_mine(gpu_dev *d, CUfunction f, unsigned smem, CUdeviceptr An, CUdeviceptr Bn, unsigned m,
                         unsigned n, unsigned k, CUdeviceptr seed, CUdeviceptr bound, CUdeviceptr cand, CUdeviceptr dbg) {
    unsigned cap = CAND_CAP;
    void *args[] = {&An, &Bn, &m, &n, &k, &seed, &bound, &cand, &cap, &dbg};
    return launch(d, f, n / BN, m / BM, smem, args);
}

// fused: A' generated in the kernel from the uniform rows + pairs (base A must be zero)
static CUresult run_mine_fused_z(gpu_dev *d, CUfunction f, unsigned threads, unsigned smem, CUdeviceptr U,
                                 CUdeviceptr P, uint64_t nonce, CUdeviceptr Bn, unsigned m, unsigned n, unsigned k,
                                 CUdeviceptr seed, CUdeviceptr bound, CUdeviceptr cand, CUdeviceptr dbg, unsigned count) {
    unsigned cap = CAND_CAP, lo = (unsigned)nonce, hi = (unsigned)(nonce >> 32);
    void *args[] = {&U, &P, &lo, &hi, &Bn, &m, &n, &k, &seed, &bound, &cand, &cap, &dbg};
    return launch3(d, f, n / BN, m / BM, count, threads, smem, args);
}
static CUresult run_mine_fused(gpu_dev *d, CUfunction f, unsigned threads, unsigned smem, CUdeviceptr U, CUdeviceptr P, uint64_t nonce,
                               CUdeviceptr Bn, unsigned m, unsigned n, unsigned k, CUdeviceptr seed, CUdeviceptr bound,
                               CUdeviceptr cand, CUdeviceptr dbg) {
    return run_mine_fused_z(d, f, threads, smem, U, P, nonce, Bn, m, n, k, seed, bound, cand, dbg, 1);
}

// ------------------------------------------------------------------ self-test
// Small problem (m=128, n=256, k=2048): GPU noise + transcripts must match the CPU core bit-for-bit.
// fused: 0 classic (A' from memory, mine_body), 1 noise generated in the kernel, 2 L2 kernel (A' from memory)
static int selftest_kernel(gpu_dev *d, CUfunction fmine, unsigned smem, const char *name, int fused, unsigned threads) {
    const unsigned m = 128, n = 256, k = 2048;
    const int nt = g_miner.nt;
    int ok = 0;
    CUdeviceptr dA = 0, dAn = 0, dB = 0, dBn = 0, dUa = 0, dUb = 0, dPa = 0, dPb = 0, dSa = 0, dSb = 0, dBound = 0,
                dCand = 0, dDbg = 0;
    size_t ntiles = (m / 2) * (n / (2 * nt));
    int8_t *A = malloc(m * k), *Bt = malloc(n * k), *An = malloc(m * k), *Bn = malloc(n * k);
    uint32_t *dbg = malloc(ntiles * 64), *cand = malloc(CAND_WORDS * 4);
    uint8_t sa[32], sb[32];
    uint64_t s = 0x1234567 + d->idx;
    for (unsigned i = 0; i < m * k; i++) { s = s * 6364136223846793005ULL + 1; A[i] = (int8_t)((int)((s >> 33) % 129) - 64); }
    if (fused) memset(A, 0, m * k);   // fused kernels assume a zero base A (as the miner uses)
    for (unsigned i = 0; i < n * k; i++) { s = s * 6364136223846793005ULL + 1; Bt[i] = (int8_t)((int)((s >> 33) % 129) - 64); }
    for (int i = 0; i < 32; i++) { s = s * 6364136223846793005ULL + 1; sa[i] = (uint8_t)(s >> 40); sb[i] = (uint8_t)(s >> 48); }
    uint64_t nonce = 0x0f1e2d3c4b5a6978ULL;
    CK(cu.MemAlloc(&dA, m * k)); CK(cu.MemAlloc(&dAn, m * k));
    CK(cu.MemAlloc(&dB, n * k)); CK(cu.MemAlloc(&dBn, n * k));
    CK(cu.MemAlloc(&dUa, m * 128)); CK(cu.MemAlloc(&dUb, n * 128));
    CK(cu.MemAlloc(&dPa, 2 * k)); CK(cu.MemAlloc(&dPb, 2 * k));
    CK(cu.MemAlloc(&dSa, 32)); CK(cu.MemAlloc(&dSb, 32)); CK(cu.MemAlloc(&dBound, 32));
    CK(cu.MemAlloc(&dCand, CAND_WORDS * 4)); CK(cu.MemAlloc(&dDbg, ntiles * 64));
    CK(cu.MemcpyHtoD(dA, A, m * k)); CK(cu.MemcpyHtoD(dB, Bt, n * k));
    CK(cu.MemcpyHtoD(dSa, sa, 32)); CK(cu.MemcpyHtoD(dSb, sb, 32));
    {
        noise_bufs na = {dA, dUa, dPa, dAn, dSa}, nb = {dB, dUb, dPb, dBn, dSb};
        CK(run_noise(d, &na, 0, m, k, nonce, 1, true));
        CK(run_noise(d, &nb, 1, n, k, 0, 0, true));
    }
    CK(cu.StreamSynchronize(d->st));
    CK(cu.MemcpyDtoH(An, dAn, m * k));
    CK(cu.MemcpyDtoH(Bn, dBn, n * k));
    {   // CPU noise reference
        uint8_t *perm = malloc(2 * k);
        int8_t uni[128], *nz = malloc(k);
        int bad = 0;
        prl_noise_perm(PRL_LABEL_A, sa, (int)k, perm);
        for (unsigned r = 0; r < m; r++) {
            int8_t row[2048];
            memcpy(row, A + r * k, k);
            if (r == 0) prl_apply_nonce(row, nonce);
            prl_noise_uniform_row(PRL_LABEL_A, sa, r, uni);
            prl_noise_row(uni, perm, (int)k, nz);
            for (unsigned l = 0; l < k; l++) bad += (int8_t)(row[l] + nz[l]) != An[r * k + l];
        }
        prl_noise_perm(PRL_LABEL_B, sb, (int)k, perm);
        for (unsigned r = 0; r < n; r++) {
            prl_noise_uniform_row(PRL_LABEL_B, sb, r, uni);
            prl_noise_row(uni, perm, (int)k, nz);
            for (unsigned l = 0; l < k; l++) bad += (int8_t)(Bt[r * k + l] + nz[l]) != Bn[r * k + l];
        }
        free(perm); free(nz);
        if (bad) { LOGE("GPU%d: self-test: noise kernels mismatch (%d bytes)", d->idx, bad); goto fail; }
    }
    {   // transcripts for every tile + candidate reporting with an "accept everything" bound
        uint32_t bound[8];
        memset(bound, 0xff, sizeof bound);
        CK(cu.MemcpyHtoD(dBound, bound, 32));
        CK(cu.MemsetD32Async(dCand, 0, CAND_WORDS, d->st));
        if (fused == 3) {   // ELt layout of the same uniform rows
            unsigned lb = 2, nbk = m * 4;
            void *ua[] = {&dSa, &lb, &nbk, &dUa};
            CK(launch(d, d->f_uni, (nbk + 255) / 256, 1, 0, ua));
        }
        if (fused) CK(run_mine_fused(d, fmine, threads, smem, fused == 2 ? dAn : dUa, dPa, nonce, dBn, m, n, k, dSa,
                                     dBound, dCand, dDbg));
        else CK(run_mine(d, fmine, smem, dAn, dBn, m, n, k, dSa, dBound, dCand, dDbg));
        CK(cu.StreamSynchronize(d->st));
        CK(cu.MemcpyDtoH(dbg, dDbg, ntiles * 64));
        CK(cu.MemcpyDtoH(cand, dCand, CAND_WORDS * 4));
        prl_geom g;
        prl_geom_init(&g, m, n, k, nt);
        int bad = 0;
        size_t tiles_c = n / (2 * nt);
        for (unsigned tr0 = 0; tr0 < m; tr0 += 16)
            for (unsigned gg = 0; gg < 8; gg++)
                for (unsigned c0 = 0; c0 < n; c0 += 8 * nt)
                    for (unsigned qq = 0; qq < 4; qq++) {
                        uint32_t tr = tr0 + gg, tc = c0 + 2 * qq, rows[2], cols[64], jp[16];
                        prl_tile_rows(&g, tr, rows);
                        prl_tile_cols(&g, tc, cols);
                        const int8_t *ap[2] = {An + rows[0] * k, An + rows[1] * k}, *bp[64];
                        for (int j = 0; j < g.w; j++) bp[j] = Bn + cols[j] * k;
                        prl_jackpot(ap, 2, bp, g.w, (int)k, 128, jp);
                        size_t rid = (tr / 16) * 8 + tr % 16, cid = (tc / (8 * nt)) * 4 + (tc % (8 * nt)) / 2;
                        if (memcmp(jp, dbg + (rid * tiles_c + cid) * 16, 64)) bad++;
                    }
        if (bad) { LOGW("GPU%d: self-test: %s kernel: %d of %zu tiles wrong", d->idx, name, bad, ntiles); goto fail; }
        if (cand[0] != ntiles) { LOGW("GPU%d: self-test: %s kernel reported %u of %zu candidates", d->idx, name, cand[0], ntiles); goto fail; }
        // verify reported hash words for the first candidate
        uint32_t tr = cand[4], tc = cand[5], rows[2], cols[64], jp[16];
        prl_tile_rows(&g, tr, rows);
        prl_tile_cols(&g, tc, cols);
        const int8_t *ap[2] = {An + rows[0] * k, An + rows[1] * k}, *bp[64];
        for (int j = 0; j < g.w; j++) bp[j] = Bn + cols[j] * k;
        prl_jackpot(ap, 2, bp, g.w, (int)k, 128, jp);
        uint8_t hh[32];
        prl_jackpot_hash(jp, sa, hh);
        uint32_t w7 = hh[28] | hh[29] << 8 | hh[30] << 16 | (uint32_t)hh[31] << 24;
        if (w7 != cand[6]) { LOGW("GPU%d: self-test: %s kernel BLAKE3 mismatch", d->idx, name); goto fail; }
    }
    ok = 1;
fail:
    CUdeviceptr bufs[] = {dA, dAn, dB, dBn, dUa, dUb, dPa, dPb, dSa, dSb, dBound, dCand, dDbg};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++)
        if (bufs[i]) cu.MemFree(bufs[i]);
    free(A); free(Bt); free(An); free(Bn); free(dbg); free(cand);
    return ok;
}

// k_seeds vs the CPU (Merkle path of chunk 0 + seed binding) for a batch of nonces, salted and legacy
static int selftest_seeds(gpu_dev *d) {
    const size_t m = 128, k = 2048, N = 37;
    int ok = 0;
    uint8_t *A = calloc(m * k, 1), key[32], bs[32];
    uint32_t *ctx = malloc(4 * PRL_SEEDCTX_WORDS), out[8 * 37];
    CUdeviceptr dctx = 0, dout = 0;
    for (int i = 0; i < 32; i++) { key[i] = (uint8_t)(i * 11 + 1); bs[i] = (uint8_t)(i * 5 + 77); }
    prl_tree t;
    prl_tree_build(&t, A, m * k, key, 1);
    CK(cu.MemAlloc(&dctx, 4 * PRL_SEEDCTX_WORDS));
    CK(cu.MemAlloc(&dout, sizeof out));
    for (int salted = 0; salted < 2; salted++) {
        if (prl_seed_ctx(&t, A, bs, (uint32_t)m, salted, ctx)) goto fail;
        CK(cu.MemcpyHtoD(dctx, ctx, 4 * PRL_SEEDCTX_WORDS));
        const uint64_t base = 0x00000007fffffff3ULL;
        unsigned lo = (unsigned)base, hi = (unsigned)(base >> 32), cnt = N;
        void *args[] = {&dctx, &lo, &hi, &cnt, &dout};
        CK(launch3(d, d->f_seeds, 1, 1, 1, 256, 0, args));
        CK(cu.StreamSynchronize(d->st));
        CK(cu.MemcpyDtoH(out, dout, sizeof out));
        for (size_t z = 0; z < N; z++) {
            uint8_t c0[PRL_CHUNK], sd[32];
            memcpy(c0, A, PRL_CHUNK);
            prl_apply_nonce((int8_t *)c0, base + z);
            prl_path0 p;
            prl_tree_path0(&t, c0, &p);
            prl_seed_a(bs, p.node[p.nlayers - 1], (uint32_t)m, salted, sd);
            if (memcmp(sd, &out[8 * z], 32)) { LOGW("GPU%d: GPU seed self-test failed, seeds stay on the CPU", d->idx); goto fail; }
        }
    }
    ok = 1;
fail:
    if (dctx) cu.MemFree(dctx);
    if (dout) cu.MemFree(dout);
    prl_tree_free(&t);
    free(A); free(ctx);
    return ok;
}

// ------------------------------------------------------------------ prepare
// load function <prefix><nt> from module, self-test it, and add it to the fused candidates
static void add_fused(gpu_dev *d, CUmodule mod, const char *prefix, unsigned smem, const char *name, int maxsm,
                      unsigned threads, int l2) {
    char fname[64];
    CUfunction f;
    snprintf(fname, sizeof fname, "%s%d", prefix, g_miner.nt);
    if ((int)smem > maxsm) {
        LOGD("GPU%d: kernel %s needs %u bytes of shared memory (max %d), skipped", d->idx, name, smem, maxsm);
        return;
    }
    if (cu.ModuleGetFunction(&f, mod, fname) ||
        (smem > 48 * 1024 && cu.FuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem))) {
        LOGW("GPU%d: kernel %s could not be loaded", d->idx, name);
        return;
    }
    if (!selftest_kernel(d, f, smem, name, l2 == 1 ? 2 : l2 == 3 ? 3 : 1, threads)) {
        LOGW("GPU%d: kernel %s failed the self-test, not used", d->idx, name);
        return;
    }
    int regs = 0, lmem = 0;
    cu.FuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, f);
    cu.FuncGetAttribute(&lmem, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f);
    LOGI("GPU%d: kernel %s-nt%d (%s): %d registers, %d bytes local memory (spills)", d->idx, name, g_miner.nt,
         l2 == 1 ? "L2-resident A'" : "fused noise", regs, lmem);
    if (d->nfv < 24) {
        d->fv[d->nfv].l2 = l2 == 1;
        d->fv[d->nfv].eg = l2 == 3;
        d->fv[d->nfv].f = f;
        d->fv[d->nfv].smem = smem;
        d->fv[d->nfv].threads = threads;
        snprintf(d->fv[d->nfv].name, sizeof d->fv[d->nfv].name, "%s", name);
        d->nfv++;
    }
}

gpu_dev *gpu_prepare(int stat_idx, const gpu_info *info, const char *kernel, int rows) {
    gpu_dev *d = calloc(1, sizeof(gpu_dev));
    d->idx = stat_idx;
    d->info = *info;
    dev_stats *ds = &g_miner.dev[stat_idx];
    snprintf(ds->tag, sizeof ds->tag, "GPU%d", stat_idx);
    snprintf(ds->name, sizeof ds->name, "%s", info->name);
    snprintf(ds->busid, sizeof ds->busid, "%s", info->busid);
    int cc = info->cc_major * 10 + info->cc_minor;
    if (cc < 61) { LOGE("GPU%d %s: compute capability %d.%d is not supported (need 6.1+)", stat_idx, info->name,
                        info->cc_major, info->cc_minor); free(d); return NULL; }
    CUdevice dv;
    CK(cu.DeviceGet(&dv, info->ordinal));
    CK(cu.CtxCreate(&d->ctx, CU_CTX_SCHED_BLOCKING_SYNC, dv));
    CK(cu.StreamCreate(&d->st, 0));
    CK(cu.StreamCreate(&d->st2, 0));
    {
        bool sm80 = cc >= 80;
        CUmodule mn;
        if (load_module(d, &mn, ptx_for(sm80, "noise"), "noise")) goto fail;
        CK(cu.ModuleGetFunction(&d->f_uni, mn, "k_noise_uniform"));
        CK(cu.ModuleGetFunction(&d->f_perm, mn, "k_noise_perm"));
        CK(cu.ModuleGetFunction(&d->f_apply, mn, "k_apply_noise"));
        if (!cu.ModuleGetFunction(&d->f_seeds, mn, "k_seeds")) d->gpu_seeds = selftest_seeds(d);
        if (d->gpu_seeds) LOGI("GPU%d: per-nonce seeds on the GPU: self-test passed", stat_idx);
        if (APPLY_SMEM(g_miner.k) > 48 * 1024)
            CK(cu.FuncSetAttribute(d->f_apply, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, APPLY_SMEM(g_miner.k)));
        char kn[32], fname[64];
        unsigned smem_dp4a = sm80 ? SMEM_MMA : SMEM_DP4A_SM61;
        bool turing = cc >= 75 && cc < 80;
        bool want_mma = (sm80 || turing) && strcmp(kernel, "dp4a") != 0;
        if (!strcmp(kernel, "mma") && !sm80 && !turing)
            LOGW("GPU%d: tensor-core kernel needs compute 7.5+, using dp4a", stat_idx);
        bool done = false;
        if (want_mma) {
            CUmodule mm;
            CUfunction fm;
            unsigned smem_mma = sm80 ? SMEM_MMA : SMEM_DP4A_SM61;   // Turing: 2-stage pipeline
            snprintf(kn, sizeof kn, "mma%d", g_miner.nt);
            snprintf(fname, sizeof fname, "k_mine_mma_nt%d", g_miner.nt);
            if (load_module(d, &mm, sm80 ? ptx_for(true, kn) : ptx_turing_mma(g_miner.nt), kn)) goto fail;
            CK(cu.ModuleGetFunction(&fm, mm, fname));
            if (smem_mma > 48 * 1024)
                CK(cu.FuncSetAttribute(fm, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_mma));
            if (selftest_kernel(d, fm, smem_mma, "tensor-core", false, 256)) {
                d->f_mine = fm; d->smem = smem_mma; d->mma = true; done = true;
            } else LOGW("GPU%d: tensor-core kernel failed the self-test, falling back to dp4a", stat_idx);
            if (done && g_fused_mode >= 0) {   // fused-noise kernels (v1 + v2 variants), all self-tested
                int maxsm = 48 * 1024;
                if (cu.DeviceGetAttribute(&maxsm, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dv) || maxsm <= 0)
                    maxsm = 48 * 1024;
                CUmodule mf;
                snprintf(kn, sizeof kn, "fmma%d", g_miner.nt);
                if (!load_module(d, &mf, ptx_fused(sm80, g_miner.nt), kn))
                    add_fused(d, mf, "k_mine_fmma_nt", smem_mma + FUSED_SMEM, "fmma", maxsm, 256, 0);
                if (g_miner.nt == 16 || g_miner.nt == 32) {
                    const char *ptx = sm80 ? (g_miner.nt == 16 ? prl_ptx_sm80_v2_16 : prl_ptx_sm80_v2_32)
                                           : (g_miner.nt == 16 ? prl_ptx_sm75_v2_16 : prl_ptx_sm75_v2_32);
                    CUmodule m2;
                    if (!load_module(d, &m2, ptx, "v2")) {
                        const v2_variant *vt = sm80 ? V2_SM80 : V2_SM75;
                        for (int i = 0; i < NV2; i++) {
                            if ((vt[i].mt || vt[i].l2 || vt[i].eg) && g_miner.nt != 16) continue;   // nt16 only
                            char fn[32];
                            snprintf(fn, sizeof fn, "k_%s_nt", vt[i].name);
                            add_fused(d, m2, fn, v2_smem(&vt[i], g_miner.nt), vt[i].name, maxsm, vt[i].threads,
                                      vt[i].eg ? 3 : vt[i].l2);
                        }
                    }
                }
                if (!d->nfv) LOGW("GPU%d: no fused-noise kernel available, using the classic one", stat_idx);
            }
        }
        if (!done) {
            CUmodule md;
            CUfunction fd;
            snprintf(kn, sizeof kn, "dp4a%d", g_miner.nt);
            snprintf(fname, sizeof fname, "k_mine_dp4a_nt%d", g_miner.nt);
            if (load_module(d, &md, ptx_for(sm80, kn), kn)) goto fail;
            CK(cu.ModuleGetFunction(&fd, md, fname));
            if (smem_dp4a > 48 * 1024)
                CK(cu.FuncSetAttribute(fd, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_dp4a));
            if (selftest_kernel(d, fd, smem_dp4a, "dp4a", false, 256)) { d->f_mine = fd; d->smem = smem_dp4a; d->mma = false; }
            else { LOGE("GPU%d: all kernels failed the self-test, device disabled", stat_idx); goto fail; }
        }
        int regs = 0, lmem = 0;
        cu.FuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, d->f_mine);
        cu.FuncGetAttribute(&lmem, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, d->f_mine);
        snprintf(ds->kernel, sizeof ds->kernel, "%s-nt%d", d->mma ? "mma" : "dp4a", g_miner.nt);
        LOGI("GPU%d: kernel %s: %d registers, %d bytes local memory (spills)", stat_idx, ds->kernel, regs, lmem);
    }
    // rows: one 128-row block per SM per 256-column block -> grid = SMs x (n/256) blocks
    d->m = rows > 0 ? rows : BM * info->sms;
    d->m = (d->m + BM - 1) / BM * BM;
    {
        size_t need = (size_t)2 * d->m * g_miner.k + (size_t)2 * g_miner.n * g_miner.k + (64u << 20);
        if (need > info->mem * 9 / 10) {
            int maxm = (int)(((info->mem * 9 / 10) - (size_t)2 * g_miner.n * g_miner.k - (64u << 20)) /
                             (2 * (size_t)g_miner.k) / BM * BM);
            LOGW("GPU%d: not enough memory for %d rows, using %d", stat_idx, d->m, maxm);
            d->m = maxm;
        }
        if (d->m < BM) goto fail;
    }
    d->geom = miner_add_geometry(d->m);
    if (d->geom < 0) { LOGE("GPU%d: invalid geometry", stat_idx); goto fail; }
    {   // L2 kernels: rows such that 2 A' slots (m x k) + B' (n x k) fit in ~80% of L2, blocks fill whole waves
        int has_l2 = 0;
        for (int v = 0; v < d->nfv; v++) has_l2 |= d->fv[v].l2;
        int l2 = 0;
        cu.DeviceGetAttribute(&l2, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dv);
        if (has_l2 && rows <= 0) {
            const int cb = g_miner.n / BN, sms = info->sms;
            const double cap = 0.80 * l2, k = g_miner.k, n = g_miner.n;
            int best = 0, fallback = 0;
            for (int rb = 1; rb <= 64 && rb * 128 <= d->m; rb++) {
                int blocks = cb * rb, waves = (blocks + sms - 1) / sms;
                double eff = (double)blocks / ((double)waves * sms);
                if (eff < 0.9) continue;
                if ((2 * rb * 128.0 + n) * k <= cap) best = rb;       // largest that fits L2 (double-buffered A')
                else if (!fallback && rb * 128 >= 1024) fallback = rb;   // smallest decent one (small-L2 cards)
            }
            int rb = best ? best : fallback;
            if (rb && rb * 128 <= d->m) {
                d->m_l2 = rb * 128;
                d->geom_l2 = miner_add_geometry(d->m_l2);
                LOGI("GPU%d: L2 %d MB -> L2-kernel matrix %dx%dx%d (%s)", stat_idx, l2 >> 20, d->m_l2, g_miner.n,
                     g_miner.k, best ? "2 x A' + B' fit in L2" : "L2 too small, streams from memory");
            }
        }
        if (!d->m_l2) {   // drop L2 variants
            int j = 0;
            for (int v = 0; v < d->nfv; v++)
                if (!d->fv[v].l2) d->fv[j++] = d->fv[v];
            d->nfv = j;
        }
    }
    LOGI("GPU%d: %s [%s] cc %d.%d, %d SMs, %llu MB — kernel %s, self-test passed, matrix %dx%dx%d", stat_idx,
         info->name, info->busid, info->cc_major, info->cc_minor, info->sms, info->mem >> 20, ds->kernel, d->m,
         g_miner.n, g_miner.k);
    cu.CtxSetCurrent(NULL);
    return d;
fail:
    if (d->ctx) cu.CtxDestroy(d->ctx);
    free(d);
    return NULL;
}

// ------------------------------------------------------------------ mining loop
typedef struct {
    job_t *job;
    uint64_t nonce;
    CUdeviceptr dseed, dcand;
    uint8_t *hseed;
    uint32_t *hcand;
    CUevent ev;
    int busy;
} slot_t;

static void fatal_cuda(gpu_dev *d, const char *what, CUresult r) {
    LOGE("GPU%d: %s failed: %s — exiting so the miner can be restarted", d->idx, what, cu_errstr(r));
    fflush(stdout);
    sleep(1);
    _exit(42);
}
#define CKF(call) do { CUresult _r = (call); if (_r) fatal_cuda(d, #call, _r); } while (0)

static void collect(gpu_dev *d, slot_t *s, uint64_t work) {
    CKF(cu.EventSynchronize(s->ev));
    uint32_t n = s->hcand[0];
    if (n > CAND_CAP) {
        LOGW("GPU%d: %u candidates in one attempt (cap %d) — share target is very easy", d->idx, n, CAND_CAP);
        n = CAND_CAP;
    }
    for (uint32_t i = 0; i < n; i++)
        miner_candidate(s->job, d->geom, d->idx, s->nonce, s->hcand[4 + 4 * i], s->hcand[5 + 4 * i]);
    atomic_fetch_add(&g_miner.dev[d->idx].work, work);
    job_release(s->job);
    s->job = NULL;
    s->busy = 0;
}

// ------------------------------------------------------------------ L2 pipeline
// Two slots, two streams: the noise (A') of attempt i+1 is produced on st2 while the GEMM of attempt i runs on
// st. A slot's A' buffer is reused only after its GEMM finished (l2_collect waits for it).
typedef struct {
    CUdeviceptr An, Ua, Pa, seed, cand;
    uint8_t *hseed;
    uint32_t *hcand;
    CUevent ev, evn;
    job_t *job;
    uint64_t nonce;
    int busy;
} l2slot;

static void l2_enqueue(gpu_dev *d, l2slot *s, CUdeviceptr Bn, CUdeviceptr bound, unsigned m, unsigned n, unsigned k,
                       uint64_t nonce, CUfunction f, unsigned threads, unsigned smem) {
    d->lst = d->st2;
    CKF(cu.MemcpyHtoDAsync(s->seed, s->hseed, 32, d->st2));
    noise_bufs na = {0, s->Ua, s->Pa, s->An, s->seed};
    CKF(run_noise(d, &na, 0, m, k, nonce, 1, true));
    CKF(cu.EventRecord(s->evn, d->st2));
    d->lst = NULL;
    CKF(cu.StreamWaitEvent(d->st, s->evn, 0));
    CKF(cu.MemsetD32Async(s->cand, 0, 4, d->st));
    CKF(run_mine_fused(d, f, threads, smem, s->An, s->Pa, nonce, Bn, m, n, k, s->seed, bound, s->cand, 0));
    CKF(cu.MemcpyDtoHAsync(s->hcand, s->cand, CAND_WORDS * 4, d->st));
    CKF(cu.EventRecord(s->ev, d->st));
    s->busy = 1;
}

static void l2_collect(gpu_dev *d, l2slot *s, uint64_t work) {
    CKF(cu.EventSynchronize(s->ev));
    s->busy = 0;
    if (!s->job) return;   // profiling run
    uint32_t n = s->hcand[0];
    if (n > CAND_CAP) {
        LOGW("GPU%d: %u candidates in one attempt (cap %d) — share target is very easy", d->idx, n, CAND_CAP);
        n = CAND_CAP;
    }
    for (uint32_t i = 0; i < n; i++)
        miner_candidate(s->job, d->geom, d->idx, s->nonce, s->hcand[4 + 4 * i], s->hcand[5 + 4 * i]);
    atomic_fetch_add(&g_miner.dev[d->idx].work, work);
    job_release(s->job);
    s->job = NULL;
}

static void *gpu_main(void *arg) {
    gpu_dev *d = arg;
    double best_rate_mac_ms_g = 0;
    miner_t *M = &g_miner;
    const unsigned m = d->m, n = M->n, k = M->k;
    CUdeviceptr dA, dAn, dB, dBn, dUa, dUb, dPa, dPb, dSb, dBound;
    cu.CtxSetCurrent(d->ctx);
    CKF(cu.MemAlloc(&dA, (size_t)m * k));
    CKF(cu.MemAlloc(&dAn, (size_t)m * k));
    CKF(cu.MemAlloc(&dB, (size_t)n * k));
    CKF(cu.MemAlloc(&dBn, (size_t)n * k));
    CKF(cu.MemAlloc(&dUa, (size_t)m * 128));
    CKF(cu.MemAlloc(&dUb, (size_t)n * 128));
    CKF(cu.MemAlloc(&dPa, 2 * k));
    CKF(cu.MemAlloc(&dPb, 2 * k));
    CKF(cu.MemAlloc(&dSb, 32));
    CKF(cu.MemAlloc(&dBound, 32));
    CKF(cu.MemcpyHtoD(dA, M->A_base, (size_t)m * k));
    CKF(cu.MemcpyHtoD(dB, M->Bt_base, (size_t)n * k));
    slot_t slots[2];
    memset(slots, 0, sizeof slots);
    for (int i = 0; i < 2; i++) {
        CKF(cu.MemAlloc(&slots[i].dseed, 32));
        CKF(cu.MemAlloc(&slots[i].dcand, CAND_WORDS * 4));
        CKF(cu.MemAllocHost((void **)&slots[i].hseed, 32));
        CKF(cu.MemAllocHost((void **)&slots[i].hcand, CAND_WORDS * 4));
        CKF(cu.EventCreate(&slots[i].ev, 1 /* blocking sync */));
    }
    uint8_t *hb = NULL;
    CKF(cu.MemAllocHost((void **)&hb, 64));
    l2slot ls[2];
    memset(ls, 0, sizeof ls);
    if (d->m_l2) {
        for (int i = 0; i < 2; i++) {
            l2slot *q = &ls[i];
            if (i == 0) { q->An = dAn; q->Ua = dUa; q->Pa = dPa; }   // reuse the fused/classic buffers
            else {
                CKF(cu.MemAlloc(&q->An, (size_t)d->m_l2 * k));
                CKF(cu.MemAlloc(&q->Ua, (size_t)d->m_l2 * 128));
                CKF(cu.MemAlloc(&q->Pa, 2 * k));
            }
            CKF(cu.MemAlloc(&q->seed, 32));
            CKF(cu.MemAlloc(&q->cand, CAND_WORDS * 4));
            CKF(cu.MemAllocHost((void **)&q->hseed, 32));
            CKF(cu.MemAllocHost((void **)&q->hcand, CAND_WORDS * 4));
            memset(q->hseed, 0x5a + i, 32);
            CKF(cu.EventCreate(&q->ev, 1));
            CKF(cu.EventCreate(&q->evn, 1));
        }
    }
    {   // one-time profile of a full-size attempt: noise vs GEMM time
        CUevent e0, e1, e2;
        CKF(cu.EventCreate(&e0, 0)); CKF(cu.EventCreate(&e1, 0)); CKF(cu.EventCreate(&e2, 0));
        memset(slots[0].hseed, 0x5a, 32);
        CKF(cu.MemcpyHtoD(slots[0].dseed, slots[0].hseed, 32));
        uint32_t zero[8] = {0};
        CKF(cu.MemcpyHtoD(dBound, zero, 32));
        noise_bufs nb = {dB, dUb, dPb, dBn, slots[0].dseed};
        CKF(run_noise(d, &nb, 1, n, k, 0, 0, true));
        float tn = 0, tg = 0;
        for (int it = 0; it < 3; it++) {
            noise_bufs na = {dA, dUa, dPa, dAn, slots[0].dseed};
            CKF(cu.EventRecord(e0, d->st));
            CKF(run_noise(d, &na, 0, m, k, 1, 1, true));
            CKF(cu.EventRecord(e1, d->st));
            CKF(run_mine(d, d->f_mine, d->smem, dAn, dBn, m, n, k, slots[0].dseed, dBound, slots[0].dcand, 0));
            CKF(cu.EventRecord(e2, d->st));
            CKF(cu.EventSynchronize(e2));
            cu.EventElapsedTime(&tn, e0, e1);
            cu.EventElapsedTime(&tg, e1, e2);
        }
        char b1[32];
        LOGI("GPU%d: profile: attempt %.2f ms = noise %.2f ms + GEMM %.2f ms; GEMM alone %sH/s", d->idx, tn + tg, tn,
             tg, fmt_rate(tg > 0 ? (double)m * n * k / (tg * 1e-3) : 0, b1, sizeof b1));
        // compare by throughput (TH/s): L2 kernels run a smaller matrix
        double best = (g_fused_mode > 0 && d->nfv) ? 0 : (tn + tg > 0 ? (double)m * n * k / (tn + tg) : 0);
        int sel = -1;
        for (int v = 0; v < d->nfv; v++) {
            if (g_variant && strcmp(d->fv[v].name, g_variant)) continue;   // forced variant: profile only it
            float tf = 0, tfn = 0, tmin = 1e30f;
            const int l2v = d->fv[v].l2;
            const unsigned mv = l2v ? (unsigned)d->m_l2 : m;
            if (l2v) {   // steady-state pipelined throughput (wall clock, includes host overhead) + serial split
                for (int pass = 0; pass < 2; pass++) {
                    const int N = pass ? 32 : 4;   // pass 0 = warm-up
                    CKF(cu.CtxSynchronize());
                    double t0 = now_sec();
                    for (int i = 0; i < N; i++) {
                        l2slot *q = &ls[i & 1];
                        if (q->busy) l2_collect(d, q, 0);
                        l2_enqueue(d, q, dBn, dBound, mv, n, k, 1 + i, d->fv[v].f, d->fv[v].threads, d->fv[v].smem);
                    }
                    for (int i = 0; i < 2; i++)
                        if (ls[i].busy) l2_collect(d, &ls[i], 0);
                    tmin = (float)((now_sec() - t0) * 1e3 / N);
                }
                noise_bufs na = {0, dUa, dPa, dAn, slots[0].dseed};
                CKF(cu.EventRecord(e0, d->st));
                CKF(run_noise(d, &na, 0, mv, k, 1, 1, true));
                CKF(cu.EventRecord(e1, d->st));
                CKF(run_mine_fused(d, d->fv[v].f, d->fv[v].threads, d->fv[v].smem, dAn, dPa, 1, dBn, mv, n, k,
                                   slots[0].dseed, dBound, slots[0].dcand, 0));
                CKF(cu.EventRecord(e2, d->st));
                CKF(cu.EventSynchronize(e2));
                cu.EventElapsedTime(&tfn, e0, e1);
                cu.EventElapsedTime(&tf, e1, e2);
                double rate = tmin > 0 ? (double)mv * n * k / tmin : 0;
                LOGI("GPU%d: profile: %-5s pipelined %.3f ms/attempt (%sH/s); serial: noise %.3f + GEMM %.3f ms "
                     "(GEMM alone %.1f TH/s)", d->idx, d->fv[v].name, tmin, fmt_rate(rate * 1e3, b1, sizeof b1), tfn,
                     tf, tf > 0 ? (double)mv * n * k / (tf * 1e-3) / 1e12 : 0);
                if (rate > best) { best = rate; sel = v; }
                continue;
            }
            for (int it = 0; it < 4; it++) {
                noise_bufs na = {l2v ? 0 : dA, dUa, dPa, dAn, slots[0].dseed};
                CKF(cu.EventRecord(e0, d->st));
                CKF(run_noise(d, &na, d->fv[v].eg ? 2 : 0, mv, k, 1, 1, l2v));
                CKF(cu.EventRecord(e1, d->st));
                CKF(run_mine_fused(d, d->fv[v].f, d->fv[v].threads, d->fv[v].smem, l2v ? dAn : dUa, dPa, 1, dBn, mv, n,
                                   k, slots[0].dseed, dBound, slots[0].dcand, 0));
                CKF(cu.EventRecord(e2, d->st));
                CKF(cu.EventSynchronize(e2));
                cu.EventElapsedTime(&tfn, e0, e1);
                cu.EventElapsedTime(&tf, e1, e2);
                if (it && tfn + tf < tmin) tmin = tfn + tf;   // first run is a warm-up
            }
            double rate = tmin > 0 ? (double)mv * n * k / tmin : 0;
            LOGI("GPU%d: profile: %-5s attempt %.2f ms (%sH/s)", d->idx, d->fv[v].name, tmin,
                 fmt_rate(rate * 1e3, b1, sizeof b1));
            if (rate > best) { best = rate; sel = v; }
        }
        for (int v = 0; g_variant && v < d->nfv; v++)
            if (!strcmp(d->fv[v].name, g_variant)) { sel = v; best = 0; }
        if (sel >= 0) {
            d->use_fused = true;
            d->f_fused = d->fv[sel].f;
            d->smem_fused = d->fv[sel].smem;
            d->threads_fused = d->fv[sel].threads;
            d->use_l2 = d->fv[sel].l2;
            d->eg_fused = d->fv[sel].eg;
            snprintf(M->dev[d->idx].kernel, sizeof M->dev[d->idx].kernel, "%s-nt%d", d->fv[sel].name, M->nt);
        }
        best_rate_mac_ms_g = best;
        LOGI("GPU%d: using kernel %s (%sH/s, matrix %ux%ux%u)", d->idx, M->dev[d->idx].kernel,
             fmt_rate(best * 1e3, b1, sizeof b1), d->use_l2 ? (unsigned)d->m_l2 : m, n, k);
    }
    if (d->use_l2) d->geom = d->geom_l2;   // attempts use the L2 geometry from here on
    double best_rate_mac_ms = best_rate_mac_ms_g;
    const unsigned mr = d->use_l2 ? (unsigned)d->m_l2 : m;
    const uint64_t work_run = (uint64_t)mr * n * k;
    M->dev[d->idx].active = 1;
    if (d->use_l2) {   // pipelined L2 loop
        uint8_t lbs[32];
        int hb_ok = 0, cur2 = 0;
        while (!M->quit) {
            job_t *job = miner_wait_job(1.0);
            if (!job) continue;
            uint64_t seq = job->seq;
            if (!hb_ok || memcmp(lbs, job->work->b_seed, 32)) {
                memcpy(hb, job->work->b_seed, 32);
                CKF(cu.MemcpyHtoDAsync(dSb, hb, 32, d->st));
                noise_bufs nb = {dB, dUb, dPb, dBn, dSb};
                CKF(run_noise(d, &nb, 1, n, k, 0, 0, true));
                memcpy(lbs, job->work->b_seed, 32);
                hb_ok = 1;
            }
            memcpy(hb + 32, job->bound.w, 32);
            CKF(cu.MemcpyHtoDAsync(dBound, hb + 32, 32, d->st));
            CKF(cu.StreamSynchronize(d->st));
            while (!M->quit && miner_seq() == seq) {
                l2slot *q = &ls[cur2];
                if (q->busy) l2_collect(d, q, work_run);
                q->nonce = miner_next_nonce();
                uint8_t chunk0[PRL_CHUNK];
                miner_attempt_chunk0(job->work, d->geom, q->nonce, chunk0, q->hseed);
                q->job = job;
                atomic_fetch_add(&job->refs, 1);
                l2_enqueue(d, q, dBn, dBound, mr, n, k, q->nonce, d->f_fused, d->threads_fused, d->smem_fused);
                cur2 ^= 1;
            }
            for (int i = 0; i < 2; i++)
                if (ls[i].busy) l2_collect(d, &ls[i], work_run);
            job_release(job);
        }
        M->dev[d->idx].active = 0;
        return NULL;
    }
    if (d->use_fused && !d->use_l2 && d->gpu_seeds && getenv("EJ_NO_BATCH") == NULL) {
        // Batched loop: B nonces per launch, seeds derived on the GPU (k_seeds) -> ~7 driver calls and no
        // hashing on the CPU per batch instead of per attempt.
        double t_ms = best_rate_mac_ms > 0 ? (double)m * n * k / best_rate_mac_ms : 1.0;
        unsigned B = (unsigned)(4.0 / t_ms + 0.5);
        if (B < 1) B = 1;
        if (B > 16) B = 16;
        LOGI("GPU%d: batches of %u nonces per launch (~%.1f ms), seeds on the GPU", d->idx, B, B * t_ms);
        typedef struct {
            CUdeviceptr seeds, cand;
            uint32_t *hcand;
            CUevent ev;
            job_t *job;
            uint64_t nonce;
            int busy;
        } bslot;
        bslot bs[2];
        memset(bs, 0, sizeof bs);
        CUdeviceptr dUb_, dPb_, dctx;
        CKF(cu.MemAlloc(&dUb_, (size_t)B * m * 128));
        CKF(cu.MemAlloc(&dPb_, (size_t)B * 2 * k));
        CKF(cu.MemAlloc(&dctx, 4 * PRL_SEEDCTX_WORDS));
        uint32_t *hctx = NULL;
        CKF(cu.MemAllocHost((void **)&hctx, 4 * PRL_SEEDCTX_WORDS));
        for (int i = 0; i < 2; i++) {
            CKF(cu.MemAlloc(&bs[i].seeds, 32 * B));
            CKF(cu.MemAlloc(&bs[i].cand, CAND_WORDS * 4));
            CKF(cu.MemAllocHost((void **)&bs[i].hcand, CAND_WORDS * 4));
            CKF(cu.EventCreate(&bs[i].ev, 1));
        }
        const uint64_t work_b = (uint64_t)m * n * k * B;
        uint8_t last_key[32];
        int have_key = 0, curb = 0;
        uint8_t lbs[32];
        int hb_ok = 0;
        while (!M->quit) {
            job_t *job = miner_wait_job(1.0);
            if (!job) continue;
            uint64_t seq = job->seq;
            if (!hb_ok || memcmp(lbs, job->work->b_seed, 32)) {
                memcpy(hb, job->work->b_seed, 32);
                CKF(cu.MemcpyHtoDAsync(dSb, hb, 32, d->st));
                noise_bufs nb = {dB, dUb, dPb, dBn, dSb};
                CKF(run_noise(d, &nb, 1, n, k, 0, 0, true));
                memcpy(lbs, job->work->b_seed, 32);
                hb_ok = 1;
            }
            if (!have_key || memcmp(last_key, job->work->job_key, 32)) {   // new header: seed context
                CKF(cu.StreamSynchronize(d->st));
                if (prl_seed_ctx(&job->work->g[d->geom].tree_a, M->A_base, job->work->b_seed,
                                 (uint32_t)job->work->g[d->geom].m, job->work->salted, hctx)) {
                    LOGE("GPU%d: seed context failed", d->idx);
                    job_release(job);
                    break;
                }
                CKF(cu.MemcpyHtoDAsync(dctx, hctx, 4 * PRL_SEEDCTX_WORDS, d->st));
                memcpy(last_key, job->work->job_key, 32);
                have_key = 1;
            }
            memcpy(hb + 32, job->bound.w, 32);
            CKF(cu.MemcpyHtoDAsync(dBound, hb + 32, 32, d->st));
            CKF(cu.StreamSynchronize(d->st));
            while (!M->quit && miner_seq() == seq) {
                bslot *q = &bs[curb];
                if (q->busy) {
                    CKF(cu.EventSynchronize(q->ev));
                    uint32_t nc = q->hcand[0];
                    if (nc > CAND_CAP) nc = CAND_CAP;
                    for (uint32_t i = 0; i < nc; i++)
                        miner_candidate(q->job, d->geom, d->idx, q->nonce + q->hcand[7 + 4 * i], q->hcand[4 + 4 * i],
                                        q->hcand[5 + 4 * i]);
                    atomic_fetch_add(&M->dev[d->idx].work, work_b);
                    job_release(q->job);
                    q->job = NULL;
                    q->busy = 0;
                }
                q->nonce = miner_next_nonces(B);
                q->job = job;
                atomic_fetch_add(&job->refs, 1);
                unsigned lo = (unsigned)q->nonce, hi = (unsigned)(q->nonce >> 32), cnt = B;
                void *sa[] = {&dctx, &lo, &hi, &cnt, &q->seeds};
                CKF(launch3(d, d->f_seeds, 1, 1, 1, 256, 0, sa));
                CKF(run_noise_batch(d, q->seeds, dUb_, dPb_, m, k, B));
                CKF(cu.MemsetD32Async(q->cand, 0, 4, d->st));
                CKF(run_mine_fused_z(d, d->f_fused, d->threads_fused, d->smem_fused, dUb_, dPb_, q->nonce, dBn, m, n, k,
                                     q->seeds, dBound, q->cand, 0, B));
                CKF(cu.MemcpyDtoHAsync(q->hcand, q->cand, CAND_WORDS * 4, d->st));
                CKF(cu.EventRecord(q->ev, d->st));
                q->busy = 1;
                curb ^= 1;
            }
            for (int i = 0; i < 2; i++) {
                bslot *q = &bs[i];
                if (!q->busy) continue;
                CKF(cu.EventSynchronize(q->ev));
                uint32_t nc = q->hcand[0];
                if (nc > CAND_CAP) nc = CAND_CAP;
                for (uint32_t j = 0; j < nc; j++)
                    miner_candidate(q->job, d->geom, d->idx, q->nonce + q->hcand[7 + 4 * j], q->hcand[4 + 4 * j],
                                    q->hcand[5 + 4 * j]);
                atomic_fetch_add(&M->dev[d->idx].work, work_b);
                job_release(q->job);
                q->job = NULL;
                q->busy = 0;
            }
            job_release(job);
        }
        M->dev[d->idx].active = 0;
        return NULL;
    }
    uint8_t last_bseed[32];
    int have_b = 0;
    int cur = 0;
    while (!M->quit) {
        job_t *job = miner_wait_job(1.0);
        if (!job) continue;
        uint64_t seq = job->seq;
        if (!have_b || memcmp(last_bseed, job->work->b_seed, 32)) {   // new header: noise B for this job
            memcpy(hb, job->work->b_seed, 32);
            CKF(cu.MemcpyHtoDAsync(dSb, hb, 32, d->st));
            noise_bufs nb = {dB, dUb, dPb, dBn, dSb};
            CKF(run_noise(d, &nb, 1, n, k, 0, 0, true));
            memcpy(last_bseed, job->work->b_seed, 32);
            have_b = 1;
        }
        memcpy(hb + 32, job->bound.w, 32);
        CKF(cu.MemcpyHtoDAsync(dBound, hb + 32, 32, d->st));
        CKF(cu.StreamSynchronize(d->st));
        while (!M->quit && miner_seq() == seq) {
            slot_t *s = &slots[cur];
            if (s->busy) collect(d, s, work_run);
            s->nonce = miner_next_nonce();
            uint8_t chunk0[PRL_CHUNK];
            miner_attempt_chunk0(job->work, d->geom, s->nonce, chunk0, s->hseed);
            s->job = job;
            atomic_fetch_add(&job->refs, 1);
            CKF(cu.MemcpyHtoDAsync(s->dseed, s->hseed, 32, d->st));
            CKF(cu.MemsetD32Async(s->dcand, 0, 4, d->st));
            noise_bufs na = {d->use_l2 ? 0 : dA, dUa, dPa, dAn, s->dseed};
            CKF(run_noise(d, &na, d->eg_fused ? 2 : 0, mr, k, s->nonce, 1, !d->use_fused || d->use_l2));
            if (d->use_fused)
                CKF(run_mine_fused(d, d->f_fused, d->threads_fused, d->smem_fused, d->use_l2 ? dAn : dUa, dPa, s->nonce,
                                   dBn, mr, n, k, s->dseed, dBound, s->dcand, 0));
            else
                CKF(run_mine(d, d->f_mine, d->smem, dAn, dBn, m, n, k, s->dseed, dBound, s->dcand, 0));
            CKF(cu.MemcpyDtoHAsync(s->hcand, s->dcand, CAND_WORDS * 4, d->st));
            CKF(cu.EventRecord(s->ev, d->st));
            s->busy = 1;
            cur ^= 1;
            slot_t *o = &slots[cur];
            if (o->busy) collect(d, o, work_run);
        }
        for (int i = 0; i < 2; i++)
            if (slots[i].busy) collect(d, &slots[i], work_run);
        job_release(job);
    }
    M->dev[d->idx].active = 0;
    return NULL;
}

int gpu_launch(gpu_dev *d) {
    if (g_ndevs >= MAX_DEV) return -1;
    g_devs[g_ndevs++] = d;
    return pthread_create(&d->th, NULL, gpu_main, d);
}

void gpu_join_all(void) {
    for (int i = 0; i < g_ndevs; i++) pthread_join(g_devs[i]->th, NULL);
}

// ------------------------------------------------------------------ --bench-mma
extern const char prl_ptx_sm80_bench[], prl_ptx_sm75_bench[];
void gpu_bench_mma(int idx, const gpu_info *info) {
    int cc = info->cc_major * 10 + info->cc_minor;
    if (cc < 75) { LOGW("GPU%d %s: no tensor cores (cc %d.%d)", idx, info->name, info->cc_major, info->cc_minor); return; }
    gpu_dev dd = {0}, *d = &dd;
    d->idx = idx;
    CUdevice dv;
    CUmodule mod;
    CUfunction f[2];
    CUdeviceptr out = 0;
    CUevent e0, e1;
    CK(cu.DeviceGet(&dv, info->ordinal));
    CK(cu.CtxCreate(&d->ctx, CU_CTX_SCHED_BLOCKING_SYNC, dv));
    CK(cu.StreamCreate(&d->st, 0));
    if (load_module(d, &mod, cc >= 80 ? prl_ptx_sm80_bench : prl_ptx_sm75_bench, "bench")) goto fail;
    CK(cu.ModuleGetFunction(&f[0], mod, "k_peak_s8"));
    CK(cu.ModuleGetFunction(&f[1], mod, "k_peak_s4"));
    CK(cu.MemAlloc(&out, 64));
    CK(cu.EventCreate(&e0, 0));
    CK(cu.EventCreate(&e1, 0));
    static const char *kname[2] = {"s8", "s4"};
    double peak[2] = {0, 0};
    char cfg[2][16] = {"", ""};
    for (int kind = 0; kind < 2; kind++) {
        char line[512];
        int pos = snprintf(line, sizeof line, "GPU%d %s", idx, kname[kind]);
        static const unsigned TH[] = {128, 256, 512, 1024};
        static const unsigned BPS[] = {1, 2};
        for (int ti = 0; ti < 4; ti++)
            for (int bi = 0; bi < 2; bi++) {
                unsigned threads = TH[ti], bps = BPS[bi], grid = (unsigned)info->sms * bps;
                if (threads * bps > 1024 && cc < 80) continue;
                if (threads * bps > 1536) continue;
                unsigned iters = 64;
                float ms = 0;
                for (int rep = 0; rep < 4; rep++) {
                    void *args[] = {&iters, &out};
                    CK(cu.EventRecord(e0, d->st));
                    CK(launch_t(d, f[kind], grid, 1, threads, 0, args));
                    CK(cu.EventRecord(e1, d->st));
                    CK(cu.EventSynchronize(e1));
                    cu.EventElapsedTime(&ms, e0, e1);
                    if (ms <= 0) break;
                    if (ms < 20) iters = (unsigned)(iters * (30.0 / ms)) + 1;   // grow to ~30 ms
                    else break;
                }
                double macs = (double)grid * (threads / 32) * iters * 8 * (kind ? 8192.0 : 4096.0);
                double t = ms > 0 ? macs / (ms * 1e-3) / 1e12 : 0;
                if (t > peak[kind]) { peak[kind] = t; snprintf(cfg[kind], sizeof cfg[kind], "%ux%u", threads / 32, bps); }
                pos += snprintf(line + pos, sizeof line - pos, " %ux%u:%.1f", threads / 32, bps, t);
            }
        LOGD("%s", line);   // full table with --debug
    }
    LOGI("GPU%d %s (%d SM, cc %d.%d): peak s8 %.1f TH/s, s4 %.1f TH/s", idx, info->name, info->sms, info->cc_major,
         info->cc_minor, peak[0], peak[1]);
    {   // breakdown of the production kernel at full size: each entry switches parts of the work off
        const unsigned m = 128u * info->sms, n = 1024, k = 8192;
        const bool t = cc < 80;   // Turing: t2b, else v4c
        const v2_variant *vb = t ? &V2_SM75[12] : &V2_SM80[7];
        const unsigned smem = v2_smem(vb, 16), threads = vb->threads;
        CUmodule m2;
        if (load_module(d, &m2, cc >= 80 ? prl_ptx_sm80_v2_16 : prl_ptx_sm75_v2_16, "v2")) goto fail;
        CUdeviceptr dU = 0, dP = 0, dB = 0, dS = 0, dBound = 0, dCand = 0;
        CK(cu.MemAlloc(&dU, (size_t)m * 128)); CK(cu.MemAlloc(&dP, 2 * k)); CK(cu.MemAlloc(&dB, (size_t)n * k));
        CK(cu.MemAlloc(&dS, 32)); CK(cu.MemAlloc(&dBound, 32)); CK(cu.MemAlloc(&dCand, CAND_WORDS * 4));
        CK(cu.MemsetD32Async(dU, 0x01020304, (size_t)m * 32, d->st));
        CK(cu.MemsetD32Async(dP, 0x00050003, k / 2, d->st));
        CK(cu.MemsetD32Async(dB, 0x05fb03fd, (size_t)n * k / 4, d->st));
        CK(cu.MemsetD32Async(dS, 7, 8, d->st));
        CK(cu.MemsetD32Async(dBound, 0, 8, d->st));
        static const char *what[] = {"full", "-gen", "-B", "-fold", "-gen-B", "mma-only"};
        const char *fn4[] = {"k_v4c_nt16", "k_dv4c_1", "k_dv4c_2", "k_dv4c_8", "k_dv4c_3", "k_dv4c_15"};
        const char *fnt[] = {"k_t2b_nt16", "k_dt2b_1", "k_dt2b_2", "k_dt2b_8", "k_dt2b_3", "k_dt2b_15"};
        char bl[640];
        int bpos = snprintf(bl, sizeof bl, "GPU%d %s parts TH/s (spill bytes):", idx, t ? "t2b" : "v4c");
        for (int i = 0; i < 6; i++) {
            CUfunction f2;
            if (cu.ModuleGetFunction(&f2, m2, t ? fnt[i] : fn4[i]) ||
                (smem > 48 * 1024 && cu.FuncSetAttribute(f2, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)))
                continue;
            int lmem = 0;
            cu.FuncGetAttribute(&lmem, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f2);
            float ms = 0, best = 1e30f;
            for (int rep = 0; rep < 4; rep++) {
                CK(cu.MemsetD32Async(dCand, 0, 4, d->st));
                CK(cu.EventRecord(e0, d->st));
                CK(run_mine_fused(d, f2, threads, smem, dU, dP, 1, dB, m, n, k, dS, dBound, dCand, 0));
                CK(cu.EventRecord(e1, d->st));
                CK(cu.EventSynchronize(e1));
                cu.EventElapsedTime(&ms, e0, e1);
                if (rep && ms < best) best = ms;
            }
            bpos += snprintf(bl + bpos, sizeof bl - bpos, " %s %.1f", what[i],
                             best > 0 && best < 1e29f ? (double)m * n * k / (best * 1e-3) / 1e12 : 0);
            if (lmem) bpos += snprintf(bl + bpos, sizeof bl - bpos, "(%d)", lmem);
        }
        LOGI("%s", bl);
        CUdeviceptr fr[] = {dU, dP, dB, dS, dBound, dCand};
        for (size_t i = 0; i < 6; i++) cu.MemFree(fr[i]);
    }
fail:
    if (out) cu.MemFree(out);
    if (d->ctx) cu.CtxDestroy(d->ctx);
}

