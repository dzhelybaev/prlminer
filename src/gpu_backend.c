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
typedef struct { const char *name; int mt, threads, bks, stg, jsm; } v2_variant;   // name: "v2a" ...
static const v2_variant V2_SM80[] = {{"v2a", 0, 256, 1, 4, 9}, {"v2b", 0, 256, 2, 2, 9}, {"v2c", 0, 256, 1, 3, 0},
                                     {"v3a", 1, 512, 1, 3, 9}, {"v3b", 1, 512, 2, 2, 9}};
static const v2_variant V2_SM75[] = {{"v2a", 0, 256, 1, 2, 1}, {"v2b", 0, 256, 1, 2, 1}, {"v2c", 0, 256, 1, 2, 0},
                                     {"v3a", 1, 512, 1, 2, 0}, {"v3b", 1, 512, 1, 2, 0}};
static unsigned v2_smem(const v2_variant *v, int nt) {
    int mt = v->mt ? v->mt : 32 / nt, js = v->jsm < mt ? v->jsm : mt;
    return (unsigned)(v->stg * v->bks * 12288 + 16384 + 16384 * js * (v->threads / 256));
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
    CUstream st;
    CUfunction f_uni, f_perm, f_apply, f_mine, f_fused;
    unsigned smem, smem_fused;
    bool mma, use_fused;
    struct { CUfunction f; unsigned smem, threads; char name[16]; } fv[8];   // self-tested fused variants
    unsigned threads_fused;
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
    return cu.LaunchKernel(f, gx, gy, 1, threads, 1, 1, smem, d->st, args, NULL);
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

static CUresult run_mine(gpu_dev *d, CUfunction f, unsigned smem, CUdeviceptr An, CUdeviceptr Bn, unsigned m,
                         unsigned n, unsigned k, CUdeviceptr seed, CUdeviceptr bound, CUdeviceptr cand, CUdeviceptr dbg) {
    unsigned cap = CAND_CAP;
    void *args[] = {&An, &Bn, &m, &n, &k, &seed, &bound, &cand, &cap, &dbg};
    return launch(d, f, n / BN, m / BM, smem, args);
}

// fused: A' generated in the kernel from the uniform rows + pairs (base A must be zero)
static CUresult run_mine_fused(gpu_dev *d, CUfunction f, unsigned threads, unsigned smem, CUdeviceptr U, CUdeviceptr P, uint64_t nonce,
                               CUdeviceptr Bn, unsigned m, unsigned n, unsigned k, CUdeviceptr seed, CUdeviceptr bound,
                               CUdeviceptr cand, CUdeviceptr dbg) {
    unsigned cap = CAND_CAP, lo = (unsigned)nonce, hi = (unsigned)(nonce >> 32);
    void *args[] = {&U, &P, &lo, &hi, &Bn, &m, &n, &k, &seed, &bound, &cand, &cap, &dbg};
    return launch_t(d, f, n / BN, m / BM, threads, smem, args);
}

// ------------------------------------------------------------------ self-test
// Small problem (m=128, n=256, k=2048): GPU noise + transcripts must match the CPU core bit-for-bit.
static int selftest_kernel(gpu_dev *d, CUfunction fmine, unsigned smem, const char *name, bool fused, unsigned threads) {
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
        if (fused) CK(run_mine_fused(d, fmine, threads, smem, dUa, dPa, nonce, dBn, m, n, k, dSa, dBound, dCand, dDbg));
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

// ------------------------------------------------------------------ prepare
// load function <prefix><nt> from module, self-test it, and add it to the fused candidates
static void add_fused(gpu_dev *d, CUmodule mod, const char *prefix, unsigned smem, const char *name, int maxsm,
                      unsigned threads) {
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
    if (!selftest_kernel(d, f, smem, name, true, threads)) {
        LOGW("GPU%d: kernel %s failed the self-test, not used", d->idx, name);
        return;
    }
    int regs = 0, lmem = 0;
    cu.FuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, f);
    cu.FuncGetAttribute(&lmem, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f);
    LOGI("GPU%d: kernel %s-nt%d (fused noise): %d registers, %d bytes local memory (spills)", d->idx, name,
         g_miner.nt, regs, lmem);
    if (d->nfv < 8) {
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
    {
        bool sm80 = cc >= 80;
        CUmodule mn;
        if (load_module(d, &mn, ptx_for(sm80, "noise"), "noise")) goto fail;
        CK(cu.ModuleGetFunction(&d->f_uni, mn, "k_noise_uniform"));
        CK(cu.ModuleGetFunction(&d->f_perm, mn, "k_noise_perm"));
        CK(cu.ModuleGetFunction(&d->f_apply, mn, "k_apply_noise"));
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
                    add_fused(d, mf, "k_mine_fmma_nt", smem_mma + FUSED_SMEM, "fmma", maxsm, 256);
                if (g_miner.nt == 16 || g_miner.nt == 32) {
                    const char *ptx = sm80 ? (g_miner.nt == 16 ? prl_ptx_sm80_v2_16 : prl_ptx_sm80_v2_32)
                                           : (g_miner.nt == 16 ? prl_ptx_sm75_v2_16 : prl_ptx_sm75_v2_32);
                    CUmodule m2;
                    if (!load_module(d, &m2, ptx, "v2")) {
                        const v2_variant *vt = sm80 ? V2_SM80 : V2_SM75;
                        for (int i = 0; i < 5; i++) {
                            if (vt[i].mt && 32 / g_miner.nt != 2) continue;   // v3 exists for nt16 only
                            char fn[32];
                            snprintf(fn, sizeof fn, "k_%s_nt", vt[i].name);
                            add_fused(d, m2, fn, v2_smem(&vt[i], g_miner.nt), vt[i].name, maxsm, vt[i].threads);
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

static void *gpu_main(void *arg) {
    gpu_dev *d = arg;
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
    const uint64_t work = (uint64_t)m * n * k;
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
        float best = (g_fused_mode > 0 && d->nfv) ? 1e30f : tn + tg;
        int sel = -1;
        for (int v = 0; v < d->nfv; v++) {
            float tf = 0, tfn = 0, tmin = 1e30f;
            for (int it = 0; it < 4; it++) {
                noise_bufs na = {dA, dUa, dPa, dAn, slots[0].dseed};
                CKF(cu.EventRecord(e0, d->st));
                CKF(run_noise(d, &na, 0, m, k, 1, 1, false));
                CKF(cu.EventRecord(e1, d->st));
                CKF(run_mine_fused(d, d->fv[v].f, d->fv[v].threads, d->fv[v].smem, dUa, dPa, 1, dBn, m, n, k, slots[0].dseed, dBound,
                                   slots[0].dcand, 0));
                CKF(cu.EventRecord(e2, d->st));
                CKF(cu.EventSynchronize(e2));
                cu.EventElapsedTime(&tfn, e0, e1);
                cu.EventElapsedTime(&tf, e1, e2);
                if (it && tfn + tf < tmin) tmin = tfn + tf;   // first run is a warm-up
            }
            LOGI("GPU%d: profile: %-5s attempt %.2f ms (%sH/s)", d->idx, d->fv[v].name, tmin,
                 fmt_rate(tmin > 0 ? (double)m * n * k / (tmin * 1e-3) : 0, b1, sizeof b1));
            if (tmin < best) { best = tmin; sel = v; }
        }
        for (int v = 0; g_variant && v < d->nfv; v++)
            if (!strcmp(d->fv[v].name, g_variant)) { sel = v; best = 0; }
        if (sel >= 0) {
            d->use_fused = true;
            d->f_fused = d->fv[sel].f;
            d->smem_fused = d->fv[sel].smem;
            d->threads_fused = d->fv[sel].threads;
            snprintf(M->dev[d->idx].kernel, sizeof M->dev[d->idx].kernel, "%s-nt%d", d->fv[sel].name, M->nt);
        }
        LOGI("GPU%d: using kernel %s (%.2f ms per attempt)", d->idx, M->dev[d->idx].kernel, best);
    }
    M->dev[d->idx].active = 1;
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
            if (s->busy) collect(d, s, work);
            s->nonce = miner_next_nonce();
            uint8_t chunk0[PRL_CHUNK];
            miner_attempt_chunk0(job->work, d->geom, s->nonce, chunk0, s->hseed);
            s->job = job;
            atomic_fetch_add(&job->refs, 1);
            CKF(cu.MemcpyHtoDAsync(s->dseed, s->hseed, 32, d->st));
            CKF(cu.MemsetD32Async(s->dcand, 0, 4, d->st));
            noise_bufs na = {dA, dUa, dPa, dAn, s->dseed};
            CKF(run_noise(d, &na, 0, m, k, s->nonce, 1, !d->use_fused));
            if (d->use_fused)
                CKF(run_mine_fused(d, d->f_fused, d->threads_fused, d->smem_fused, dUa, dPa, s->nonce, dBn, m, n, k, s->dseed, dBound,
                                   s->dcand, 0));
            else
                CKF(run_mine(d, d->f_mine, d->smem, dAn, dBn, m, n, k, s->dseed, dBound, s->dcand, 0));
            CKF(cu.MemcpyDtoHAsync(s->hcand, s->dcand, CAND_WORDS * 4, d->st));
            CKF(cu.EventRecord(s->ev, d->st));
            s->busy = 1;
            cur ^= 1;
            slot_t *o = &slots[cur];
            if (o->busy) collect(d, o, work);
        }
        for (int i = 0; i < 2; i++)
            if (slots[i].busy) collect(d, &slots[i], work);
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
    LOGI("GPU%d: %s, %d SMs, cc %d.%d — peak mma TH/s for (warps per block)x(blocks per SM):", idx,
         info->name, info->sms, info->cc_major, info->cc_minor);
    static const char *kname[2] = {"s8 (mining)", "s4"};
    for (int kind = 0; kind < 2; kind++) {
        char line[512];
        int pos = snprintf(line, sizeof line, "GPU%d %-12s", idx, kname[kind]);
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
                pos += snprintf(line + pos, sizeof line - pos, " | %2ux%u: %6.2f", threads / 32, bps, t);
            }
        LOGI("%s", line);
    }
    {   // breakdown of the v2b mining kernel at full size: each line switches parts of the work off
        const unsigned m = 128u * info->sms, n = 2048, k = 8192;
        const v2_variant *vb = cc >= 80 ? &V2_SM80[1] : &V2_SM75[1];
        unsigned smem = v2_smem(vb, 16), threads = 256;
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
        static const struct { const char *fn, *what; } DK[] = {
            {"k_v2b_nt16", "v2b full kernel"},           {"k_diag1_nt16", "- noise generation"},
            {"k_diag2_nt16", "- B loads"},               {"k_diag3_nt16", "- generation - B loads"},
            {"k_diag7_nt16", "- gen - loads - barriers"}, {"k_diag15_nt16", "- gen - loads - bar - fold"},
            {"k_diag8_nt16", "- fold only"}};
        for (size_t i = 0; i < sizeof DK / sizeof DK[0]; i++) {
            CUfunction f2;
            if (cu.ModuleGetFunction(&f2, m2, DK[i].fn) ||
                (smem > 48 * 1024 && cu.FuncSetAttribute(f2, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)))
                continue;
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
            LOGI("GPU%d breakdown %-28s %6.3f ms  %7.2f TH/s", idx, DK[i].what, best,
                 best > 0 ? (double)m * n * k / (best * 1e-3) / 1e12 : 0);
        }
        CUdeviceptr fr[] = {dU, dP, dB, dS, dBound, dCand};
        for (size_t i = 0; i < 6; i++) cu.MemFree(fr[i]);
    }
fail:
    if (out) cu.MemFree(out);
    if (d->ctx) cu.CtxDestroy(d->ctx);
}

