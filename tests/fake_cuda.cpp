// Fake libcuda.so.1 for integration tests: host memory + the SIMT emulator running the real
// kernel source. Lets the unmodified GPU backend (src/gpu_backend.c) run on a machine without a GPU.
#define PRL_EMU 1
#include "../cuda/kernels.cu"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

typedef int CUresult;
typedef unsigned long long CUdeviceptr;

static const char *g_names[] = {"k_noise_uniform", "k_noise_perm", "k_apply_noise", "k_mine_mma_nt32",
                                "k_mine_mma_nt16", "k_mine_mma_nt8", "k_mine_dp4a_nt32", "k_mine_dp4a_nt16",
                                "k_mine_dp4a_nt8", "k_mine_fmma_nt32", "k_mine_fmma_nt16", "k_mine_fmma_nt8",
                                "k_v2a_nt32", "k_v2b_nt32", "k_v2c_nt32", "k_v2a_nt16", "k_v2b_nt16", "k_v2c_nt16", "k_v3a_nt16", "k_v3b_nt16"};
struct Func { int id; };
static Func g_funcs[20];

template <class T> static T A(void **p, int i) { return *(T *)p[i]; }

extern "C" {
CUresult cuInit(unsigned) { return 0; }
CUresult cuDriverGetVersion(int *v) { *v = 12040; return 0; }
CUresult cuDeviceGetCount(int *n) { const char *e = getenv("FAKE_CUDA_DEVICES"); *n = e ? atoi(e) : 1; return 0; }
CUresult cuDeviceGet(int *d, int i) { *d = i; return 0; }
CUresult cuDeviceGetName(char *s, int n, int d) { snprintf(s, n, "Fake GPU %d (emulator)", d); return 0; }
CUresult cuDeviceGetAttribute(int *v, int attr, int d) {
    switch (attr) {
    case 16: *v = 1; break;                       // SMs
    case 33: *v = 1 + d; break;                   // bus
    case 34: *v = 0; break;
    case 50: *v = 0; break;
    case 97: *v = (getenv("FAKE_CUDA_CC") && atoi(getenv("FAKE_CUDA_CC")) < 80) ? 65536 : 101376; break;
    case 75: *v = getenv("FAKE_CUDA_CC") ? atoi(getenv("FAKE_CUDA_CC")) / 10 : 8; break;
    case 76: *v = getenv("FAKE_CUDA_CC") ? atoi(getenv("FAKE_CUDA_CC")) % 10 : 6; break;
    default: *v = 0;
    }
    return 0;
}
CUresult cuDeviceTotalMem_v2(size_t *m, int) { *m = (size_t)8 << 30; return 0; }
CUresult cuCtxCreate_v2(void **c, unsigned, int) { *c = (void *)1; return 0; }
CUresult cuCtxDestroy_v2(void *) { return 0; }
CUresult cuCtxSetCurrent(void *) { return 0; }
CUresult cuCtxSynchronize() { return 0; }
CUresult cuModuleLoadDataEx(void **m, const void *, unsigned, int *, void **) { *m = (void *)1; return 0; }
CUresult cuModuleGetFunction(void **f, void *, const char *name) {
    for (int i = 0; i < 20; i++)
        if (!strcmp(name, g_names[i])) { g_funcs[i].id = i; *f = &g_funcs[i]; return 0; }
    return 500;   // CUDA_ERROR_NOT_FOUND
}
CUresult cuFuncSetAttribute(void *, int, int) { return 0; }
CUresult cuFuncGetAttribute(int *v, int, void *) { *v = 168; return 0; }
CUresult cuMemAlloc_v2(CUdeviceptr *p, size_t n) { *p = (CUdeviceptr)aligned_alloc(256, (n + 255) / 256 * 256); return *p ? 0 : 2; }
CUresult cuMemFree_v2(CUdeviceptr p) { free((void *)p); return 0; }
CUresult cuMemAllocHost_v2(void **p, size_t n) { *p = malloc(n); return 0; }
CUresult cuMemFreeHost(void *p) { free(p); return 0; }
CUresult cuMemcpyHtoD_v2(CUdeviceptr d, const void *s, size_t n) { memcpy((void *)d, s, n); return 0; }
CUresult cuMemcpyDtoH_v2(void *d, CUdeviceptr s, size_t n) { memcpy(d, (void *)s, n); return 0; }
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr d, const void *s, size_t n, void *) { memcpy((void *)d, s, n); return 0; }
CUresult cuMemcpyDtoHAsync_v2(void *d, CUdeviceptr s, size_t n, void *) { memcpy(d, (void *)s, n); return 0; }
CUresult cuMemsetD32Async(CUdeviceptr d, unsigned v, size_t n, void *) {
    for (size_t i = 0; i < n; i++) ((unsigned *)d)[i] = v;
    return 0;
}
CUresult cuMemGetInfo_v2(size_t *f, size_t *t) { *f = *t = (size_t)8 << 30; return 0; }
CUresult cuStreamCreate(void **s, unsigned) { *s = (void *)1; return 0; }
CUresult cuStreamSynchronize(void *) { return 0; }
CUresult cuGetErrorString(CUresult, const char **s) { *s = "fake error"; return 0; }
CUresult cuEventCreate(void **e, unsigned) { *e = (void *)1; return 0; }
CUresult cuEventRecord(void *, void *) { return 0; }
CUresult cuEventSynchronize(void *) { return 0; }
CUresult cuEventElapsedTime(float *t, void *, void *) { *t = 0; return 0; }

CUresult cuLaunchKernel(void *f, unsigned gx, unsigned gy, unsigned, unsigned bx, unsigned, unsigned, unsigned smem,
                        void *, void **p, void **) {
    int id = ((Func *)f)->id;
    if (bx != 256 && !(bx == 512 && id >= 18)) return 1;
    switch (id) {
    case 0: {
        auto key = A<const u32 *>(p, 0); auto lb = A<u32>(p, 1); auto nb = A<u32>(p, 2); auto out = A<i8 *>(p, 3);
        emu_launch_serial(gx, gy, bx, [&]() { k_noise_uniform(key, lb, nb, out); });
        break;
    }
    case 1: {
        auto key = A<const u32 *>(p, 0); auto lb = A<u32>(p, 1); auto k = A<u32>(p, 2); auto out = A<u8 *>(p, 3);
        emu_launch_serial(gx, gy, bx, [&]() { k_noise_perm(key, lb, k, out); });
        break;
    }
    case 2: {
        auto base = A<const i8 *>(p, 0); auto uni = A<const i8 *>(p, 1); auto pairs = A<const u8 *>(p, 2);
        auto out = A<i8 *>(p, 3); auto rows = A<u32>(p, 4); auto k = A<u32>(p, 5); auto lo = A<u32>(p, 6);
        auto hi = A<u32>(p, 7); auto patch = A<u32>(p, 8);
        emu_launch(gx, gy, bx, smem, [&]() { k_apply_noise(base, uni, pairs, out, rows, k, lo, hi, patch); });
        break;
    }
    case 9: case 10: case 11: case 12: case 13: case 14: case 15: case 16: case 17: case 18: case 19: {
        auto U = A<const i8 *>(p, 0); auto P = A<const u8 *>(p, 1); auto lo = A<u32>(p, 2); auto hi = A<u32>(p, 3);
        auto Bt = A<const i8 *>(p, 4); auto m = A<u32>(p, 5); auto n = A<u32>(p, 6); auto k = A<u32>(p, 7);
        auto seed = A<const u32 *>(p, 8); auto bound = A<const u32 *>(p, 9); auto cand = A<u32 *>(p, 10);
        auto cap = A<u32>(p, 11); auto dbg = A<u32 *>(p, 12);
        typedef void (*ff)(const i8 *, const u8 *, u32, u32, const i8 *, u32, u32, u32, const u32 *, const u32 *, u32 *,
                           u32, u32 *);
        ff fns[] = {k_mine_fmma_nt32, k_mine_fmma_nt16, k_mine_fmma_nt8, k_v2a_nt32, k_v2b_nt32, k_v2c_nt32,
                    k_v2a_nt16,       k_v2b_nt16,       k_v2c_nt16, k_v3a_nt16, k_v3b_nt16};
        ff fn = fns[id - 9];
        // the emulated kernels use the sm_80 layout; the host may have sized smem for Turing
        unsigned need = id >= 12 ? 131072 : STAGES * STAGE_BYTES + FUSED_SMEM;
        emu_launch(gx, gy, bx, smem < need ? need : smem,
                   [&]() { fn(U, P, lo, hi, Bt, m, n, k, seed, bound, cand, cap, dbg); });
        break;
    }
    default: {
        auto Am = A<const i8 *>(p, 0); auto Bt = A<const i8 *>(p, 1); auto m = A<u32>(p, 2); auto n = A<u32>(p, 3);
        auto k = A<u32>(p, 4); auto seed = A<const u32 *>(p, 5); auto bound = A<const u32 *>(p, 6);
        auto cand = A<u32 *>(p, 7); auto cap = A<u32>(p, 8); auto dbg = A<u32 *>(p, 9);
        typedef void (*mf)(const i8 *, const i8 *, u32, u32, u32, const u32 *, const u32 *, u32 *, u32, u32 *);
        mf fns[] = {k_mine_mma_nt32, k_mine_mma_nt16, k_mine_mma_nt8, k_mine_dp4a_nt32, k_mine_dp4a_nt16, k_mine_dp4a_nt8};
        mf fn = fns[id - 3];
        // emulated kernels are compiled with 3 stages; real sm_61 PTX uses 2
        emu_launch(gx, gy, bx, smem < 55296 ? 55296 : smem, [&]() { fn(Am, Bt, m, n, k, seed, bound, cand, cap, dbg); });
        break;
    }
    }
    return 0;
}
}
