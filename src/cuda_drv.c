#include "cuda_drv.h"
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

cuda_api cu;

int cuda_load(char *err, size_t errlen) {
    static int loaded;
    if (loaded) return 0;
    void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) { snprintf(err, errlen, "cannot load libcuda.so.1 (NVIDIA driver not installed?)"); return -1; }
#define SYM(field, name)                                                                     \
    do {                                                                                     \
        *(void **)&cu.field = dlsym(h, name);                                                \
        if (!cu.field) { snprintf(err, errlen, "libcuda: missing symbol %s", name); return -1; } \
    } while (0)
    SYM(Init, "cuInit");
    SYM(DriverGetVersion, "cuDriverGetVersion");
    SYM(DeviceGetCount, "cuDeviceGetCount");
    SYM(DeviceGet, "cuDeviceGet");
    SYM(DeviceGetName, "cuDeviceGetName");
    SYM(DeviceGetAttribute, "cuDeviceGetAttribute");
    SYM(DeviceTotalMem, "cuDeviceTotalMem_v2");
    SYM(CtxCreate, "cuCtxCreate_v2");
    SYM(CtxDestroy, "cuCtxDestroy_v2");
    SYM(CtxSetCurrent, "cuCtxSetCurrent");
    SYM(CtxSynchronize, "cuCtxSynchronize");
    SYM(ModuleLoadDataEx, "cuModuleLoadDataEx");
    SYM(ModuleGetFunction, "cuModuleGetFunction");
    SYM(FuncSetAttribute, "cuFuncSetAttribute");
    SYM(FuncGetAttribute, "cuFuncGetAttribute");
    SYM(MemAlloc, "cuMemAlloc_v2");
    SYM(MemFree, "cuMemFree_v2");
    SYM(MemAllocHost, "cuMemAllocHost_v2");
    SYM(MemFreeHost, "cuMemFreeHost");
    SYM(MemcpyHtoD, "cuMemcpyHtoD_v2");
    SYM(MemcpyDtoH, "cuMemcpyDtoH_v2");
    SYM(MemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2");
    SYM(MemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2");
    SYM(MemsetD32Async, "cuMemsetD32Async");
    SYM(MemGetInfo, "cuMemGetInfo_v2");
    SYM(LaunchKernel, "cuLaunchKernel");
    SYM(StreamCreate, "cuStreamCreate");
    SYM(StreamSynchronize, "cuStreamSynchronize");
    SYM(GetErrorString, "cuGetErrorString");
    SYM(EventCreate, "cuEventCreate");
    SYM(EventRecord, "cuEventRecord");
    SYM(EventSynchronize, "cuEventSynchronize");
    SYM(EventElapsedTime, "cuEventElapsedTime");
#undef SYM
    CUresult r = cu.Init(0);
    if (r) { snprintf(err, errlen, "cuInit failed: %s", cu_errstr(r)); return -1; }
    loaded = 1;
    return 0;
}

const char *cu_errstr(CUresult r) {
    const char *s = NULL;
    if (cu.GetErrorString && cu.GetErrorString(r, &s) == 0 && s) return s;
    static char buf[32];
    snprintf(buf, sizeof buf, "CUDA error %d", r);
    return buf;
}
