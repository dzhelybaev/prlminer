// Minimal CUDA driver API loader (dlopen libcuda.so.1 — no toolkit/runtime needed).
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef unsigned long long CUdeviceptr;
typedef struct CUevent_st *CUevent;

#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#define CU_DEVICE_ATTRIBUTE_PCI_BUS_ID 33
#define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID 34
#define CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID 50
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN 97
#define CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE 38
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES 8
#define CU_FUNC_ATTRIBUTE_NUM_REGS 4
#define CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES 3
#define CU_JIT_INFO_LOG_BUFFER 3
#define CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES 4
#define CU_JIT_ERROR_LOG_BUFFER 5
#define CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES 6
#define CU_CTX_SCHED_BLOCKING_SYNC 0x04

typedef struct {
    CUresult (*Init)(unsigned);
    CUresult (*DriverGetVersion)(int *);
    CUresult (*DeviceGetCount)(int *);
    CUresult (*DeviceGet)(CUdevice *, int);
    CUresult (*DeviceGetName)(char *, int, CUdevice);
    CUresult (*DeviceGetAttribute)(int *, int, CUdevice);
    CUresult (*DeviceTotalMem)(size_t *, CUdevice);
    CUresult (*CtxCreate)(CUcontext *, unsigned, CUdevice);
    CUresult (*CtxDestroy)(CUcontext);
    CUresult (*CtxSetCurrent)(CUcontext);
    CUresult (*CtxSynchronize)(void);
    CUresult (*ModuleLoadDataEx)(CUmodule *, const void *, unsigned, int *, void **);
    CUresult (*ModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*FuncSetAttribute)(CUfunction, int, int);
    CUresult (*FuncGetAttribute)(int *, int, CUfunction);
    CUresult (*MemAlloc)(CUdeviceptr *, size_t);
    CUresult (*MemFree)(CUdeviceptr);
    CUresult (*MemAllocHost)(void **, size_t);
    CUresult (*MemFreeHost)(void *);
    CUresult (*MemcpyHtoD)(CUdeviceptr, const void *, size_t);
    CUresult (*MemcpyDtoH)(void *, CUdeviceptr, size_t);
    CUresult (*MemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
    CUresult (*MemcpyDtoHAsync)(void *, CUdeviceptr, size_t, CUstream);
    CUresult (*MemsetD32Async)(CUdeviceptr, unsigned, size_t, CUstream);
    CUresult (*MemGetInfo)(size_t *, size_t *);
    CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                             CUstream, void **, void **);
    CUresult (*StreamCreate)(CUstream *, unsigned);
    CUresult (*StreamSynchronize)(CUstream);
    CUresult (*GetErrorString)(CUresult, const char **);
    CUresult (*EventCreate)(CUevent *, unsigned);
    CUresult (*EventRecord)(CUevent, CUstream);
    CUresult (*EventSynchronize)(CUevent);
    CUresult (*EventElapsedTime)(float *, CUevent, CUevent);
} cuda_api;

extern cuda_api cu;
int cuda_load(char *err, size_t errlen);   // 0 ok
const char *cu_errstr(CUresult r);
