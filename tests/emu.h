// Minimal SIMT emulator used to run cuda/kernels.cu on the CPU (tests only).
#pragma once
#include <functional>
typedef unsigned int u32;
typedef int i32;
typedef unsigned char u8;

u32 emu_tid();
u32 emu_bid_x();
u32 emu_bid_y();
u8 *emu_smem();
void emu_bar_sync();
void emu_mma_s8(i32 d[4], const u32 a[4], const u32 b[2]);
void emu_mma_m8n8k16(i32 *d, u32 a, u32 b);   // Turing fragment model
void emu_ldmatrix_x4(u32 out[4], const void *row_addr);   // ldmatrix.sync.m8n8.x4.b16

// Runs body() for every thread of every block (blocks sequentially, threads concurrently).
// Same, but threads run one after another in the caller (only for kernels without barriers).
void emu_launch_serial(u32 gx, u32 gy, u32 block, const std::function<void()> &body);
void emu_launch(u32 gx, u32 gy, u32 block, size_t smem_bytes, const std::function<void()> &body);
