# prlminer build. Host code is plain C (C11) + POSIX threads; CUDA kernels are
# compiled to PTX with clang (no CUDA toolkit needed) and embedded in the binary.
CC      ?= gcc
CLANG   ?= clang
CFLAGS  ?= -O2 -g
CFLAGS  += -MMD -MP -std=gnu11 -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -pthread -fno-strict-aliasing
LDFLAGS += -pthread
LDLIBS  += -ldl -lm

B3DIR   = third_party/blake3
B3SRC   = $(B3DIR)/blake3.c $(B3DIR)/blake3_dispatch.c $(B3DIR)/blake3_portable.c
B3ASM   = $(B3DIR)/blake3_sse2_x86-64_unix.S $(B3DIR)/blake3_sse41_x86-64_unix.S \
          $(B3DIR)/blake3_avx2_x86-64_unix.S $(B3DIR)/blake3_avx512_x86-64_unix.S

CORE    = src/pearl.c
HOST    = src/main.c src/json.c src/stratum.c src/miner.c src/cpu_backend.c src/gpu_backend.c \
          src/cuda_drv.c src/api.c src/log.c src/gz.c

BUILD   = build
OBJ_B3  = $(patsubst %.c,$(BUILD)/%.o,$(B3SRC)) $(patsubst %.S,$(BUILD)/%.o,$(B3ASM))
OBJ_CORE= $(patsubst %.c,$(BUILD)/%.o,$(CORE))
OBJ_HOST= $(patsubst %.c,$(BUILD)/%.o,$(HOST)) $(BUILD)/kernels_ptx.o

PTX_ARCH ?= sm_80
PTX_FEAT ?= +ptx70

all: $(BUILD)/prlminer

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/%.o: %.S
	@mkdir -p $(dir $@)
	$(CC) -c $< -o $@

# ---- CUDA -> PTX. One module per kernel so the driver only JIT-compiles what a GPU uses.
#   sm_80 PTX (cp.async + mma.sync tensor cores): RTX 30xx / 40xx / 50xx, A100, H100 ...
#   sm_61 PTX (dp4a): GTX 10xx / 16xx, RTX 20xx
PTX80 = noise mma32 mma16 mma8 dp4a32 dp4a16 dp4a8
PTX61 = noise dp4a32 dp4a16 dp4a8
PTX75 = mma32 mma16 mma8
KDEF_noise = -DK_NOISE
KDEF_mma32 = -DK_MMA32
KDEF_mma16 = -DK_MMA16
KDEF_mma8 = -DK_MMA8
KDEF_dp4a32 = -DK_DP4A32
KDEF_dp4a16 = -DK_DP4A16
KDEF_dp4a8 = -DK_DP4A8
CUFLAGS = -x cuda --cuda-device-only -nocudainc -nocudalib -O3 -std=c++17 -S

$(BUILD)/ptx/sm80_%.ptx: cuda/kernels.cu cuda/common.cuh
	@mkdir -p $(dir $@)
	$(CLANG) $(CUFLAGS) --cuda-gpu-arch=sm_80 -Xclang -target-feature -Xclang +ptx70 -DPRL_MMA=1 $(KDEF_$*) -o $@ $<

$(BUILD)/ptx/sm61_%.ptx: cuda/kernels.cu cuda/common.cuh
	@mkdir -p $(dir $@)
	$(CLANG) $(CUFLAGS) --cuda-gpu-arch=sm_61 -Xclang -target-feature -Xclang +ptx60 -DPRL_MMA=0 $(KDEF_$*) -o $@ $<

# Turing (sm_75): tensor cores via mma.m8n8k16, synchronous copies
$(BUILD)/ptx/sm75_%.ptx: cuda/kernels.cu cuda/common.cuh
	@mkdir -p $(dir $@)
	$(CLANG) $(CUFLAGS) --cuda-gpu-arch=sm_75 -Xclang -target-feature -Xclang +ptx65 -DPRL_MMA=1 -DPRL_TURING=1 $(KDEF_$*) -o $@ $<

PTX_FILES = $(patsubst %,$(BUILD)/ptx/sm80_%.ptx,$(PTX80)) $(patsubst %,$(BUILD)/ptx/sm61_%.ptx,$(PTX61)) \
            $(patsubst %,$(BUILD)/ptx/sm75_%.ptx,$(PTX75))

$(BUILD)/kernels_ptx.c: $(PTX_FILES) tools/embed.py
	python3 tools/embed.py $@ $(foreach k,$(PTX80),prl_ptx_sm80_$(k)=$(BUILD)/ptx/sm80_$(k).ptx) \
	  $(foreach k,$(PTX61),prl_ptx_sm61_$(k)=$(BUILD)/ptx/sm61_$(k).ptx) \
	  $(foreach k,$(PTX75),prl_ptx_sm75_$(k)=$(BUILD)/ptx/sm75_$(k).ptx)

$(BUILD)/kernels_ptx.o: $(BUILD)/kernels_ptx.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/prlminer: $(OBJ_HOST) $(OBJ_CORE) $(OBJ_B3)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# ---- tests
$(BUILD)/test_core: tests/test_core.c $(OBJ_CORE) $(OBJ_B3)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

$(BUILD)/libprlcore.so: $(CORE) $(B3SRC) $(B3ASM) tests/pycore.c
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $^ $(LDFLAGS)

test: $(BUILD)/test_core
	$(BUILD)/test_core

clean:
	rm -rf $(BUILD)

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)

.PHONY: all test clean

$(BUILD)/test_kernels: tests/test_kernels.cpp tests/emu.cpp cuda/kernels.cu cuda/common.cuh $(OBJ_CORE) $(OBJ_B3)
	$(CXX) -O2 -g -std=c++17 -pthread -Wno-unused-parameter -o $@ tests/test_kernels.cpp tests/emu.cpp $(OBJ_CORE) $(OBJ_B3) -pthread

$(BUILD)/fakecuda/libcuda.so.1: tests/fake_cuda.cpp tests/emu.cpp cuda/kernels.cu cuda/common.cuh
	@mkdir -p $(dir $@)
	$(CXX) -O2 -g -std=c++17 -shared -fPIC -pthread -Wno-unused-parameter -o $@ tests/fake_cuda.cpp tests/emu.cpp

# ---- portable binary: runs on glibc >= 2.17 (HiveOS), see tools/mkstubs.py
OBJ_ALL = $(OBJ_HOST) $(OBJ_CORE) $(OBJ_B3)
$(BUILD)/stubs/.done: $(BUILD)/prlminer tools/mkstubs.py
	python3 tools/mkstubs.py $(BUILD)/prlminer $(BUILD)/stubs
	@touch $@

$(BUILD)/prlminer-portable: $(BUILD)/stubs/.done $(OBJ_ALL)
	$(CC) -pie -nodefaultlibs -o $@ $(OBJ_ALL) $(wildcard $(BUILD)/stubs/*.so.*) -lgcc
	@python3 tools/check_glibc.py $@

portable: $(BUILD)/prlminer-portable
.PHONY: portable

# ---- HiveOS package
VERSION = 1.0.5
package: $(BUILD)/prlminer-portable
	rm -rf $(BUILD)/pkg && mkdir -p $(BUILD)/pkg/prlminer
	cp $(BUILD)/prlminer-portable $(BUILD)/pkg/prlminer/prlminer
	cp hiveos/h-manifest.conf hiveos/h-config.sh hiveos/h-run.sh hiveos/h-stats.sh README.md LICENSE $(BUILD)/pkg/prlminer/
	chmod +x $(BUILD)/pkg/prlminer/prlminer $(BUILD)/pkg/prlminer/h-*.sh
	tar -C $(BUILD)/pkg -czf $(BUILD)/prlminer-$(VERSION).tar.gz --owner=0 --group=0 prlminer
	tar -czf $(BUILD)/prlminer-src-$(VERSION).tar.gz --owner=0 --group=0 --exclude=build --exclude=__pycache__ --transform 's,^\.,prlminer-src-$(VERSION),' .
	@ls -la $(BUILD)/prlminer-$(VERSION).tar.gz $(BUILD)/prlminer-src-$(VERSION).tar.gz
.PHONY: package
