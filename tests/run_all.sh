#!/usr/bin/env bash
# Full test suite (no GPU needed). Optional: PEARL_REPO=/path/to/pearl for the official cross-check.
set -e
cd "$(dirname "$0")/.."
make -s all portable build/test_core build/test_kernels build/libprlcore.so build/fakecuda/libcuda.so.1
echo "== core (BLAKE3 vectors, Merkle trees, pinned commitment vectors, patterns, proofs)"
./build/test_core
if [[ -n "$PEARL_REPO" ]]; then
  echo "== cross-check vs official Pearl Python miner code"
  python3 tests/official_crosscheck.py "$PEARL_REPO" build/libprlcore.so
fi
echo "== CUDA kernels in the SIMT emulator vs CPU reference"
./build/test_kernels
echo "== Turing (mma.m8n8k16) kernels in the emulator"
g++ -O2 -std=c++17 -pthread -DPRL_TURING=1 -DPRL_STAGES=2 -Wno-unused-parameter -o build/test_kernels_turing \
  tests/test_kernels.cpp tests/emu.cpp build/src/pearl.o build/third_party/blake3/*.o
./build/test_kernels_turing
echo "== GPU self-test through the real host backend (emulated libcuda)"
LD_LIBRARY_PATH=build/fakecuda ./build/ejjjminer-portable --selftest
FAKE_CUDA_CC=75 LD_LIBRARY_PATH=build/fakecuda ./build/ejjjminer-portable --selftest --tile 16
e2e() {   # name port diff args...
  local name=$1 port=$2 diff=$3; shift 3
  python3 tests/mock_pool.py $port $diff 4 > build/pool_$name.log 2>&1 &
  local pool=$!
  sleep 1
  timeout 600 "$@" -o 127.0.0.1:$port -u prl1qtestwallet.$name --cols 256 --k 2048 --api-port 0 > build/miner_$name.log 2>&1 &
  local miner=$!
  wait $pool && rc=0 || rc=$?
  kill $miner 2>/dev/null || true
  tail -1 build/pool_$name.log
  return $rc
}
echo "== end-to-end: CPU backend -> mock pool (independent Python verifier)"
e2e cpu 34101 0.01 ./build/ejjjminer-portable --cpu
echo "== end-to-end: GPU backend (emulated) -> mock pool"
LD_LIBRARY_PATH=build/fakecuda e2e gpu 34102 0.05 ./build/ejjjminer-portable --rows 128
echo "ALL TESTS PASSED"
