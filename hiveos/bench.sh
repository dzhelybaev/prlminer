#!/usr/bin/env bash
# ejbench: tensor-core micro-benchmark from a HiveOS worker command (stops the miner, runs, starts it again).
cd "$(dirname "$(readlink -f "$0")")"
restart() { miner start >/dev/null 2>&1; }
trap restart EXIT
miner stop >/dev/null 2>&1
for i in $(seq 1 20); do pgrep -x ejjjminer >/dev/null || break; sleep 1; done
./ejjjminer --bench-mma --plain 2>&1 | grep -v "JIT"
echo "(miner restarted)"
