#!/usr/bin/env bash
cd "$(dirname "$0")"
. h-manifest.conf

if pgrep -x prlminer >/dev/null; then
  echo -e "${RED}prlminer is already running${NOCOLOR}"
  exit 1
fi
mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"
unset LD_LIBRARY_PATH
conf=$(cat "$CUSTOM_CONFIG_FILENAME")

# Restart after a GPU fault (exit code 42 = CUDA error, e.g. unstable overclock).
while true; do
  ./prlminer $conf --log "$CUSTOM_LOG_BASENAME.log"
  code=$?
  [[ $code -eq 0 || $code -eq 1 || $code -eq 2 || $code -eq 3 ]] && exit $code
  echo "prlminer exited with code $code, restarting in 10 s"
  sleep 10
done
