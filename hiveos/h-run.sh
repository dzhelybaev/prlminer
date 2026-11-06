#!/usr/bin/env bash
cd "$(dirname "$0")"
. h-manifest.conf

if pgrep -x ejjjminer >/dev/null; then
  echo -e "${RED}ejjjminer is already running${NOCOLOR}"
  exit 1
fi
mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"
# phone-friendly worker commands: ejbench (benchmark), ejprof (profile summary)
ln -sf "$PWD/bench.sh" /usr/local/bin/ejbench 2>/dev/null
ln -sf "$PWD/prof.sh" /usr/local/bin/ejprof 2>/dev/null
unset LD_LIBRARY_PATH
conf=$(cat "$CUSTOM_CONFIG_FILENAME")

# Restart after a GPU fault (exit code 42 = CUDA error, e.g. unstable overclock).
while true; do
  ./ejjjminer $conf --log "$CUSTOM_LOG_BASENAME.log"
  code=$?
  [[ $code -eq 0 || $code -eq 1 || $code -eq 2 || $code -eq 3 ]] && exit $code
  echo "ejjjminer exited with code $code, restarting in 10 s"
  sleep 10
done
