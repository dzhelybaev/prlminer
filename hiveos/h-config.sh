#!/usr/bin/env bash
# Builds the command line from the flight sheet:
#   Pool URL            -> CUSTOM_URL       (several URLs separated by spaces = failover)
#   Wallet and worker   -> CUSTOM_TEMPLATE  (e.g. %WAL%.%WORKER_NAME%)
#   Pass                -> CUSTOM_PASS
#   Extra config args   -> CUSTOM_USER_CONFIG (e.g. --devices 0,1 --kernel auto)
[[ -z $CUSTOM_TEMPLATE ]] && echo -e "${YELLOW}ejjjminer: CUSTOM_TEMPLATE (wallet) is empty${NOCOLOR}" && return 1
[[ -z $CUSTOM_URL ]] && echo -e "${YELLOW}ejjjminer: CUSTOM_URL (pool) is empty${NOCOLOR}" && return 1

conf=""
for url in $CUSTOM_URL; do
  conf+=" -o $url"
done
conf+=" -u $CUSTOM_TEMPLATE"
[[ -n $CUSTOM_PASS ]] && conf+=" -p $CUSTOM_PASS"
conf+=" --api-port $MINER_API_PORT"
[[ -n $CUSTOM_USER_CONFIG ]] && conf+=" $CUSTOM_USER_CONFIG"

mkfile_from_symlink $CUSTOM_CONFIG_FILENAME
echo "$conf" > $CUSTOM_CONFIG_FILENAME
