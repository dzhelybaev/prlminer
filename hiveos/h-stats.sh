#!/usr/bin/env bash
# Sourced by the HiveOS agent: must set $khs and $stats.
. "$(dirname "${BASH_SOURCE[0]}")/h-manifest.conf"

stats_raw=$(curl --connect-timeout 2 --max-time 5 --silent --noproxy '*' "http://127.0.0.1:${MINER_API_PORT}/")
if [[ $? -ne 0 || -z $stats_raw ]]; then
  echo -e "${YELLOW}Failed to read ejjjminer API on 127.0.0.1:${MINER_API_PORT}${NOCOLOR}"
  khs=0
  stats="null"
else
  gs_json="${gpu_stats:-}"
  [[ -z $gs_json ]] && gs_json='{}'
  # Pearl hashrates are huge in H/s (1 H = one int8 multiply-accumulate), so report kH/s.
  khs=$(jq -r '(.hashrate // 0) / 1000' <<< "$stats_raw")
  stats=$(jq -c --argjson gs "$gs_json" --arg ver "$CUSTOM_VERSION" '
    ($gs.busids // []) as $bus |
    def pick($arr; $b): (($bus | index($b)) as $i | if $i == null then 0 else ($arr[$i] // 0) end);
    {
      hs: [.gpus[] | (.hashrate // 0) / 1000],
      hs_units: "khs",
      temp: [.gpus[] | .busid as $b | pick($gs.temp // []; $b)],
      fan: [.gpus[] | .busid as $b | pick($gs.fan // []; $b)],
      bus_numbers: [.gpus[] | .bus],
      uptime: (.uptime // 0),
      ver: $ver,
      algo: "pearlhash",
      ar: [(.accepted // 0), (.rejected // 0)]
    }' <<< "$stats_raw" 2>/dev/null)
fi
[[ -z $khs ]] && khs=0
[[ -z $stats ]] && stats="null"
