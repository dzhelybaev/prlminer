#!/usr/bin/env bash
# ejprof: compact summary of the last miner start (kernel profile, chosen kernel, current hashrate).
LOG=${EJLOG:-/var/log/miner/custom/ejjjminer.log}
[[ -f $LOG ]] || { echo "no log $LOG"; exit 1; }
start=$(grep -n "pearlhash, no dev fee" "$LOG" | tail -1 | cut -d: -f1)
tail -n +"${start:-1}" "$LOG" | sed -E 's/^\[[^]]*\] (INF|DBG) //; s/^\[[^]]*\] (WRN|ERR) /! /' | awk '
function num(s) { gsub(/[^0-9.]/, "", s); return s }
function gpu(s) { sub(/:$/, "", s); if (!(s in seen)) { seen[s] = 1; order[++ng] = s } return s }
/^ejjjminer [0-9]/ { print $1, $2; next }
/^GPU[0-9]+: NVIDIA/ {
    g = gpu($1); line = $0; sub(/^GPU[0-9]+: NVIDIA (GeForce )?/, "", line); sub(/ \[.*matrix /, " ", line)
    head[g] = line; next }
/^GPU[0-9]+: L2 / { g = gpu($1); l = $0; sub(/^GPU[0-9]+: /, "", l); l2[g] = l; next }
/^GPU[0-9]+: kernel .* registers/ {
    g = gpu($1); n = $3; sub(/-nt[0-9]+$/, "", n)
    for (i = 1; i <= NF; i++) if ($(i+1) == "bytes" && $(i+2) == "local") sp = $i
    if (sp + 0 > 0) spill[g] = spill[g] " " n ":" sp; next }
/^GPU[0-9]+: profile: attempt .*GEMM alone/ { g = gpu($1); prof[g] = prof[g] " mma(gemm " num($(NF-1)) ")"; next }
/^GPU[0-9]+: profile: .* pipelined/ {
    g = gpu($1); n = $3; r = $0; sub(/^.* pipelined [^(]*\(/, "", r); sub(/ .*/, "", r)
    ga = $0; sub(/^.*GEMM alone /, "", ga); sub(/ .*/, "", ga)
    prof[g] = prof[g] " " n " " r "/g" ga; next }
/^GPU[0-9]+: profile: [a-z0-9]+ +attempt/ {
    g = gpu($1); r = $0; sub(/^.*\(/, "", r); sub(/ .*/, "", r); prof[g] = prof[g] " " $3 " " r; next }
/^GPU[0-9]+: using kernel/ { g = gpu($1); u = $0; sub(/^GPU[0-9]+: using kernel /, "", u); sub(/, matrix.*/, ")", u); use[g] = u; next }
/^GPU[0-9]+: batches of / { g = gpu($1); bt[g] = $4; next }
/^total / { tot = $0; next }
/^! / { w[++nw] = $0; next }
END {
    for (i = 1; i <= ng; i++) {
        g = order[i]; print g " " head[g]
        if (l2[g] != "") print "  " l2[g]
        if (prof[g] != "") print " " prof[g]
        if (spill[g] != "") print "  spills:" spill[g]
        if (use[g] != "") print "  => " use[g] (bt[g] != "" ? ", batch " bt[g] : "")
    }
    for (i = (nw > 6 ? nw - 5 : 1); i <= nw; i++) print w[i]
    if (tot != "") print tot
}'
