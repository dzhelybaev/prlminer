#!/usr/bin/env python3
"""Build link-time stub libraries so a binary linked on a new glibc only requires
glibc <= 2.17 at run time (HiveOS images are based on older Ubuntu releases).

The symbols that moved into libc in glibc 2.34 (pthread_create, dlopen, ...) are bound
to their pre-2.34 homes (libpthread.so.0 / libdl.so.2, version GLIBC_2.2.5), exactly as a
binary built on an old system would be. Those old bindings keep working on new glibc
(the stub libraries still exist there for backward compatibility).

usage: mkstubs.py BINARY OUTDIR    (BINARY = normally linked prlminer)"""
import os
import re
import subprocess
import sys

binary, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
MAXVER = (2, 17)

# version index -> library file
ver = subprocess.run(["readelf", "-V", "-W", binary], capture_output=True, text=True, check=True).stdout
idx2file, cur = {}, None
for line in ver.splitlines():
    m = re.search(r"File: (\S+)", line)
    if m: cur = m.group(1)
    m = re.search(r"Name: (\S+)\s+Flags: \S+\s+Version: (\d+)", line)
    if m and cur: idx2file[m.group(1)] = cur

syms = subprocess.run(["readelf", "--dyn-syms", "-W", binary], capture_output=True, text=True, check=True).stdout
MOVED_PTHREAD = {"pthread_create", "pthread_join", "pthread_detach", "pthread_attr_setstacksize",
                 "pthread_barrier_init", "pthread_barrier_wait", "pthread_barrier_destroy", "pthread_kill"}
MOVED_DL = {"dlopen", "dlsym", "dlclose", "dlerror", "dladdr"}
libs = {}   # file -> {version: [(sym, kind)]}
for line in syms.splitlines():
    p = line.split()
    if len(p) < 8 or "@" not in p[7] or (p[6] != "UND" and p[3] != "OBJECT"):
        continue
    kind = p[3]
    sym, v = p[7].split("@", 1)
    v = v.split(" ")[0].lstrip("@")
    f = idx2file.get(v, "libc.so.6")
    if sym in MOVED_PTHREAD:
        f, v = "libpthread.so.0", "GLIBC_2.2.5"
    elif sym in MOVED_DL:
        f, v = "libdl.so.2", "GLIBC_2.2.5"
    elif sym == "__libc_start_main":
        v = "GLIBC_2.2.5"
    if v.startswith("GLIBC_"):
        nums = tuple(int(x) for x in v[6:].split("."))
        if nums[:2] > MAXVER:
            sys.exit(f"symbol {sym}@{v} needs glibc newer than 2.17 — avoid it")
    libs.setdefault(f, {}).setdefault(v, []).append((sym, kind))
libs.setdefault("libc.so.6", {}).setdefault("GLIBC_2.2.5", [])

for f, vers in libs.items():
    base = os.path.join(out, f.replace(".so", "_so"))
    with open(base + ".c", "w") as c:
        for v, lst in vers.items():
            for s, kind in lst:
                c.write(f"char {s}[8];\n" if kind == "OBJECT" else f"void {s}(void) {{}}\n")
    def key(v):
        return tuple(int(x) for x in re.findall(r"\d+", v))
    with open(base + ".map", "w") as m:
        for v in sorted(vers, key=key):
            names = "".join(f"{s}; " for s, _ in vers[v])
            m.write(f"{v} {{ global: {names}}};\n")
    subprocess.run(["gcc", "-w", "-shared", "-nostdlib", "-fPIC", f"-Wl,-soname,{f}", f"-Wl,--version-script={base}.map",
                    "-o", os.path.join(out, f), base + ".c"], check=True)
    print(f"stub {f}: " + ", ".join(f"{v}:{len(l)}" for v, l in vers.items()))
