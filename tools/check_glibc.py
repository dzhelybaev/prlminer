#!/usr/bin/env python3
"""Fail if a binary needs glibc symbol versions newer than 2.17."""
import re, subprocess, sys
out = subprocess.run(["readelf", "-V", "-W", sys.argv[1]], capture_output=True, text=True).stdout
vers = sorted({tuple(int(x) for x in v.split(".")) for v in re.findall(r"Name: GLIBC_([\d.]+)", out)})
files = re.findall(r"File: (\S+)", out)
print("needs:", ", ".join(files), "| max GLIBC_%s" % ".".join(map(str, vers[-1])))
sys.exit(1 if vers[-1][:2] > (2, 17) else 0)
