"""Mock Pearl stratum pool (object dialect: authorize -> notify -> submit {job_id, plain_proof}).
Every submitted share is verified with the independent Python verifier (pyverify.py).

usage: mock_pool.py PORT DIFF SHARES [--salted 0|1] [--height H]
Exits 0 after SHARES valid shares (switching to a new block halfway), 1 on any invalid share."""
import argparse
import base64
import gzip
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import pyverify  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("port", type=int)
ap.add_argument("diff", type=float)
ap.add_argument("shares", type=int)
ap.add_argument("--height", type=int, default=120000)
ap.add_argument("--timeout", type=float, default=600)
ap.add_argument("--nogzip", action="store_true")
args = ap.parse_args()
salted = args.height >= 99000

target = int(2 ** 224 / args.diff)
srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", args.port))
srv.listen(1)
srv.settimeout(args.timeout)
print(f"[pool] listening on {args.port}, diff {args.diff}, target {target:064x}", flush=True)
conn, _ = srv.accept()
conn.settimeout(args.timeout)
f = conn.makefile("rwb")


def send(obj):
    f.write((json.dumps(obj) + "\n").encode()); f.flush()


jobs = {}
height = args.height


def new_job(n):
    hdr = os.urandom(76)
    jid = f"{os.urandom(4).hex()}_{n}"
    jobs[jid] = hdr
    send({"id": None, "method": "mining.notify",
          "params": {"job_id": jid, "header": hdr.hex(), "target": f"{target:064x}", "height": height}})
    return jid


good = bad = stale = 0
cur = None
t0 = time.time()
while True:
    line = f.readline()
    if not line:
        break
    msg = json.loads(line)
    meth, params = msg.get("method"), msg.get("params")
    if meth == "mining.authorize":
        assert isinstance(params, dict) and params.get("wallet"), params
        print(f"[pool] authorize wallet={params['wallet']} worker={params.get('worker')} agent={params.get('agent')}", flush=True)
        gz = params.get("type") == "v2" and not args.nogzip
        cur = new_job(0)
        reply = {"id": msg["id"], "result": True, "error": None}
        if gz:
            reply["type"] = "v2"
        send(reply)
    elif meth == "mining.submit":
        jid, b64 = params["job_id"], params["plain_proof"]
        if jid not in jobs or jid != cur:
            stale += 1
            send({"id": msg["id"], "result": None, "error": {"code": 21, "msg": "stale job"}})
            print(f"[pool] stale share for {jid}", flush=True)
            continue
        ts = time.time()
        raw = base64.b64decode(b64)
        if gz:
            raw = gzip.decompress(raw)
        ok, why, hv = pyverify.verify(raw, jobs[jid], salted, target)
        print(f"[pool] share job={jid} {len(b64)}B b64 (proof {len(raw)}B, gzip={gz}) -> {why} (verified in {time.time() - ts:.1f}s)", flush=True)
        if ok:
            good += 1
            send({"id": msg["id"], "result": True, "error": None})
        else:
            bad += 1
            send({"id": msg["id"], "result": None, "error": {"code": 26 if why != "low difficulty" else 23, "msg": why}})
        if good == args.shares // 2 and ok:
            height += 1
            cur = new_job(good)
            print(f"[pool] new block {height}, job {cur}", flush=True)
        if good >= args.shares or bad:
            break
print(f"[pool] done: good={good} bad={bad} stale={stale} in {time.time() - t0:.0f}s", flush=True)
conn.close()
sys.exit(0 if bad == 0 and good >= args.shares else 1)
