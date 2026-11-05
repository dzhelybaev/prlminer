"""Independent pure-Python Pearl PlainProof verifier (port of zk-pow verify_plain_proof),
written separately from the C code so the two implementations check each other.
Pure-Python BLAKE3 included (only what Pearl needs)."""
import base64
import struct

import numpy as np

IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]
PERM = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
CHUNK_START, CHUNK_END, PARENT, ROOT, KEYED_HASH = 1, 2, 4, 8, 16
M32 = 0xFFFFFFFF


def _g(v, a, b, c, d, x, y):
    v[a] = (v[a] + v[b] + x) & M32; v[d] ^= v[a]; v[d] = ((v[d] >> 16) | (v[d] << 16)) & M32
    v[c] = (v[c] + v[d]) & M32; v[b] ^= v[c]; v[b] = ((v[b] >> 12) | (v[b] << 20)) & M32
    v[a] = (v[a] + v[b] + y) & M32; v[d] ^= v[a]; v[d] = ((v[d] >> 8) | (v[d] << 24)) & M32
    v[c] = (v[c] + v[d]) & M32; v[b] ^= v[c]; v[b] = ((v[b] >> 7) | (v[b] << 25)) & M32


def compress(cv, block, counter, blen, flags):
    m = list(struct.unpack("<16I", block.ljust(64, b"\0")))
    v = list(cv) + IV[:4] + [counter & M32, counter >> 32, blen, flags]
    for r in range(7):
        _g(v, 0, 4, 8, 12, m[0], m[1]); _g(v, 1, 5, 9, 13, m[2], m[3])
        _g(v, 2, 6, 10, 14, m[4], m[5]); _g(v, 3, 7, 11, 15, m[6], m[7])
        _g(v, 0, 5, 10, 15, m[8], m[9]); _g(v, 1, 6, 11, 12, m[10], m[11])
        _g(v, 2, 7, 8, 13, m[12], m[13]); _g(v, 3, 4, 9, 14, m[14], m[15])
        m = [m[i] for i in PERM]
    return [v[i] ^ v[i + 8] for i in range(8)]


def words(b):
    return list(struct.unpack("<8I", b))


def wbytes(w):
    return struct.pack("<8I", *w)


def chunk_cv(key_words, data, counter, flags_base, root=False):
    """CV of one chunk (<= 1024 bytes)."""
    cv = key_words
    blocks = [data[i:i + 64] for i in range(0, max(len(data), 1), 64)] or [b""]
    for i, blk in enumerate(blocks):
        f = flags_base
        if i == 0: f |= CHUNK_START
        if i == len(blocks) - 1:
            f |= CHUNK_END
            if root: f |= ROOT
        cv = compress(cv, blk, counter, len(blk), f)
    return cv


def hash_bytes(data, key=None):
    """BLAKE3 for inputs up to 1024 bytes (single chunk)."""
    assert len(data) <= 1024
    kw = words(key) if key else IV
    return wbytes(chunk_cv(kw, data, 0, KEYED_HASH if key else 0, root=True))


def parent(key_words, l, r, root=False):
    return wbytes(compress(key_words, l + r, 0, 64, KEYED_HASH | PARENT | (ROOT if root else 0)))


# ------------------------------------------------------------------ periodic pattern (zk-pow port)
def pattern_from_list(p):
    assert p[0] == 0 and all(a < b for a, b in zip(p, p[1:]))
    p = list(p); shape = []
    while len(p) > 1:
        for period in range(1, len(p)):
            if len(p) % period == 0:
                s = p[period]
                if all(p[i] + s == p[i + period] for i in range(len(p) - period)):
                    shape.append((s, len(p) // period)); p = p[:period]; break
        else:
            raise ValueError("not periodic")
    shape.reverse()
    per = shape[-1][0] * shape[-1][1] if shape else 1
    while len(shape) < 3: shape.append((per, 1))
    return shape


def pattern_bytes(shape):
    out, ms = b"", 1
    for st, ln in shape:
        out += bytes([st // ms - 1, ln - 1]); ms = st * ln
    return out


# ------------------------------------------------------------------ bincode reader
class R:
    def __init__(self, b): self.b, self.o = b, 0
    def u64(self):
        v = struct.unpack_from("<Q", self.b, self.o)[0]; self.o += 8; return v
    def raw(self, n):
        v = self.b[self.o:self.o + n]; assert len(v) == n; self.o += n; return v


def read_mproof(r):
    nl = r.u64(); leaves = []
    for _ in range(nl):
        assert r.u64() == 1024; leaves.append(r.raw(1024))
    ni = r.u64(); assert ni == nl
    idx = [r.u64() for _ in range(ni)]
    total = r.u64(); root = r.raw(32)
    ns = r.u64(); sibs = [r.raw(32) for _ in range(ns)]
    nr = r.u64(); rows = [r.u64() for _ in range(nr)]
    return dict(leaves=leaves, idx=idx, total=total, root=root, sibs=sibs, rows=rows)


def merkle_root(mp, key):
    kw = words(key)
    cur = {i: wbytes(chunk_cv(kw, d, i, KEYED_HASH)) for i, d in zip(mp["idx"], mp["leaves"])}
    sib = iter(mp["sibs"]); ln = mp["total"]
    while ln > 2:
        nx = {}
        for i in sorted(cur):
            if i % 2 == 0:
                if i + 1 in cur: nx[i // 2] = parent(kw, cur[i], cur[i + 1])
                elif i + 1 < ln: nx[i // 2] = parent(kw, cur[i], next(sib))
                else: nx[i // 2] = cur[i]
            elif i - 1 not in cur:
                nx[i // 2] = parent(kw, next(sib), cur[i])
        cur = nx; ln = (ln + 1) // 2
    l = cur.get(0) or next(sib); r_ = cur.get(1) or next(sib)
    assert next(sib, None) is None, "unused siblings"
    return parent(kw, l, r_, root=True)


def extract(mp, start, n):
    out = bytearray(n); got = 0
    for i, d in zip(mp["idx"], mp["leaves"]):
        ls, le = i * 1024, i * 1024 + 1024
        s, e = max(start, ls), min(start + n, le)
        if s < e: out[s - start:e - start] = d[s - ls:e - ls]; got += e - s
    assert got == n, "strip not covered"
    return np.frombuffer(bytes(out), dtype=np.int8)


SALT_A = hash_bytes(b"pearl/cert-v3/noise-seed/A")
SALT_B = hash_bytes(b"pearl/cert-v3/noise-seed/B")
LABEL_A = b"A_tensor" + b"\0" * 24
LABEL_B = b"B_tensor" + b"\0" * 24


def noise_hash(idx, label, key, slot):
    msg = bytearray(64); struct.pack_into("<i", msg, slot * 4, 1 + idx); msg[32:] = label
    return hash_bytes(bytes(msg), key)


def uniform_row(label, key, row, r=128):
    out = b"".join(noise_hash(row * (r // 32) + b, label, key, 0) for b in range(r // 32))
    return (np.frombuffer(out, np.uint8) & 63).astype(np.int16) - 32


def perm(label, key, k, r=128):
    f, s = np.zeros(k, np.int64), np.zeros(k, np.int64)
    for i in range((k + 7) // 8):
        h = struct.unpack("<8I", noise_hash(i, label, key, 1))
        for j in range(8):
            l = 8 * i + j
            if l >= k: break
            u = h[j]; a = u & (r - 1); b = a ^ (1 + (((r - 1) * u) >> 32))
            f[l], s[l] = a, b
    return f, s


def verify(proof, header, salted, target):
    """Returns (ok, reason, jackpot_hash_int)."""
    r = R(proof)
    m, n, k, rank = r.u64(), r.u64(), r.u64(), r.u64()
    pa, pb = read_mproof(r), read_mproof(r)
    if r.raw(1) != b"\0" or r.o != len(proof): return False, "trailing/moe", 0
    if rank != 128: return False, "rank", 0
    if not (k % 64 == 0 and k >= 16 * rank and k >= 1024 and k <= 4 * rank * rank): return False, "k", 0
    if pa["total"] != -(-m * k // 1024) or pb["total"] != -(-n * k // 1024): return False, "tree size", 0
    tr, tc = pa["rows"][0], pb["rows"][0]
    rs, cs = pattern_from_list([x - tr for x in pa["rows"]]), pattern_from_list([x - tc for x in pb["rows"]])
    h, w = len(pa["rows"]), len(pb["rows"])
    if not (32 <= h * w <= 256): return False, "tile size", 0
    if not (tr + max(x - tr for x in pa["rows"]) < m and tc + max(x - tc for x in pb["rows"]) < n): return False, "range", 0
    cfg = struct.pack("<IHH", k, rank, 0) + pattern_bytes(rs) + pattern_bytes(cs) + b"\0" * 32
    job_key = hash_bytes(header + cfg)
    if merkle_root(pa, job_key) != pa["root"]: return False, "A root", 0
    if merkle_root(pb, job_key) != pb["root"]: return False, "B root", 0
    ha, hb = pa["root"], pb["root"]
    if salted:
        ha = hash_bytes(ha + struct.pack("<I", m) + b"\0" * 28, SALT_A)
        hb = hash_bytes(hb + struct.pack("<I", n) + b"\0" * 28, SALT_B)
    bseed = hash_bytes(job_key + hb)
    aseed = hash_bytes(bseed + ha)
    A = np.stack([extract(pa, x * k, k) for x in pa["rows"]]).astype(np.int32)
    B = np.stack([extract(pb, x * k, k) for x in pb["rows"]]).astype(np.int32)
    if A.min() < -64 or A.max() > 64 or B.min() < -64 or B.max() > 64: return False, "values", 0
    fa, sa = perm(LABEL_A, aseed, k)
    for i, row in enumerate(pa["rows"]):
        u = uniform_row(LABEL_A, aseed, row); A[i] += u[fa] - u[sa]
    fb, sb = perm(LABEL_B, bseed, k)
    for i, col in enumerate(pb["rows"]):
        u = uniform_row(LABEL_B, bseed, col); B[i] += u[fb] - u[sb]
    jp = [0] * 16
    acc = np.zeros((h, w), np.int64)
    for c in range(k // rank):
        acc += A[:, c * rank:(c + 1) * rank] @ B[:, c * rank:(c + 1) * rank].T
        x = 0
        for v in (acc.astype(np.int64) & M32).flatten().tolist(): x ^= v
        t = c % 16; jp[t] = (((jp[t] << 13) | (jp[t] >> 19)) & M32) ^ x
    hh = hash_bytes(struct.pack("<16I", *jp), aseed)
    hv = int.from_bytes(hh, "little")
    factor = h * w * (k - k % rank) // rank * 128
    if hv > target * factor: return False, "low difficulty", hv
    return True, "ok", hv


def verify_b64(b64, header, salted, target):
    return verify(base64.b64decode(b64), header, salted, target)
