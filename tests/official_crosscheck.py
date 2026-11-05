"""Cross-check the C core against the OFFICIAL Pearl Python miner code
(pearl-research-labs/pearl miner/miner-base), executed unmodified on top of a
small numpy-backed `torch` shim and a `blake3` shim (backed by the official
BLAKE3 C library inside libprlcore.so).

Checks:
  1. NoiseGenerator.generate_noise_metrices  ==  C noise rows (A and B)
  2. NoisyGemm transcript (jackpot words) for the first hash tile == C jackpot,
     and keyed jackpot hash, on noised operands produced by the official code.

usage: python3 tests/official_crosscheck.py <pearl-repo> <libprlcore.so>
"""
import ctypes
import sys
import types

import numpy as np

PEARL, LIB = sys.argv[1], sys.argv[2]
lib = ctypes.CDLL(LIB)

# ----------------------------------------------------------------- blake3 shim
class _B3:
    digest_size = 32

    def __init__(self, data=b"", key=None):
        self.data, self.key = bytes(data), key

    def update(self, d):
        self.data += bytes(d)

    def digest(self):
        out = ctypes.create_string_buffer(32)
        if self.key is None:
            lib.py_hash(self.data, ctypes.c_size_t(len(self.data)), out)
        else:
            lib.py_keyed_hash(bytes(self.key), self.data, ctypes.c_size_t(len(self.data)), out)
        return out.raw


b3mod = types.ModuleType("blake3")
b3mod.blake3 = _B3
sys.modules["blake3"] = b3mod

# ----------------------------------------------------------------- torch shim
class Tensor(np.ndarray):
    def to(self, dtype):
        return np.asarray(self).astype(dtype).view(Tensor)

    def numpy(self):
        return np.asarray(self)

    def view(self, *shape):
        if len(shape) == 1 and not isinstance(shape[0], int):
            return np.ndarray.view(self, shape[0])
        return np.asarray(self).reshape(shape).view(Tensor)

    def contiguous(self):
        return np.ascontiguousarray(self).view(Tensor)

    def flatten(self):
        return np.asarray(self).flatten().view(Tensor)


def _t(a):
    return np.asarray(a).view(Tensor)


torch = types.ModuleType("torch")
torch.int8, torch.int32, torch.uint8 = np.dtype("int8"), np.dtype("int32"), np.dtype("uint8")
torch.Tensor = Tensor
torch.frombuffer = lambda buf, dtype: _t(np.frombuffer(bytes(buf), dtype=dtype).copy())
torch.zeros = lambda *shape, dtype=None: _t(np.zeros(shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape, dtype=dtype))
torch.matmul = lambda a, b: _t(np.matmul(np.asarray(a), np.asarray(b)))
torch.any = lambda x: bool(np.any(x))
torch.all = lambda x: bool(np.all(x))
torch.from_numpy = _t
torch.tensor = lambda x, dtype=None: _t(np.array(x, dtype=dtype))
sys.modules["torch"] = torch

# ----------------------------------------------------------------- stub deps
mu = types.ModuleType("miner_utils")
import logging
mu.get_logger = lambda name: logging.getLogger(name)
sys.modules["miner_utils"] = mu
for name in ["pearl_gateway", "pearl_gateway.comm", "pearl_gateway.comm.dataclasses",
             "pearl_gateway.comm.mining_configuration", "miner_base.matmul_config"]:
    sys.modules[name] = types.ModuleType(name)


class OpenedBlockInfo:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class CommitmentHash:
    def __init__(self, a, b):
        self.noise_seed_A, self.noise_seed_B = a, b


sys.modules["pearl_gateway.comm.dataclasses"].OpenedBlockInfo = OpenedBlockInfo
sys.modules["pearl_gateway.comm.dataclasses"].CommitmentHash = CommitmentHash
sys.modules["pearl_gateway.comm.mining_configuration"].PeriodicPattern = object
sys.modules["miner_base.matmul_config"].MatmulConfig = object

sys.path.insert(0, PEARL + "/miner/miner-base/src")
import importlib.util


def load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


pkg = types.ModuleType("miner_base")
pkg.__path__ = [PEARL + "/miner/miner-base/src/miner_base"]
sys.modules["miner_base"] = pkg
ng_mod = load("miner_base.noise_generation", PEARL + "/miner/miner-base/src/miner_base/noise_generation.py")
ih_mod = load("miner_base.inner_hash", PEARL + "/miner/miner-base/src/miner_base/inner_hash.py")
gm_mod = load("miner_base.noisy_gemm", PEARL + "/miner/miner-base/src/miner_base/noisy_gemm.py")

fails = 0


def check(cond, msg):
    global fails
    if not cond:
        fails += 1
        print("FAIL:", msg)


rng = np.random.default_rng(1)
for trial, (m, k, n) in enumerate([(128, 2048, 256), (160, 4096, 128), (128, 2304, 192)]):
    key_A = bytes(rng.integers(0, 256, 32, dtype=np.uint8))
    key_B = bytes(rng.integers(0, 256, 32, dtype=np.uint8))
    g = ng_mod.NoiseGenerator(noise_rank=128, noise_range=128)
    A_L, A_R, B_L, B_R = g.generate_noise_metrices(key_A, key_B, m, k, n)
    E_A = np.matmul(np.asarray(A_L, np.int32), np.asarray(A_R, np.int32)).astype(np.int8)   # m x k
    E_B = np.matmul(np.asarray(B_L, np.int32), np.asarray(B_R, np.int32)).astype(np.int8)   # k x n

    rows = (ctypes.c_uint32 * m)(*range(m))
    outA = np.zeros((m, k), np.int8)
    lib.py_noise_rows(0, key_A, k, rows, m, outA.ctypes.data_as(ctypes.c_void_p))
    check(np.array_equal(outA, E_A), f"trial {trial}: noise A mismatch")
    cols = (ctypes.c_uint32 * n)(*range(n))
    outB = np.zeros((n, k), np.int8)
    lib.py_noise_rows(1, key_B, k, cols, n, outB.ctypes.data_as(ctypes.c_void_p))
    check(np.array_equal(outB, E_B.T), f"trial {trial}: noise B mismatch")

    # jackpot: run the official NoisyGemm with the official noise, capture the first transcript
    A = rng.integers(-64, 64, (m, k), dtype=np.int8)
    B = rng.integers(-64, 64, (k, n), dtype=np.int8)
    for hh, hw in [(2, 16), (2, 64), (16, 16)]:
        gem = gm_mod.NoisyGemm(noise_range=128, noise_rank=128, hash_tile_h=hh, hash_tile_w=hw,
                               matmul_tile_h=128, matmul_tile_w=128)
        captured = {}
        orig = gem._check_pow_target

        def cap(transcript, pow_key, pow_target, _orig=orig):
            if "words" not in captured:
                captured["words"] = [int(x) for x in transcript.data]
                captured["key"] = pow_key
            return _orig(transcript, pow_key, pow_target)

        gem._check_pow_target = cap
        C, found = gem.noisy_gemm(_t(A), _t(B), A_L, A_R, B_L, B_R,
                                  commitment_hash=CommitmentHash(key_A, key_B),
                                  pow_target=2 ** 256 - 1)
        check(found, "official found with easiest target")
        check(np.array_equal(np.asarray(C), A.astype(np.int32) @ B.astype(np.int32)), "denoise identity")
        info = gem.get_opened_block_info()
        r0, c0 = info.A_row_indices[0], info.B_column_indices[0]
        check((r0, c0) == (0, 0), "first tile is (0,0)")
        An = (A.astype(np.int32) + E_A).astype(np.int8)
        Bn = (B.astype(np.int32) + E_B).astype(np.int8)
        a_rows = np.ascontiguousarray(An[0:hh, :])
        b_cols = np.ascontiguousarray(Bn[:, 0:hw].T)
        jp = (ctypes.c_uint32 * 16)()
        lib.py_jackpot(a_rows.ctypes.data_as(ctypes.c_void_p), hh, b_cols.ctypes.data_as(ctypes.c_void_p), hw, k, jp)
        check(list(jp) == captured["words"], f"trial {trial} tile {hh}x{hw}: jackpot words mismatch")
        # keyed hash of the transcript with noise_seed_A
        msg = b"".join(int(w).to_bytes(4, "little") for w in jp)
        check(_B3(msg, key=key_A).digest() == _B3(msg, key=captured["key"]).digest(), "pow key is noise_seed_A")
    print(f"trial {trial}: m={m} k={k} n={n} noise A/B + jackpot match official")

if fails:
    print(f"{fails} FAILURES")
    sys.exit(1)
print("OFFICIAL CROSS-CHECK PASSED")
