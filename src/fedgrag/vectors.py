"""Deterministic hashed token vectors (no model, no downloads, CPU only)."""
import hashlib
import numpy as np

DIM = 256


def _hash_token(token: str):
    h = hashlib.sha256(token.encode("utf-8")).digest()
    idx = int.from_bytes(h[:4], "little") % DIM
    sign = 1.0 if (h[4] % 2 == 0) else -1.0
    return idx, sign


def vec_from_tokens(tokens_weights):
    """tokens_weights: iterable of (token, weight). Returns L2-normalized vector."""
    v = np.zeros(DIM, dtype=np.float64)
    for token, w in tokens_weights:
        idx, sign = _hash_token(token)
        v[idx] += sign * w
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
