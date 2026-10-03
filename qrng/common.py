"""Shared bit-file format for BOTH conditions.

Every condition (quantum or pseudo) is stored the same way:
  bits/<name>.npy        raw uint16 words
  bits/<name>.json       manifest: source, count, sha256, created

Uniforms are made the same way for both: two uint16 words -> one 32-bit
integer -> u in [0, 1). Same code path for A and B is the whole point.
"""
import hashlib
import json
import os
import time

import numpy as np


def sha256_of(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def save_words(words, path_npy, source, extra=None):
    words = np.asarray(words, dtype=np.uint16)
    os.makedirs(os.path.dirname(path_npy) or ".", exist_ok=True)
    np.save(path_npy, words)
    manifest = {"source": source, "count": int(words.size),
                "sha256": sha256_of(words),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if extra:
        manifest.update(extra)
    with open(path_npy.replace(".npy", ".json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_words(path_npy, check=True):
    words = np.load(path_npy)
    if words.dtype != np.uint16:
        raise ValueError(f"{path_npy}: expected uint16, got {words.dtype}")
    mpath = path_npy.replace(".npy", ".json")
    if check and os.path.exists(mpath):
        m = json.load(open(mpath))
        if m["sha256"] != sha256_of(words):
            raise ValueError(f"{path_npy}: sha256 mismatch, file changed")
    return words


def words_to_uniforms(words):
    """Pairs of uint16 -> 32-bit ints -> u in [0, 1). Drops an odd last word."""
    w = np.asarray(words, dtype=np.uint64)
    n = (w.size // 2) * 2
    hi, lo = w[0:n:2], w[1:n:2]
    return ((hi << np.uint64(16)) | lo).astype(np.float64) / 2.0**32


class UniformStream:
    """Hands out uniforms in order. Raises when empty: NEVER falls back to another RNG."""

    def __init__(self, path_npy):
        self.path = path_npy
        self.u = words_to_uniforms(load_words(path_npy))
        self.i = 0

    def next(self):
        if self.i >= self.u.size:
            raise RuntimeError(f"{self.path}: out of uniforms after {self.i}")
        x = float(self.u[self.i])
        self.i += 1
        return x

    def remaining(self):
        return int(self.u.size - self.i)
