"""Make pseudo-random uint16 words in the exact same format as fetch_bits.py.

Usage:
  python qrng/make_pseudo.py --words 51200 --seed 20261002 --out bits/pseudo_a.npy
"""
import argparse

import numpy as np

from common import save_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.Generator(np.random.PCG64(a.seed))
    words = rng.integers(0, 2**16, size=a.words, dtype=np.uint16)
    print(save_words(words, a.out, source="numpy PCG64",
                     extra={"seed": a.seed}))


if __name__ == "__main__":
    main()
