"""Fetch quantum random words from ANU and save them to disk BEFORE any run.

Free key: 100 requests/month, 1 request/sec, up to 1024 numbers per request.
Get a key at https://quantumnumbers.anu.edu.au and set QRNG_API_KEY.
Verify the endpoint and header name against their API docs before first use.

Usage:
  set QRNG_API_KEY=your_key
  python qrng/fetch_bits.py --requests 50 --out bits/quantum_a.npy
"""
import argparse
import os
import time

import numpy as np
import requests

from common import save_words

URL = "https://api.quantumnumbers.anu.edu.au"


def fetch(n_requests, key):
    words = []
    for i in range(n_requests):
        r = requests.get(URL, params={"length": 1024, "type": "uint16"},
                         headers={"x-api-key": key}, timeout=30)
        r.raise_for_status()
        j = r.json()
        if not j.get("success", True):
            raise RuntimeError(f"ANU error on request {i}: {j}")
        words.extend(j["data"])
        print(f"request {i + 1}/{n_requests}: {len(words)} words")
        time.sleep(1.1)  # stay under 1 request/sec
    return np.array(words, dtype=np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = os.environ.get("QRNG_API_KEY")
    if not key:
        raise SystemExit("Set QRNG_API_KEY first.")
    words = fetch(a.requests, key)
    m = save_words(words, a.out, source="ANU QRNG (vacuum fluctuations)",
                   extra={"requests": a.requests, "endpoint": URL})
    print(m)


if __name__ == "__main__":
    main()
