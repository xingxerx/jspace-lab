"""Pre-registered analysis. Run ONLY after PREREG.md is committed.

Tests (alpha = 0.001 total, Bonferroni over 4 tests => 0.00025 each):
  T0  A's consumed uniforms are U(0,1)          (KS one-sample)  sanity
  T1  B's consumed uniforms are U(0,1)          (KS one-sample)  sanity
  T2  A vs B per-token logprob distributions    (Mann-Whitney U)
  T3  A vs B first-token choices, per prompt    (chi-square, Fisher-combined)

Usage:
  python qrng/analyze.py runs/qrng.jsonl
"""
import json
import sys
from collections import Counter

import numpy as np
from scipy import stats

ALPHA = 0.001
N_TESTS = 4


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def first_token_test(rows):
    pvals = []
    for pid in sorted({r["prompt_id"] for r in rows}):
        a = Counter(r["tokens"][0] for r in rows if r["prompt_id"] == pid and r["cond"] == "A" and r["tokens"])
        b = Counter(r["tokens"][0] for r in rows if r["prompt_id"] == pid and r["cond"] == "B" and r["tokens"])
        keys = sorted(set(a) | set(b), key=lambda k: -(a[k] + b[k]))
        common = [k for k in keys if a[k] + b[k] >= 5]
        if len(common) < 2:
            continue  # nothing to compare for this prompt
        table = [[a[k] for k in common] + [sum(a.values()) - sum(a[k] for k in common)],
                 [b[k] for k in common] + [sum(b.values()) - sum(b[k] for k in common)]]
        if table[0][-1] + table[1][-1] == 0:
            table = [row[:-1] for row in table]
        pvals.append(stats.chi2_contingency(table)[1])
    if not pvals:
        return float("nan"), 0
    return float(stats.combine_pvalues(pvals, method="fisher")[1]), len(pvals)


def main(path):
    rows = load(path)
    A = [r for r in rows if r["cond"] == "A"]
    B = [r for r in rows if r["cond"] == "B"]
    uA = np.concatenate([r["u"] for r in A]); uB = np.concatenate([r["u"] for r in B])
    lpA = [r["logprob"] / max(r["n_tokens"], 1) for r in A]
    lpB = [r["logprob"] / max(r["n_tokens"], 1) for r in B]

    t0 = stats.kstest(uA, "uniform").pvalue
    t1 = stats.kstest(uB, "uniform").pvalue
    t2 = stats.mannwhitneyu(lpA, lpB).pvalue
    t3, n_prompts = first_token_test(rows)
    thr = ALPHA / N_TESTS

    res = {"n_A": len(A), "n_B": len(B), "uniforms_A": int(uA.size), "uniforms_B": int(uB.size),
           "T0_ks_A": t0, "T1_ks_B": t1, "T2_logprob_mw": t2,
           "T3_first_token_fisher": t3, "T3_prompts_used": n_prompts,
           "threshold_each": thr}
    print(json.dumps(res, indent=2))

    sanity_fail = [n for n, p in (("T0", t0), ("T1", t1)) if p < thr]
    effect = [n for n, p in (("T2", t2), ("T3", t3)) if p == p and p < thr]
    if sanity_fail:
        print(f"\nVERDICT: SANITY FAIL {sanity_fail}. A random source is not uniform. "
              "Check hardware bias / conversion before reading anything else.")
    elif effect:
        print(f"\nVERDICT: DIFFERENCE on {effect}. Assume a bug first: run the checklist in PREREG.md, "
              "then replicate on fresh bits before believing it.")
    else:
        print("\nVERDICT: NULL HOLDS. No difference between quantum and pseudo-random sampling.")


if __name__ == "__main__":
    main(sys.argv[1])
