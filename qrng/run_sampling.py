"""Sample from a model where each token's random number comes from a bit file.

Conditions A and B run through the IDENTICAL code path and are interleaved
(A sample, B sample, A sample, ...) so timing, caching, and thermal drift hit
both equally. If a stream runs out, the run stops; it never falls back.

Usage (real model, GPU):
  python qrng/run_sampling.py --bits-a bits/pseudo_a.npy --bits-b bits/quantum_a.npy \
      --prompts qrng/prompts.txt --samples 40 --max-new 48 --out runs/qrng.jsonl

Usage (no GPU, toy model, for testing the pipeline):
  python qrng/run_sampling.py --dry-run --bits-a ... --bits-b ... --prompts ... --out ...
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

from common import UniformStream


def pick(probs, u):
    """Inverse-CDF sampling: one uniform picks one token."""
    k = int(np.searchsorted(np.cumsum(probs), u, side="right"))
    return min(k, len(probs) - 1)


class ToyModel:
    """Deterministic fake LM for pipeline tests: vocab 50, logits seeded by context."""
    vocab, eos = 50, 0

    def start(self, prompt):
        return [int(b) % self.vocab for b in prompt.encode()][-8:]

    def probs(self, ctx, temp):
        seed = int(hashlib.sha256(bytes(ctx[-4:])).hexdigest()[:8], 16)
        logits = np.random.Generator(np.random.PCG64(seed)).normal(size=self.vocab) * 2
        z = np.exp((logits - logits.max()) / temp)
        return z / z.sum()

    def decode(self, toks):
        return " ".join(f"t{t}" for t in toks)


def run_toy(model, prompt, stream, temp, max_new):
    ctx, toks, us, lp = model.start(prompt), [], [], 0.0
    for _ in range(max_new):
        p = model.probs(ctx, temp)
        u = stream.next()
        k = pick(p, u)
        toks.append(k); us.append(u); lp += float(np.log(p[k] + 1e-300)); ctx.append(k)
        if k == model.eos:
            break
    return toks, model.decode(toks), us, lp


def run_hf(hf, tok, prompt, stream, temp, max_new):
    import torch
    dev = hf.device
    cur = tok(prompt, return_tensors="pt").input_ids.to(dev)
    past, toks, us, lp = None, [], [], 0.0
    for _ in range(max_new):
        with torch.no_grad():
            o = hf(input_ids=cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        logits = o.logits[0, -1].double() / temp
        p = torch.softmax(logits, -1).cpu().numpy()
        u = stream.next()
        k = pick(p, u)
        toks.append(k); us.append(u); lp += float(np.log(p[k] + 1e-300))
        if k == tok.eos_token_id:
            break
        cur = torch.tensor([[k]], device=dev)
    return toks, tok.decode(toks), us, lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits-a", required=True)
    ap.add_argument("--bits-b", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--samples", type=int, default=40, help="per prompt per condition")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    streams = {"A": UniformStream(a.bits_a), "B": UniformStream(a.bits_b)}
    prompts = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()]
    need = len(prompts) * a.samples * a.max_new
    for c, s in streams.items():
        if s.remaining() < need:
            raise SystemExit(f"condition {c}: {s.remaining()} uniforms, worst case needs {need}")

    if a.dry_run:
        model = ToyModel()
        gen = lambda pr, st: run_toy(model, pr, st, a.temp, a.max_new)
        model_id = "toy"
    else:
        import torch, transformers
        torch.manual_seed(0)
        tok = transformers.AutoTokenizer.from_pretrained(a.model)
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            a.model, dtype=torch.bfloat16).cuda().eval()
        gen = lambda pr, st: run_hf(hf, tok, pr, st, a.temp, a.max_new)
        model_id = a.model

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for pi, prompt in enumerate(prompts):
            for si in range(a.samples):
                for cond in ("A", "B"):  # interleaved on purpose
                    st = streams[cond]
                    start = st.i
                    toks, text, us, lp = gen(prompt, st)
                    f.write(json.dumps({
                        "ts": time.time(), "model": model_id, "temp": a.temp,
                        "cond": cond, "bits": a.bits_a if cond == "A" else a.bits_b,
                        "prompt_id": pi, "sample": si, "u_start": start,
                        "tokens": toks, "text": text, "u": us,
                        "logprob": lp, "n_tokens": len(toks)}, ensure_ascii=False) + "\n")
            print(f"prompt {pi + 1}/{len(prompts)} done; remaining A={streams['A'].remaining()} B={streams['B'].remaining()}")


if __name__ == "__main__":
    main()
