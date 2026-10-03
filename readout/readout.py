"""J-lens readout logger: top-k lens tokens per layer at chosen positions -> JSONL.

Uses Anthropic's jlens reference implementation and Neuronpedia's pre-fitted
Qwen3.5-4B lens (n=1000), so no fitting is needed.

Usage:
  python readout/readout.py --prompts readout/prompts.txt --out runs/readout.jsonl
"""
import argparse
import json
import os
import time

import torch
import transformers
import jlens

MODEL_ID = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="runs/readout.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--position", type=int, default=-1)
    a = ap.parse_args()

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16).cuda()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)

    prompts = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            lens_logits, model_logits, _ = lens.apply(
                model, prompt, positions=[a.position])
            layers = {}
            for layer, logits in sorted(lens_logits.items()):
                top = logits[0].topk(a.k).indices.tolist()
                layers[int(layer)] = [tok.decode([t]) for t in top]
            out_top = [tok.decode([t]) for t in
                       model_logits[0, a.position].topk(a.k).indices.tolist()] \
                if model_logits is not None and model_logits.dim() == 3 else None
            rec = {"ts": time.time(), "model": MODEL_ID, "lens": LENS_FILE,
                   "prompt_id": i, "prompt": prompt, "position": a.position,
                   "layers": layers, "model_top": out_top}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mid = sorted(layers)[len(layers) // 2]
            print(f"[{i}] mid L{mid}: {layers[mid]}  |  {prompt[:60]}")


if __name__ == "__main__":
    main()
