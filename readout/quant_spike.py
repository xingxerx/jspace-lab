"""Quantization spike: does the Qwen3.5-4B J-lens readout survive 8-bit / 4-bit?

The lens was fit on bf16 weights. This script reads it at the last prompt token (where a
gate would decide, before the model writes anything) for decision-style prompts, once in
bf16 on CPU (the reference) and once quantized on the GPU, then compares the top-k lens
tokens layer by layer.

Why two stages: bf16 weights (about 8-9 GB) do not fit in 8 GB of VRAM, so the reference
runs on CPU (slow, run once) and the quantized model runs on the GPU.

  python readout/quant_spike.py ref                    # stage 1, CPU, saves the reference
  python readout/quant_spike.py quant --quant int8     # stage 2, GPU, prints the report
  python readout/quant_spike.py quant --quant nf4      # optional, 4-bit
  python readout/quant_spike.py compare --quant int8   # re-print a report, no model load

Before running, free memory: `lms unload --all` and close anything heavy.
Prompts come from the brain eval set (dev split, round-robin over act/answer/ask/abort).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

MODEL_ID = os.environ.get("QWEN_PATH", "Qwen/Qwen3.5-4B")  # set QWEN_PATH to load from a local folder
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
JLENS_PATH = os.environ.get("JLENS_PATH")  # set to a local .pt to skip the Hub

# Pre-registered pass criterion (see SPIKE_PREREG.md). Fixed before the 24-prompt run.
# The gate reads the LATE band only, so only the late band decides pass/fail.
PREREG_MIN_TOP1 = 0.80
PREREG_MIN_OVL5 = 0.80
PREREG_MIN_PROMPTS = 24


def load_lens(jlens):
    """Load the lens from JLENS_PATH if set, else from the Hub."""
    if not JLENS_PATH:
        return jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
    for name in ("load", "from_file", "from_path"):
        fn = getattr(jlens.JacobianLens, name, None)
        if callable(fn):
            return fn(JLENS_PATH)
    # Fallback: make the Hub download resolve to the local file.
    import huggingface_hub

    def local(*a, **k):
        return JLENS_PATH

    huggingface_hub.hf_hub_download = local
    for m in list(sys.modules.values()):
        if m and getattr(m, "__name__", "").startswith("jlens") and hasattr(m, "hf_hub_download"):
            m.hf_hub_download = local
    return jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "quant_spike"
DEFAULT_TASKS = ROOT.parent / "brain" / "eval" / "tasks.jsonl"
TOPK = 20

# Stand-in for brain's real system prompt (loop/src/agent.rs). The spike measures
# quantization drift, so any decision-style prompt works.
SYSTEM = (
    "You are an agent that works one step at a time. Reply with one JSON object with "
    "fields thought, decision, tool, input, answer, question. decision is one of: act "
    "(call a tool), answer (give the final answer), ask (ask the user a question), abort "
    "(impossible or unsafe). Tools: calculator, read_file, list_dir."
)


def load_prompts(path, n):
    """Dev-split tasks, interleaved by expected first move so every class is covered."""
    groups = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        t = json.loads(line)
        if t.get("split") == "dev":
            groups.setdefault(t["expected_first"], []).append(t)
    total = sum(len(g) for g in groups.values())
    order, i = [], 0
    while len(order) < total:
        for g in groups.values():
            if i < len(g):
                order.append(g[i])
        i += 1
    return order[:n]


def build_text(tok, task):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task["prompt"]},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def read_all(hf, tok, tasks, torch):
    import jlens

    model = jlens.from_hf(hf, tok)
    lens = load_lens(jlens)
    recs = []
    for i, t in enumerate(tasks):
        t0 = time.time()
        lens_logits, model_logits, _ = lens.apply(model, build_text(tok, t), positions=[-1])
        layers = {}
        for layer, logits in sorted(lens_logits.items()):
            layers[int(layer)] = logits[0].float().topk(TOPK).indices.tolist()
        model_top = None
        if model_logits is not None:
            ml = model_logits[0] if model_logits.dim() == 3 else model_logits
            row = ml[-1] if ml.dim() == 2 else ml
            model_top = row.float().topk(TOPK).indices.tolist()
        recs.append(
            {"id": t["id"], "expected_first": t["expected_first"], "layers": layers, "model_top": model_top}
        )
        print(f"[{i + 1}/{len(tasks)}] {t['id']}  {time.time() - t0:.1f}s", flush=True)
    return recs


def save(path, mode, recs, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": MODEL_ID, "lens": LENS_FILE, "mode": mode, "topk": TOPK, "recs": recs}
    payload.update(extra or {})
    path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"saved {path}")


def overlap(a, b, k):
    return len(set(a[:k]) & set(b[:k])) / k


def compare(ref, got):
    """Per-layer agreement between reference and quantized top-k lens tokens."""
    rmap = {r["id"]: r for r in ref["recs"]}
    shared = [g for g in got["recs"] if g["id"] in rmap]
    if not shared:
        sys.exit("no shared prompts between reference and quantized run (different -n?)")
    layers = sorted(int(l) for l in shared[0]["layers"])
    per = {}
    for l in layers:
        top1, o5, o20 = [], [], []
        for g in shared:
            r = rmap[g["id"]]
            a, b = r["layers"][str(l)], g["layers"][str(l)]
            top1.append(1.0 if a[0] == b[0] else 0.0)
            o5.append(overlap(a, b, 5))
            o20.append(overlap(a, b, 20))
        n = len(shared)
        per[l] = (sum(top1) / n, sum(o5) / n, sum(o20) / n)
    final = [
        1.0 if rmap[g["id"]]["model_top"][0] == g["model_top"][0] else 0.0
        for g in shared
        if g.get("model_top") and rmap[g["id"]].get("model_top")
    ]
    return layers, per, (sum(final) / len(final) if final else None), len(shared)


def report(ref, got, label):
    layers, per, final, n = compare(ref, got)
    third = max(1, len(layers) // 3)
    bands = {
        "early": layers[:third],
        "mid": layers[third : 2 * third],
        "late": layers[2 * third :],
    }

    def avg(ls, i):
        return sum(per[l][i] for l in ls) / len(ls)

    print(f"\n=== {label} vs bf16 reference: {n} prompts, {len(layers)} layers ===")
    print(f"{'band':<7}{'layers':<10}{'top1':>7}{'ovl@5':>8}{'ovl@20':>8}")
    for name, ls in bands.items():
        print(f"{name:<7}{ls[0]}-{ls[-1]:<7}{avg(ls, 0):>7.2f}{avg(ls, 1):>8.2f}{avg(ls, 2):>8.2f}")
    if final is not None:
        print(f"final-layer model top-1 agreement: {final:.2f}")
    worst = min(layers, key=lambda l: per[l][1])
    print(f"worst layer by ovl@5: L{worst} ({per[worst][1]:.2f})")

    late = bands["late"]
    late_top1, late_o5 = avg(late, 0), avg(late, 1)
    worst_late = min(late, key=lambda l: per[l][1])
    print(f"late band (the gate's band): top1 {late_top1:.2f}, ovl@5 {late_o5:.2f}, worst L{worst_late} ({per[worst_late][1]:.2f})")
    print("mid band is informational only (the gate does not read it).")
    enough = n >= PREREG_MIN_PROMPTS
    ok = late_top1 >= PREREG_MIN_TOP1 and late_o5 >= PREREG_MIN_OVL5 and enough
    print(
        f"\nPREREG verdict ({label}): late top1 >= {PREREG_MIN_TOP1} and ovl@5 >= {PREREG_MIN_OVL5} "
        f"on >= {PREREG_MIN_PROMPTS} prompts -> "
        + ("PASS" if ok else ("INSUFFICIENT PROMPTS" if not enough else "FAIL"))
    )


def stage_ref(a):
    import torch
    import transformers

    tasks = load_prompts(a.tasks, a.n)
    print(f"{len(tasks)} prompts; loading {MODEL_ID} in bf16 on CPU (slow)...")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    t0 = time.time()
    recs = read_all(hf, tok, tasks, torch)
    save(OUT_DIR / "ref.json", "bf16-cpu", recs, {"seconds": round(time.time() - t0, 1)})


def stage_quant(a):
    import torch
    import transformers

    ref_path = OUT_DIR / "ref.json"
    if not ref_path.exists():
        sys.exit(f"missing {ref_path}; run the 'ref' stage first")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available in this environment; run env_check.py")

    tasks = load_prompts(a.tasks, a.n)
    cfg_cls = transformers.BitsAndBytesConfig
    if a.quant == "int8":
        cfg = cfg_cls(load_in_8bit=True)
    else:
        cfg = cfg_cls(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
    print(f"{len(tasks)} prompts; loading {MODEL_ID} as {a.quant} on GPU...")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, quantization_config=cfg, device_map={"": 0}
    )
    print(f"weights on GPU: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    t0 = time.time()
    recs = read_all(hf, tok, tasks, torch)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"peak VRAM during readout: {peak:.2f} GB")
    save(OUT_DIR / f"{a.quant}.json", a.quant, recs, {"seconds": round(time.time() - t0, 1), "peak_vram_gb": round(peak, 2)})
    report(json.loads(ref_path.read_text()), json.loads((OUT_DIR / f"{a.quant}.json").read_text()), a.quant)


def stage_compare(a):
    ref = json.loads((OUT_DIR / "ref.json").read_text())
    got = json.loads((OUT_DIR / f"{a.quant}.json").read_text())
    report(ref, got, a.quant)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["ref", "quant", "compare"])
    ap.add_argument("--quant", choices=["int8", "nf4"], default="int8")
    ap.add_argument("--tasks", default=str(DEFAULT_TASKS))
    ap.add_argument("-n", type=int, default=24, help="number of prompts (same value for ref and quant)")
    a = ap.parse_args()
    {"ref": stage_ref, "quant": stage_quant, "compare": stage_compare}[a.stage](a)


if __name__ == "__main__":
    main()
