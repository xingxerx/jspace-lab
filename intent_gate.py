"""Intent gate v0: P(intent | message) + normalized entropy.

Rule: if entropy > ASK_THRESHOLD, ask a clarifying question instead of guessing.
CPU only. Edit INTENTS to match your routing targets.

Usage:
  python intent_gate.py "your message here"
  python intent_gate.py --prior 0.4,0.3,0.2,0.1 "message"
"""
import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer

INTENTS = {
    "build": "write code, make a file, implement this, set it up",
    "explain": "how does this work, teach me, why, what does this mean",
    "explore": "what if, imagine, hypothetically, could we",
    "fact": "what is, when, who, give me the number, current status",
}
ASK_THRESHOLD = 0.7
TEMP = 0.05

_model = None
_proto = None


def _load():
    global _model, _proto
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _proto = _model.encode(list(INTENTS.values()), normalize_embeddings=True)
    return _model, _proto


def p_intent(msg, prior=None):
    model, proto = _load()
    sim = proto @ model.encode(msg, normalize_embeddings=True)
    logits = sim / TEMP
    if prior is not None:
        logits = logits + np.log(np.asarray(prior, dtype=float))
    p = np.exp(logits - logits.max())
    p /= p.sum()
    entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    names = list(INTENTS)
    return {
        "probs": {n: round(float(x), 3) for n, x in zip(names, p)},
        "top": names[int(p.argmax())],
        "entropy": round(entropy, 3),
        "action": "ask" if entropy > ASK_THRESHOLD else "proceed",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("message")
    ap.add_argument("--prior", help="comma-separated, same order as INTENTS")
    a = ap.parse_args()
    prior = [float(x) for x in a.prior.split(",")] if a.prior else None
    print(json.dumps(p_intent(a.message, prior), indent=2))
