# jspace-lab

Experiment bin. Not a project, not a pillar. Time-boxed: **Oct 2 to Oct 16, 2026**.
Things leave this folder only when they pass their go/no-go.

## What it unlocks

0. **Intent gate**: a calibrated guess at what a request wants, plus an entropy score that says "ask instead of guess". Works today, CPU only.
1. **J-lens readout logger**: a per-layer trace of what a model is "considering" at each step. Human-readable decision traces for any LLM brain.
2. **Quantum vs pseudo-random sampling test**: a pre-registered, honest test of whether quantum randomness changes model outputs. Expected result: no difference.

What it does NOT unlock: a smarter model, reception of outside signals (see `qrng/PREREG.md`), or anything production-ready.

## Layout

```
jspace-lab/
  intent_gate.py          0  intent probabilities + entropy gate
  readout/readout.py      1  J-lens top tokens per layer -> JSONL
  qrng/PREREG.md          2  pre-registration (commit BEFORE running)
  qrng/common.py             shared bit-file format + uniform conversion
  qrng/fetch_bits.py         fetch ANU quantum numbers, save to disk
  qrng/make_pseudo.py        make pseudo-random words in the same format
  qrng/run_sampling.py       sample with u from bit files, A/B alternating
  qrng/analyze.py            pre-registered tests + verdict
  requirements.txt
```

## Setup (Windows, from D:\)

```bash
cd D:\jspace-lab
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
git clone https://github.com/anthropics/jacobian-lens D:\jacobian-lens
pip install -e D:\jacobian-lens
```

Install a CUDA build of PyTorch for the 5080 from pytorch.org if `torch.cuda.is_available()` is False.

## Build order

0. Intent gate: `python intent_gate.py "what if we made an ai whose context is the internet"`
1. Readout: `python readout/readout.py --prompts readout/prompts.txt`
2. Reproduce one paper result by eye in the readout (e.g. the boot-shaped country prompt should surface "Italy"/"euro" at mid layers).
3. Quantum test: follow `qrng/PREREG.md` exactly.
4. GO/NO-GO for the concept mixer (not built yet on purpose).

## Go/no-go rules

- Readout graduates to Player Harness as an optional observer if traces are readable on 10 real prompts.
- Intent gate graduates to DUCTEI routing if it picks the right intent on 80%+ of 30 hand-labeled requests.
- Mixer only gets built if steering beats both random-vector and higher-temperature baselines (see earlier plan).
