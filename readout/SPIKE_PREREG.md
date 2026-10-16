# Quantization spike: pre-registered criterion

Written 2026-10-16, before the 24-prompt run. Commit this file before running.

**Question:** does the J-lens readout in the late layers (L20-30) of Qwen3.5-4B survive int8
weights, compared with the bf16 CPU reference?

**Prompts:** all 24 dev-split tasks from `brain/eval/tasks.jsonl`. The heldout split is not
touched. Labels are not used (this is a fidelity test, not an accuracy test).

**Pass (int8):** late band (L20-30) mean top-1 agreement >= 0.80 AND mean overlap@5 >= 0.80,
on 24 prompts. Same bar for nf4 if it is run.

**Not decisive:** mid band (L10-19) and final-layer model top-1 are reported but do not decide
pass or fail. The gate does not read the mid band.

**Disclosure:** the 0.80 bar is the one from the earlier 12-prompt rule of thumb, not tuned to
that result. The earlier 12-prompt int8 run (late band 0.86 / 0.87) motivated choosing the late
band, so this run is a confirmation on more prompts, not a blind test.

**If it fails:** fall back to the selective-precision test (bf16 for L10-19), or stay on the
CPU bf16 path.

**Result:** TODO
