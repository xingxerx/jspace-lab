# PREREG: quantum vs pseudo-random sampling

Fill in the date, then `git commit` this file BEFORE fetching bits or running anything.
Changing it after you have seen results voids the test.

- Registered on: 2026-10-02
- Model: Qwen/Qwen3.5-4B, bf16, temperature 1.0, max 48 new tokens
- Prompts: `qrng/prompts.txt` (4 prompts, frozen)
- Samples: 40 per prompt per condition (160 per condition)

## Hypothesis

- H0 (expected): outputs sampled with quantum uniforms are statistically indistinguishable from outputs sampled with pseudo-random uniforms.
- H1: they differ.

## Why H0 is expected (the key logic)

Given fixed weights, prompt, and temperature, each sampled token is a deterministic function of its uniform `u`. So the outputs can only differ if the **uniforms themselves** differ in distribution. Tests T0/T1 check exactly that. If both streams are uniform, any T2/T3 difference must come from the pipeline, not from the model "receiving" anything.

## Design

0. Condition A: PCG64 pseudo-random words, seed written above.
1. Condition B: ANU QRNG words (quantum vacuum fluctuations).
2. Both stored as uint16 `.npy` with sha256 manifests BEFORE the run.
3. Identical conversion (`common.words_to_uniforms`) and identical sampler for both.
4. Conditions interleaved per sample (A, B, A, B, ...).
5. A stream that runs out stops the run. No fallback RNG, ever.

## Tests and threshold

Total alpha 0.001, Bonferroni over 4 tests (0.00025 each). Run `python qrng/analyze.py runs/qrng.jsonl`.

- T0 / T1: KS test of each condition's consumed uniforms vs U(0,1). Sanity.
- T2: Mann-Whitney U on per-token mean logprob, A vs B.
- T3: chi-square on first-token choices per prompt, combined with Fisher's method.

## Decision rule

- All pass: NULL HOLDS. Write it up as a clean negative result.
- T0/T1 fail: the source is biased. Hardware/conversion issue, not a signal.
- T2/T3 fail: run the bug checklist, then replicate on a fresh batch of bits (new fetch, new seed). Only a result that survives replication AND an independent rerun counts.

## Bug checklist (run before believing any difference)

- [ ] Both bit files' sha256 match their manifests
- [ ] Same code path for both (only `--bits-a` / `--bits-b` differ)
- [ ] No seed or file reused between replication rounds
- [ ] Condition order really alternated (check `ts` in the JSONL)
- [ ] GPU nondeterminism: rerun A with the same file, outputs should match
- [ ] Swap labels (A file as B, B file as A) and confirm the difference follows the file, not the slot

## Bit budget (free ANU tier)

50 requests x 1024 words = 51,200 words = 25,600 uniforms per fetch. Worst case need: 4 x 40 x 48 = 7,680 per condition. Leaves room for one replication.
