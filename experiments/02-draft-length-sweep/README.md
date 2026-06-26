# Experiment 02: draft length k vs. acceptance behavior

## Question

For a fixed draft/target pairing, how do the acceptance rate α and the mean
number of confirmed tokens per round depend on the draft length k (the number of
tokens the draft proposes before each target verification), and does the
implementation stay lossless for every k?

This experiment deliberately measures only the *model-level* quantities that
greedy decoding makes deterministic (α, accepted tokens per round). These are
independent of machine load and reproduce bit-for-bit, which complements the
*wall-clock* speedup measured in
[experiment 01](../01-gpt2-speculative-speedup/). Single-stream CPU wall-clock on
this laptop is too load-dependent to report as a stable function of k, so it is
intentionally not tabulated here; see experiment 01 for timing.

## Setup

- **Hardware:** Apple M2, 8 GB unified memory, macOS 26.5 (build 25F80).
- **Software:** Python 3.13, PyTorch 2.6.0, transformers 4.49.0, float32.
- **Backend:** CPU.
- **Models:** `gpt2` (124M) target, `distilgpt2` (82M) draft; shared GPT-2 tokenizer.
- **Decoding:** greedy; fixed prompt; 96 generated tokens per setting.
- **Sweep:** k ∈ {1, 2, 3, 4, 6, 8}.
- **Entry point:** [`run.py`](run.py) (sets the HuggingFace offline flags; uses
  the local checkpoint cache and never reaches the network).

`α` here is the harness definition `accepted / proposed`, where `proposed = k`
for every round. Because the draft proposes all k tokens before the target
verifies, any draft tokens generated past the first rejection in a round are
counted in the denominator but can never be accepted.

## Findings

```
device=cpu target=gpt2 draft=distilgpt2 max_new_tokens=96 decoding=greedy
```

| k | α (accepted/proposed) | accepted/round | tokens/round | rounds | lossless |
|---|---|---|---|---|---|
| 1 | 0.902 | 0.90 | 1.88 | 51 | yes |
| 2 | 0.871 | 1.74 | 2.74 | 35 | yes |
| 3 | 0.805 | 2.41 | 3.31 | 29 | yes |
| 4 | 0.720 | 2.88 | 3.84 | 25 | yes |
| 6 | 0.642 | 3.85 | 4.80 | 20 | yes |
| 8 | 0.562 | 4.50 | 5.33 | 18 | yes |

1. **Output stays identical to greedy autoregressive decoding for every k.** The
   losslessness column is `yes` across the sweep, confirming that the KV-cache
   bookkeeping and the verification rule are correct independent of the draft
   length.
2. **k = 1 isolates the per-position acceptance probability.** With one proposal
   per round nothing can be wasted, so α = 0.902 is an unbiased estimate of the
   probability that the draft's greedy token equals the target's greedy token at
   a realized context for this pairing.
3. **α (accepted/proposed) falls as k grows** (0.902 → 0.562). Longer proposals
   reach deeper, conditionally less certain positions, and every draft token
   generated after the first rejection in a round inflates the denominator
   without being accepted. The acceptance rate is therefore not a fixed property
   of the model pair alone; it depends on how far ahead the draft is asked to
   speculate.
4. **Confirmed tokens per round rise but with diminishing returns.** Accepted
   tokens per round grow from 0.90 (k = 1) to 4.50 (k = 8), and emitted tokens
   per round (accepted plus one corrected/bonus token) from 1.88 to 5.33. The
   marginal gain per added draft token shrinks: going from k = 1 to k = 2 adds
   0.84 accepted tokens per round, whereas k = 6 to k = 8 adds 0.65 over two
   extra positions. This is the saturation predicted by the geometric
   tokens-per-round model `(1 − α^(k+1)) / (1 − α)` derived in
   [`notes/derivation-acceptance-sampling.md`](../../notes/derivation-acceptance-sampling.md).

The two trends pull in opposite directions: larger k confirms more tokens per
target forward pass (good for throughput) but lowers the fraction of draft work
that pays off (each rejected tail wastes draft compute). The throughput-optimal k
is therefore the one that best amortizes draft cost against confirmed tokens,
which depends on the relative speed of the draft and target and on the hardware —
exactly the wall-clock question studied in experiment 01.

## Reproduce

```bash
# from the repository root, with the .venv active and the gpt2/distilgpt2
# checkpoints already in the local HuggingFace cache
python experiments/02-draft-length-sweep/run.py
```

Greedy decoding makes the table deterministic; repeated runs reproduce these
numbers exactly.

## Next

- Repeat the sweep with a draft closer to the target (e.g. `gpt2` drafting for
  `gpt2-large`) to see how the per-position acceptance probability at k = 1
  shifts the whole curve upward.
- Pair this acceptance curve with per-k wall-clock timing on a memory-bandwidth-
  bound backend (server GPU) to locate the throughput-optimal k empirically.
