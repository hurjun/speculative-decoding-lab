# Experiment 01: speculative decoding speedup on the GPT-2 family

## Question

Does the speculative decoding implementation in [`specdec/`](../../specdec)
produce a real wall-clock speedup over autoregressive decoding on this laptop,
without changing the output, and how do the acceptance rate and the speedup
depend on the draft/target pairing and the compute backend?

## Setup

- **Hardware:** Apple M2, 8 GB unified memory, macOS 26.5 (build 25F80).
- **Software:** Python 3.13, PyTorch 2.6.0, transformers 4.49.0, float32.
- **Backends:** Apple MPS and CPU.
- **Models:** pretrained GPT-2 checkpoints from HuggingFace
  (`gpt2` 124M, `gpt2-medium` 355M, `gpt2-large` 774M, `distilgpt2` 82M); draft
  and target share the GPT-2 tokenizer.
- **Decoding:** greedy; `k = 4` proposed draft tokens per round; 96–128
  generated tokens from a fixed prompt; median wall-clock over 3–5 runs after one
  warmup.
- **Entry point:** [`run.sh`](run.sh) (wraps `benchmark/benchmark.py`).

The benchmark performs an explicit correctness check each run: greedy
speculative output must equal greedy autoregressive output token for token.

## Findings

| Target | Draft | Backend | α | accepted/round | AR tok/s | spec tok/s | Speedup | output identical |
|---|---|---|---|---|---|---|---|---|
| gpt2-large | distilgpt2 | MPS | 0.710 | 2.84 | 3.3 | 6.0 | 1.80× | yes |
| gpt2-medium | distilgpt2 | MPS | 0.642 | 2.46 | 4.6 | 7.8 | 1.71× | yes |
| gpt2-large | gpt2 | MPS | 0.841 | 3.36 | 3.2 | 3.6 | 1.13× | yes |
| gpt2 | distilgpt2 | MPS | 0.781 | 3.00 | 55.5 | 49.1 | 0.88× | yes |
| gpt2-medium | distilgpt2 | CPU | 0.612 | 2.31 | 12.8 | 9.0 | 0.70× | yes |

1. On MPS the method yields up to a 1.80× speedup with output identical to
   greedy autoregressive decoding.
2. The acceptance rate α is highest when draft and target are closest
   (gpt2 → gpt2-large, α = 0.841).
3. Speedup is not monotone in α. gpt2-large / gpt2 has the highest α yet a lower
   speedup than gpt2-large / distilgpt2, because the cheaper draft (distilgpt2,
   82M) costs less per proposed token. The relevant ratio is acceptance per unit
   of draft cost.
4. On CPU the same configuration is slower than autoregressive (0.70×).
   Single-stream CPU decoding is compute-bound, so verifying `k` tokens in one
   pass does not come for free and the extra draft work is not amortized. The
   speedup of speculative decoding is a property of the memory-bandwidth-bound
   regime, which MPS approximates here and CPU does not.

These are small models on a memory-constrained laptop; absolute throughput is low
and these results should not be extrapolated to server GPUs.

## Reproduce

```bash
# from the repository root, with the .venv active and deps installed
bash experiments/01-gpt2-speculative-speedup/run.sh
```

Each invocation downloads the referenced checkpoints on first use and prints the
acceptance rate, accepted-tokens-per-round, the correctness check, and the
speedup table.

## Next

- Sweep `k` to locate the speedup-maximizing proposal length for each pairing.
- Quantize the draft model (FP16 → INT8 → INT4) and measure how α and the
  speedup respond — the central hypothesis recorded in the reading notes.
