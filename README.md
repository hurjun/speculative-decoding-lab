# speculative-decoding-lab

A from-scratch implementation of **speculative decoding** for autoregressive
language model (LM) inference, with distribution-preserving acceptance sampling,
unit-tested correctness, and measured speedups on a laptop. The repository also
keeps the paper reading notes that motivated the implementation (see
[`notes/`](notes/)).

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) accelerates
decoding by letting a small **draft** model propose several tokens that the
larger **target** model verifies in a single forward pass. A modified rejection
sampling step makes the emitted tokens follow the target model's distribution
*exactly*, so the speedup is lossless with respect to output quality.

## Results

All numbers below were measured on this hardware; nothing is copied from the
papers. Decoding is greedy and, for every configuration, the speculative output
was verified to be **token-for-token identical** to plain autoregressive
decoding of the target model.

**Hardware:** Apple M2 (8 GB unified memory), macOS 26.5; PyTorch 2.6.0, float32.
`tok/s` is throughput (generated tokens / wall-clock seconds); `speedup` is the
ratio of autoregressive to speculative median wall-clock time over the runs.

| Target (params) | Draft (params) | Backend | α (accept rate) | accepted/round | AR tok/s | spec tok/s | Speedup |
|---|---|---|---|---|---|---|---|
| gpt2-large (774M) | distilgpt2 (82M) | MPS | 0.710 | 2.84 | 3.3 | 6.0 | **1.80×** |
| gpt2-medium (355M) | distilgpt2 (82M) | MPS | 0.642 | 2.46 | 4.6 | 7.8 | **1.71×** |
| gpt2-large (774M) | gpt2 (124M) | MPS | 0.841 | 3.36 | 3.2 | 3.6 | 1.13× |
| gpt2 (124M) | distilgpt2 (82M) | MPS | 0.781 | 3.00 | 55.5 | 49.1 | 0.88× |
| gpt2-medium (355M) | distilgpt2 (82M) | CPU | 0.612 | 2.31 | 12.8 | 9.0 | 0.70× |

`k = 4` draft tokens per round; 96–128 generated tokens; median of 3–5 runs after
one warmup. The exact commands are in
[`experiments/01-gpt2-speculative-speedup/`](experiments/01-gpt2-speculative-speedup/).

### What the measurements show

- **The speedup is real and lossless** in the memory-bandwidth-bound regime that
  speculative decoding targets: up to 1.80× on the MPS backend, with output
  identical to greedy autoregressive decoding.
- **Acceptance rate α tracks draft/target similarity.** A draft from the same
  model family as the target (gpt2 → gpt2-large) yields the highest α (0.841).
- **High α does not by itself maximize speedup; draft cost matters.** The
  gpt2-large / gpt2 pair has the highest α (0.841) but a *lower* speedup (1.13×)
  than gpt2-large / distilgpt2 (α = 0.710, 1.80×), because gpt2 (124M) is a more
  expensive draft than distilgpt2 (82M). The useful quantity is acceptance per
  unit of draft cost, not acceptance alone.
- **On CPU the method is slower (0.70×).** At batch size 1 on CPU, decoding is
  compute-bound rather than memory-bandwidth-bound, so verifying `k` tokens in
  one pass costs roughly `k` single-token forwards and the extra draft work is
  not amortized. This is consistent with the premise of Leviathan et al. (2023):
  the gain comes from underutilized memory bandwidth, which the MPS runs
  approximate and the CPU runs do not.

These are small models on a laptop; the absolute throughput is low and the
ranking of backends here should not be extrapolated to server GPUs. The point of
the table is the *mechanism* (α, accepted tokens per round, the cost/acceptance
trade-off, and the regime dependence), all measured directly.

### Analysis: how acceptance moves with draft length and temperature

The table above fixes `k = 4` and greedy decoding. To see how the two model-level
quantities — the acceptance rate α and the confirmed tokens per round — respond to
the draft length `k` and to the sampling temperature, the figure below sweeps both
on the fully offline toy backend (the same randomly-initialized GPT-2 pair the
tests use, so the study reproduces from a clean checkout with no download). The
greedy half is bit-for-bit deterministic; the sampling half is seeded and averaged
over 8 generator seeds. Full tables, the script, and the raw CSVs are in
[`experiments/03-toy-acceptance-sweep/`](experiments/03-toy-acceptance-sweep/).

![acceptance vs draft length and temperature](experiments/03-toy-acceptance-sweep/results/acceptance_sweep.png)

- **(a) Confirmed tokens per round follow the geometric model.** Measured greedy
  tokens per round track `(1 − α^(k+1)) / (1 − α)` (with α the `k = 1` acceptance)
  to within 0.35 token across `k = 1..8`, an empirical check of the derivation in
  [`notes/derivation-acceptance-sampling.md`](notes/derivation-acceptance-sampling.md).
- **(b) On homogeneous random weights, per-position acceptance does not decay with
  depth.** The toy `accepted / proposed` stays near 0.95 and sits at or above the
  constant-α envelope, whereas the *same* measurement on real GPT-2 falls from
  0.902 to 0.562 as `k` grows
  ([experiment 02](experiments/02-draft-length-sweep/)). The gap is informative:
  the constant-α model is near-exact for the structureless toy backend but only an
  upper bound for real text, where positions deeper in a round are conditionally
  harder.
- **(c) Acceptance is U-shaped in temperature, between two analytic limits.**
  Under sampling, α bottoms out near `T = 0.2` (0.27) and rises on both sides —
  toward the greedy argmax-agreement rate (≈ 0.97) as `T → 0`, and toward 1 as
  `T → ∞` — exactly as `α = 1 − TV(p, q)` predicts: low temperature concentrates
  `p` and `q` on their argmaxes (which usually agree), high temperature flattens
  both toward uniform (which always agrees), and the most dissimilar distributions
  sit in between. Output stays exactly target-distributed at every temperature.

## The algorithm

Each round generates one or more confirmed tokens:

```
context  ──►  draft model q ──►  propose  x1 x2 x3 x4         (k autoregressive steps)
context x1 x2 x3 x4  ──►  target model p ──►  p1 p2 p3 p4 p5   (ONE parallel forward)

verify left to right:
  accept x_j  with prob  min(1, p_j(x_j) / q_j(x_j))
  on first rejection at i:  emit  t ~ normalize(max(0, p_i - q_i))   and stop
  if all k accepted:        emit bonus token  t ~ p_{k+1}
```

where `p_j = p(· | context, x_1..x_{j-1})` and `q_j` is the draft distribution
that `x_j` was sampled from.

**Correctness (distribution preservation).** For a single proposed token `x ~ q`,
accepted with probability `min(1, p(x)/q(x))` and otherwise replaced by a draw
from the residual `norm(max(0, p − q))`, the emitted token is distributed
*exactly* as `p`, independent of `q`. Applying this position by position makes
the whole speculative sequence identically distributed to target-only sampling.
This repository verifies the claim two ways: a Monte Carlo test on the
model-independent primitive (`tests/test_sampling.py`) and a bit-for-bit greedy
equivalence test through the full cached generation loop
(`tests/test_generate_toy.py`).

**Expected tokens per round.** Under the paper's i.i.d. acceptance assumption
with rate α, a round emits

```
E[tokens / round] = (1 − α^(k+1)) / (1 − α)
```

confirmed tokens (the accepted prefix plus the one corrected/bonus token), which
recovers the `1 + kα` accounting used in the reading notes for small α.

A self-contained derivation of all three results above — distribution
preservation, the acceptance probability `α = Σ min(p, q) = 1 − TV(p, q)`, and the
expected tokens per round, including the exact relationship between the geometric
form and the `1 + kα` estimate — is in
[`notes/derivation-acceptance-sampling.md`](notes/derivation-acceptance-sampling.md);
each claim there names the test that verifies it numerically. The way α and
accepted-tokens-per-round vary with the draft length `k` is measured on real
GPT-2 in
[`experiments/02-draft-length-sweep/`](experiments/02-draft-length-sweep/).

## Repository layout

```
specdec/            core library
  sampling.py         acceptance/rejection primitives (model-independent)
  generate.py         KV-cache-managed autoregressive and speculative loops
  models.py           toy GPT-2 builder (offline) and gated HF loader
tests/              unit tests (run offline, no model download)
benchmark/          benchmark CLI
experiments/        experiment logs with exact reproduction commands
notes/              paper reading notes (speculative decoding, quantization)
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # makes `import specdec` available

# Unit tests (offline; uses tiny randomly-initialized models):
pip install -r requirements-dev.txt
pytest -q

# Offline structural/correctness benchmark on toy models (no download):
python benchmark/benchmark.py --backend toy

# Real measured speedup (downloads gpt2 / distilgpt2 on first run):
python benchmark/benchmark.py --backend hf --target gpt2 --draft distilgpt2 \
    --device auto --max-new-tokens 128 --k 4 --runs 5
```

Model downloads happen only with `--backend hf`; the default toy backend and the
entire test suite are fully offline.

## Reading notes and how they connect

The implementation grew out of the notes in [`notes/`](notes/):

- [Leviathan et al., 2023](notes/leviathan2023-speculative-decoding.md) — the
  original speculative decoding algorithm and the `1 + kα` speedup model.
- [Chen et al., 2023](notes/chen2023-speculative-sampling.md) — concurrent
  speculative sampling and the modified rejection scheme implemented here.
- [EAGLE](notes/li2024-eagle.md) / [EAGLE-2](notes/li2024-eagle2.md) —
  feature-level drafting and dynamic draft trees (directions for future work).
- [GPTQ](notes/frantar2023-gptq.md) / [AWQ](notes/lin2024-awq.md) — post-training
  quantization, relevant to the open question below.

**Open question / roadmap.** The notes converge on one testable hypothesis: how
does the acceptance rate α (and therefore end-to-end speedup) change as the
*draft* model is quantized from FP16 to INT8 to INT4? The harness here already
isolates α and the cost/acceptance trade-off, which is the measurement substrate
that experiment would need.

## References

- Y. Leviathan, M. Kalman, Y. Matias. *Fast Inference from Transformers via
  Speculative Decoding.* ICML 2023. arXiv:2211.17192.
- C. Chen, S. Borgeaud, G. Irving, et al. *Accelerating Large Language Model
  Decoding with Speculative Sampling.* 2023. arXiv:2302.01318.
