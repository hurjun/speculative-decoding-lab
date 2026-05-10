# Fast Inference from Transformers via Speculative Decoding

**Authors:** Yaniv Leviathan, Matan Kalman, Yossi Matias (Google)
**Venue:** ICML 2023
**Link:** https://arxiv.org/abs/2211.17192

## One-line summary

A small draft model generates K candidate tokens autoregressively; a single parallel forward pass of the target model then accepts or rejects each token, guaranteeing output distribution equivalence while achieving 2–3x wall-clock speedup.

## Problem

Autoregressive decoding in large language models is memory-bandwidth bound: each forward pass generates one token, and the bottleneck is loading model weights from HBM rather than compute. Larger models do not improve hardware utilization — they simply move more data per token. The goal is to produce more tokens per unit time without changing the model or degrading output quality.

## Method

The algorithm operates in rounds. In each round:

1. A small draft model $q$ generates $K$ tokens $\tilde{x}_1, \dots, \tilde{x}_K$ autoregressively from the current context.
2. The target model $p$ processes the context plus all $K$ draft tokens in a single forward pass, producing $K+1$ distributions in parallel.
3. Each draft token $\tilde{x}_i$ is accepted with probability $\min\!\left(1,\, \frac{p(\tilde{x}_i \mid x_{<i})}{q(\tilde{x}_i \mid x_{<i})}\right)$.
4. On the first rejection at position $i$, a corrected token is sampled from $p' = \text{normalize}(\max(0,\, p - q))$ at that position, and the remaining draft tokens are discarded.
5. If all $K$ tokens are accepted, one additional token is sampled from $p(\cdot \mid x_{<K+1})$.

**Key guarantee:** the marginal distribution of every output token is identical to that of the target model $p$, regardless of the acceptance rate. Speedup is lossless with respect to output quality.

**Expected tokens per step:** $1 + K\alpha$ where $\alpha$ is the per-token acceptance rate (assuming i.i.d. for analysis). Actual throughput gain depends on the ratio of draft model cost to target model cost and $\alpha$.

## Results

Evaluated on T5-XXL (11B) with T5-Small (60M) as draft on WMT En→De and CNN/DailyMail summarization:

- 2.4–3.1x wall-clock speedup with no measurable quality change (BLEU, ROUGE).
- Acceptance rate varies by task and temperature; lower temperature → higher acceptance.

## Strengths

- Theoretical guarantee of exact distribution preservation is a strong property — the method does not approximate.
- Speedup is additive: the target model's parallel attention allows processing $K$ tokens at nearly the same cost as 1 token when batch size is small.
- Simple to implement on top of any existing transformer inference stack.

## Limitations

- Speedup is highly sensitive to acceptance rate $\alpha$, which depends on how well the draft model approximates the target. With low $\alpha$, many draft tokens are wasted.
- Requires running two models simultaneously, doubling peak memory if both are kept resident.
- The analysis assumes i.i.d. acceptance rates; in practice, $\alpha$ is context-dependent and hard to predict.
- Draft model must be chosen or trained for each target model, limiting general applicability.

## Connection to my work

This paper defines the acceptance rate $\alpha$ as the central metric for speculative decoding efficiency. My experiment will measure how $\alpha$ changes as a function of draft model quantization level (FP16 → INT8 → INT4). The theoretical speedup formula $1 + K\alpha$ provides the analytical baseline against which measured speedups should be compared.
