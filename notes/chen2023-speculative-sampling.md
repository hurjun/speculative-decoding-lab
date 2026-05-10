# Accelerating Large Language Model Decoding with Speculative Sampling

**Authors:** Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, John Jumper (DeepMind)
**Venue:** arXiv preprint, 2023
**Link:** https://arxiv.org/abs/2302.01318

## One-line summary

An independently derived concurrent proposal for speculative decoding, framed as speculative sampling, with a cleaner theoretical treatment of the acceptance–rejection correction step.

## Problem

Same as Leviathan et al. (2023): autoregressive decoding is memory-bandwidth bound, and each serial token generation step underutilizes available compute on modern accelerators.

## Method

The algorithm is essentially identical to Leviathan et al. (2023), derived independently. The paper frames it as speculative sampling:

1. Draft model $q$ generates $\gamma$ tokens speculatively.
2. Target model $p$ evaluates all positions in one pass.
3. Token $\tilde{x}_i$ is accepted with probability $\min(1, p / q)$; on rejection, sample from the renormalized residual $p' \propto \max(0, p - q)$.

**Theoretical contribution:** the paper provides a cleaner proof that the correction distribution $p' = \text{normalize}(\max(0, p - q))$ ensures the output distribution exactly matches $p$. It also analyzes the expected number of tokens generated per step as a function of the draft length $\gamma$ and the acceptance rate.

The optimal $\gamma$ (number of draft tokens) is shown to depend on the ratio of draft to target model inference cost and on $\alpha$: a larger $\gamma$ is beneficial when the draft model is cheap and $\alpha$ is high.

## Results

Evaluated using Chinchilla 70B as target and Chinchilla 7B as draft:

- Approximately 2x wall-clock speedup on XSum summarization.
- Exact output distribution preservation verified empirically.

## Strengths

- The derivation of the correction distribution is rigorous and clearly explained — this is the canonical reference for why speculative decoding is lossless.
- The analysis of optimal $\gamma$ is useful for practical system design.

## Limitations

- Like Leviathan et al., performance is bounded by the draft–target distribution gap.
- Evaluated only on a single model family (Chinchilla), limiting generalizability of the reported speedup numbers.

## Connection to my work

This paper and Leviathan et al. (2023) are the two canonical references for the theoretical foundation of speculative decoding. The acceptance criterion and the correction step defined here appear in all subsequent work, including EAGLE. The analysis of optimal draft length $\gamma$ is directly relevant when choosing how many tokens to draft per step in my quantization experiments.
