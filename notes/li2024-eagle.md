# EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty

**Authors:** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang
**Venue:** ICML 2024
**Link:** https://arxiv.org/abs/2401.15077

## One-line summary

Rather than using a separate smaller model as the draft, EAGLE trains a lightweight autoregressive head that operates on the target model's own feature representations, achieving higher acceptance rates and approximately 3x speedup with no quality loss.

## Problem

Vanilla speculative decoding (Leviathan et al., 2023; Chen et al., 2023) relies on a draft model whose distribution diverges from the target, limiting the acceptance rate $\alpha$. The draft model has no access to the target's internal representations, so its predictions must be made from token embeddings alone — a weaker signal than the features the target model actually uses for next-token prediction.

## Method

EAGLE introduces a draft component that plugs into the target model's feature space rather than operating independently.

**Architecture:**
- The target model is frozen. After each forward pass, its penultimate-layer hidden states (features) are available.
- A lightweight autoregressive module — one transformer decoder layer with an embedding layer — is trained to predict the next feature vector given the current feature vector and the current token embedding.
- Formally: $\hat{f}_{t+1} = \text{DraftHead}(f_t, \text{embed}(x_t))$, where $f_t$ is the target's feature at step $t$.
- The predicted feature $\hat{f}_{t+1}$ is passed through the target model's LM head to obtain a draft probability distribution over the vocabulary.

**Training:**
- The draft head is trained on the target model's own feature trajectories (collected via a single forward pass over training data).
- Loss is cross-entropy on next-token predictions using the draft head's output.
- Training is fast: the draft head has roughly 1/30 the parameters of a 7B target model.

**Inference:**
- The draft head generates $K$ tokens speculatively, producing both token predictions and feature vectors.
- A tree-structured draft is constructed: multiple candidate sequences are generated simultaneously using beam-like expansion.
- The target model verifies all candidates in a single forward pass using tree attention masking.
- Accepted tokens are selected using the same acceptance criterion as Leviathan et al. (2023).

## Results

Evaluated on Vicuna-7B, Vicuna-13B, LLaMA-2-Chat 7B/13B/70B, and Mixtral-8x7B Instruct on MT-Bench:

- 2.8–3.5x wall-clock speedup depending on model size.
- Acceptance rate approximately 0.80, compared to ~0.60 for vanilla speculative decoding with a comparable-cost draft model.
- Output quality (MT-Bench score) is unchanged.

## Strengths

- Feature-level drafting is well-motivated: the draft head has access to the same information the target uses, reducing the distribution gap.
- Training is cheap and requires no additional data beyond what was used to train the target.
- The tree-structured draft efficiently explores multiple candidate continuations per step.

## Limitations

- The draft head is tied to a specific target model; it cannot be reused across models without retraining.
- Memory overhead: the target model's penultimate-layer features must be retained during draft generation, which increases activation memory.
- The tree attention mechanism adds implementation complexity relative to vanilla speculative decoding.

## Connection to my work

EAGLE represents the current state of the art in draft model design and sets the acceptance rate baseline ($\alpha \approx 0.80$) against which my experiments should compare. The key question for my work is how this $\alpha$ degrades when the draft head (or an equivalent external draft model) is quantized. EAGLE's architecture also suggests a specific experiment: quantize only the draft head, not the target, and measure the resulting $\alpha$ as a function of quantization precision.
