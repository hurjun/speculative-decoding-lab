# GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

**Authors:** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
**Venue:** ICLR 2023
**Link:** https://arxiv.org/abs/2210.17323

## One-line summary

GPTQ quantizes LLM weights to INT4 layer-by-layer using approximate second-order (Hessian) information, achieving near-lossless compression of models up to 175B parameters with a one-time calibration cost of a few GPU hours.

## Problem

Post-training quantization (PTQ) of LLMs to low bit-widths (INT4, INT3) is difficult because the loss landscape is highly non-convex and small weight perturbations can compound across layers. Methods designed for smaller models do not scale to billions of parameters. The goal is a PTQ method that is fast enough to apply to production-scale models and accurate enough to be practically useful.

## Method

GPTQ is based on the Optimal Brain Quantization (OBQ) framework, which quantizes one weight at a time and updates remaining unquantized weights to compensate for the introduced error.

**Key insight:** OBQ is quadratic in the number of weights per row (too slow for large models). GPTQ introduces three approximations that reduce this to near-linear cost while preserving accuracy:

1. **Arbitrary order quantization:** quantize all weights in a row in left-to-right order rather than the greedy OBQ order. Empirically, the greedy order provides negligible benefit for large matrices.

2. **Lazy batch updates:** rather than updating all remaining weights after each quantization step, accumulate updates and apply them in blocks. This dramatically improves GPU memory access patterns.

3. **Cholesky reformulation:** precompute the inverse Hessian using a numerically stable Cholesky decomposition, avoiding repeated matrix inversion during quantization.

**Calibration:** a small calibration set (~128 samples from the training distribution) is used to compute activation statistics for the Hessian approximation. No gradient computation or backpropagation is required.

**Bit-width:** GPTQ targets INT4 (4 bits per weight) with optional INT3 support. Activations remain in FP16.

## Results

Evaluated on OPT (125M to 175B) and BLOOM (176B) on perplexity (WikiText-2, PTB, C4):

- INT4 GPTQ achieves perplexity within ~1 point of FP16 for models ≥6.7B.
- INT3 GPTQ remains usable for models ≥30B, with larger degradation for smaller models.
- Quantization of OPT-175B takes approximately 4 GPU hours on a single A100.
- 3.24x memory reduction vs. FP16 at INT4 (with minor overhead for scale factors).

## Strengths

- Scales to the largest publicly available models at the time of publication.
- Requires only a small calibration set, not the full training data.
- Produces hardware-friendly INT4 weights directly usable with optimized CUDA kernels.

## Limitations

- Accuracy degrades noticeably for small models (≤1B parameters) at INT4, which is directly relevant to my use case since draft models are typically small.
- The method quantizes weights only; activations remain in FP16. True INT4 inference requires dedicated kernels (e.g., ExLlama, AutoGPTQ).
- Calibration data distribution affects the Hessian approximation; out-of-distribution calibration can degrade quality.

## Connection to my work

GPTQ is the quantization method I will likely use for the draft model in my experiments, as it is the most widely supported for post-training INT4 quantization of open-weight models. The critical limitation noted above — accuracy degrades for small models at INT4 — directly motivates my research question: draft models are small by design, so they may be disproportionately harmed by INT4 quantization compared to target models. Quantifying this degradation in terms of speculative decoding acceptance rate (rather than perplexity) is the primary contribution of my paper.
