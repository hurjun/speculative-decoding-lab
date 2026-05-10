# AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

**Authors:** Ji Lin, Jiaming Tang, Haotian Liu, Shang Yang, Wei Ao, Ligeng Zhu, Weichen Xu, Yukang Chen, Zhekai Zhang, Guohao Dai, Hongxu Yin, Yao Lu, Song Han
**Venue:** MLSys 2024
**Link:** https://arxiv.org/abs/2306.00978

## One-line summary

AWQ identifies a small fraction of salient weights (those corresponding to large-magnitude activations) and protects them from quantization error by applying per-channel activation-aware scaling, outperforming GPTQ particularly on instruction-following and reasoning tasks.

## Problem

GPTQ (Frantar et al., 2023) treats all weights as equally important and minimizes a layer-wise reconstruction error. However, not all weights contribute equally to the model's output: some channels are activated much more frequently and with larger magnitudes, making their corresponding weights disproportionately influential. Quantizing these weights uniformly introduces asymmetric error that degrades performance on tasks sensitive to precise token distributions.

## Method

**Key observation:** approximately 1% of weight channels are "salient" — they are consistently activated by large activation magnitudes across diverse inputs. Quantization error in these channels has an outsized effect on output quality.

**Naive fix:** keep salient weights in FP16 (mixed-precision quantization). This preserves accuracy but is hardware-unfriendly because mixed-precision matrix multiplications do not map efficiently to tensor cores.

**AWQ's solution:** instead of keeping salient weights in FP16, scale them up by a per-channel factor $s > 1$ before quantization, and scale the corresponding activations down by $1/s$ to preserve the mathematical equivalence of the computation. After scaling, the quantization grid covers the now-larger weight values with smaller relative error — equivalent to effectively devoting more quantization levels to the salient channel.

Formally, for a linear layer $y = Wx$: $y = (W \cdot \text{diag}(s)) \cdot (\text{diag}(s)^{-1} x)$. Quantize $W \cdot \text{diag}(s)$ at INT4; run inference with unscaled weights and pre-scaled activations.

**Calibration:** the scaling factors $s$ are determined by minimizing a layer-wise mean squared error on a small calibration set. No gradient computation is required.

## Results

Evaluated on LLaMA-1/2 (7B to 70B), OPT, Falcon, and instruction-tuned variants on perplexity (WikiText-2) and zero-shot reasoning benchmarks (WinoGrande, HellaSwag, ARC):

- INT4 AWQ matches or exceeds INT4 GPTQ on perplexity for most model sizes.
- On instruction-following tasks (MT-Bench, HumanEval), AWQ consistently outperforms GPTQ, suggesting better preservation of the fine-tuned distribution.
- Compatible with efficient inference kernels (AWQ CUDA kernels integrated into the `autoawq` library).

## Strengths

- The scaling approach is hardware-friendly: the entire computation remains in INT4 (no mixed-precision branching).
- Intuition is clean and well-supported: protecting the most influential weights is directly motivated by analyzing what matters for output quality.
- Generalizes well across model families without re-tuning the method.

## Limitations

- The 1% salience threshold is a heuristic; the optimal threshold likely varies by model and task.
- Like GPTQ, AWQ degrades more for smaller models, which is relevant for draft model quantization.
- Evaluation focuses on perplexity and accuracy; calibration of output distributions (relevant to speculative decoding acceptance rates) is not studied.

## Connection to my work

AWQ is an important alternative to GPTQ for my experiments. Because instruction-tuned models show a larger gap between GPTQ and AWQ, and because draft models in speculative decoding are often instruction-tuned alongside the target, I should compare both quantization methods. The observation that AWQ better preserves the fine-tuned distribution suggests it may also better preserve acceptance rates — testing this hypothesis is a secondary contribution of my paper. Practically, I will use `autoawq` for AWQ quantization and `auto-gptq` for GPTQ quantization in my experiment pipeline.
