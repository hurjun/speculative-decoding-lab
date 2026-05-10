# EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees

**Authors:** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang
**Venue:** arXiv preprint, 2024
**Link:** https://arxiv.org/abs/2406.16858

## One-line summary

EAGLE-2 extends EAGLE by replacing the fixed draft tree with a context-adaptive tree that dynamically allocates draft budget to high-confidence branches, yielding an additional ~30% speedup over EAGLE-1.

## Problem

In EAGLE-1, the draft tree topology is fixed: each step generates the same number of candidate tokens regardless of how confident the draft model is about the current context. This wastes verification budget on low-probability branches that will almost certainly be rejected, while underexploring high-probability regions where acceptance is likely.

## Method

**Dynamic tree construction:**
- After the draft head generates each candidate token, it also produces a confidence estimate: the probability mass assigned to the selected token under the draft distribution.
- A threshold is used to decide whether to expand a given node: nodes with high confidence are expanded (more children), while low-confidence nodes are pruned.
- The total number of nodes in the tree (draft budget $K$) is kept roughly constant across steps, but the topology adapts to the current context.

**Confidence estimation:**
- The draft model's softmax probability for the chosen token serves as the confidence signal — no additional module is required.
- A simple heuristic schedules expansion depth based on cumulative probability along each path.

**Verification:**
- Unchanged from EAGLE-1: tree attention masking allows the target model to verify all candidates in a single forward pass.

## Results

Evaluated on the same model suite as EAGLE-1 (Vicuna, LLaMA-2-Chat, Mixtral):

- Approximately 3.5–4.5x wall-clock speedup over standard autoregressive decoding.
- Roughly 20–30% throughput improvement over EAGLE-1 at the same draft budget.
- Acceptance rate improves because rejected branches are pruned early, and the same budget is redirected toward more promising paths.

## Strengths

- The adaptation mechanism requires no additional learned components — confidence is read directly from the existing draft distribution.
- Compatible with any model that uses the EAGLE-1 draft head; no retraining is needed.
- The dynamic tree is a conceptually clean extension: it makes the implicit assumption of EAGLE-1 (all branches are equally worth exploring) explicit and removes it.

## Limitations

- The confidence threshold is a hyperparameter that may require tuning per task or temperature.
- The theoretical analysis of the speedup gain from dynamic trees remains informal.
- Evaluation is limited to chat and instruction-following tasks; behavior on reasoning-intensive tasks (code, math) is less characterized.

## Connection to my work

EAGLE-2's dynamic budget allocation is relevant as a baseline and as a potential interaction effect: if the draft head is quantized, its confidence estimates become less reliable, which could degrade the tree pruning decisions beyond what the acceptance rate alone captures. This suggests a secondary metric to track in my experiments: not just $\alpha$ but also the precision of the draft model's confidence calibration under quantization.
