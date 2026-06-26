# Notes

Paper reading notes and literature reviews relevant to speculative decoding and LLM inference efficiency.

## Speculative Decoding — Foundations

- [leviathan2023-speculative-decoding.md](leviathan2023-speculative-decoding.md) — original speculative decoding algorithm (ICML 2023)
- [chen2023-speculative-sampling.md](chen2023-speculative-sampling.md) — concurrent speculative sampling from DeepMind

## Speculative Decoding — State of the Art

- [li2024-eagle.md](li2024-eagle.md) — EAGLE: feature-level draft model (ICML 2024)
- [li2024-eagle2.md](li2024-eagle2.md) — EAGLE-2: dynamic draft tree pruning

## Quantization

- [frantar2023-gptq.md](frantar2023-gptq.md) — GPTQ: layer-wise INT4 post-training quantization (ICLR 2023)
- [lin2024-awq.md](lin2024-awq.md) — AWQ: activation-aware weight quantization (MLSys 2024)

## Derivations

- [derivation-acceptance-sampling.md](derivation-acceptance-sampling.md) — proofs
  of distribution preservation, the acceptance probability
  `α = Σ min(p, q) = 1 − TV(p, q)`, and the expected tokens per round, each cross-
  referenced to the unit test that verifies it
