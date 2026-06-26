"""Speculative decoding lab: a from-scratch implementation of speculative
decoding for autoregressive language models.

The public surface is intentionally small. The two pieces a reader is most
likely to want are:

- ``specdec.sampling``: the distribution-preserving acceptance/rejection
  primitives (Leviathan et al., 2023; Chen et al., 2023). These operate on
  plain probability tensors and have no dependency on any particular model.
- ``specdec.generate``: KV-cache-based autoregressive and speculative
  generation loops built on top of those primitives.
"""

from specdec.sampling import (
    logits_to_probs,
    residual_distribution,
    sample_from_probs,
    speculative_verify,
    speculative_verify_greedy,
)

__all__ = [
    "logits_to_probs",
    "residual_distribution",
    "sample_from_probs",
    "speculative_verify",
    "speculative_verify_greedy",
]

__version__ = "0.1.0"
