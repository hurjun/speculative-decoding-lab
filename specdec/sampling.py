"""Distribution-preserving acceptance/rejection primitives for speculative decoding.

These functions implement the modified rejection sampling scheme of
Leviathan et al. (2023) and Chen et al. (2023). They operate on probability
tensors only and do not depend on any model, which makes the correctness of
the core algorithm directly testable by Monte Carlo (see ``tests/``).

Notation follows the speculative decoding papers: ``p`` is the target
distribution and ``q`` is the draft distribution. A draft token ``x`` sampled
from ``q`` is accepted with probability ``min(1, p(x) / q(x))``. On rejection,
a replacement token is drawn from the normalized residual
``norm(max(0, p - q))``. The marginal distribution of the emitted token is then
exactly ``p``, independent of the draft ``q``.
"""

from __future__ import annotations

import torch

_EPS = 1e-12


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Map logits to a (optionally temperature/top-k/top-p adjusted) distribution.

    Args:
        logits: Tensor of shape ``(..., vocab)``. The same transformation is
            applied independently to every leading row, so a batched
            ``(K + 1, vocab)`` tensor and a single ``(vocab,)`` tensor are both
            valid.
        temperature: Softmax temperature. Must be strictly positive; greedy
            decoding is handled separately in :mod:`specdec.generate` rather
            than by ``temperature = 0`` to avoid degenerate divisions.
        top_k: If greater than 0, keep only the ``top_k`` highest-logit
            entries per row before normalizing.
        top_p: If less than 1.0, apply nucleus filtering: keep the smallest set
            of entries whose cumulative probability mass reaches ``top_p``.

    Returns:
        A tensor of the same shape as ``logits`` whose last dimension sums to 1.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0; use greedy decoding for argmax behavior")

    logits = logits.to(torch.float32) / temperature

    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth_vals = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth_vals, torch.full_like(logits, float("-inf")), logits)

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Keep tokens up to and including the one that crosses the top_p mass.
        remove_sorted = cumulative - sorted_probs > top_p
        remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_idx, remove_sorted)
        logits = torch.where(remove, torch.full_like(logits, float("-inf")), logits)

    return torch.softmax(logits, dim=-1)


def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Return the normalized residual ``norm(max(0, p - q))``.

    This is the distribution a replacement token is drawn from when a draft
    token is rejected. On a genuine rejection the residual is guaranteed to
    carry positive mass (its total mass equals the total variation distance
    between ``p`` and ``q``, which is positive unless ``p == q``); the ``eps``
    guard only protects against numerical degeneracy, in which case the
    function falls back to the target distribution ``p``.

    Args:
        p: Target distribution of shape ``(..., vocab)``.
        q: Draft distribution of shape ``(..., vocab)``.

    Returns:
        A distribution of the same shape whose last dimension sums to 1.
    """
    residual = torch.clamp(p - q, min=0.0)
    mass = residual.sum(dim=-1, keepdim=True)
    safe = mass > _EPS
    normalized = residual / torch.where(safe, mass, torch.ones_like(mass))
    return torch.where(safe, normalized, p)


def sample_from_probs(probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw one categorical sample per distribution row.

    Args:
        probs: Tensor of shape ``(..., vocab)`` whose last dimension sums to 1.
        generator: Optional ``torch.Generator`` for reproducibility.

    Returns:
        A long tensor of shape ``probs.shape[:-1]`` (a scalar tensor when
        ``probs`` is 1-D).
    """
    flat = probs.reshape(-1, probs.shape[-1])
    idx = torch.multinomial(flat, num_samples=1, generator=generator).squeeze(-1)
    return idx.reshape(probs.shape[:-1])


def speculative_verify(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, int]:
    """Verify ``K`` draft tokens against the target via modified rejection sampling.

    Args:
        target_probs: Target distributions, shape ``(K + 1, vocab)``. Row ``j``
            for ``j < K`` is ``p(. | context, draft_tokens[:j])`` and is used to
            verify ``draft_tokens[j]``. Row ``K`` is the distribution for the
            bonus token emitted when every draft token is accepted.
        draft_probs: Draft distributions, shape ``(K, vocab)``. Row ``j`` is the
            distribution ``q`` that ``draft_tokens[j]`` was sampled from.
        draft_tokens: Proposed tokens, shape ``(K,)``.
        generator: Optional ``torch.Generator`` for reproducibility.

    Returns:
        ``(emitted, n_accepted)`` where ``emitted`` is a 1-D long tensor of the
        tokens to append (length ``n_accepted + 1``) and ``n_accepted`` is the
        number of draft tokens accepted before the first rejection (``K`` if all
        were accepted).
    """
    k = int(draft_tokens.shape[0])
    emitted: list[int] = []
    for j in range(k):
        token = int(draft_tokens[j])
        p_x = float(target_probs[j, token])
        q_x = float(draft_probs[j, token])
        ratio = 1.0 if q_x <= 0.0 else p_x / q_x
        r = float(torch.rand((), generator=generator))
        if r <= min(1.0, ratio):
            emitted.append(token)
        else:
            corrected = sample_from_probs(residual_distribution(target_probs[j], draft_probs[j]), generator)
            emitted.append(int(corrected))
            return torch.tensor(emitted, dtype=torch.long), j
    bonus = sample_from_probs(target_probs[k], generator)
    emitted.append(int(bonus))
    return torch.tensor(emitted, dtype=torch.long), k


def speculative_verify_greedy(
    target_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Greedy (temperature-0) verification.

    A draft token is accepted iff it equals the target's argmax at that
    position. On rejection the target's argmax token is emitted instead. The
    emitted sequence is therefore identical, token for token, to greedy
    autoregressive decoding of the target model -- a property the tests assert
    directly.

    Args:
        target_logits: Target logits of shape ``(K + 1, vocab)``.
        draft_tokens: Proposed tokens of shape ``(K,)``.

    Returns:
        ``(emitted, n_accepted)`` with the same meaning as
        :func:`speculative_verify`.
    """
    k = int(draft_tokens.shape[0])
    target_argmax = target_logits.argmax(dim=-1)
    emitted: list[int] = []
    for j in range(k):
        if int(draft_tokens[j]) == int(target_argmax[j]):
            emitted.append(int(draft_tokens[j]))
        else:
            emitted.append(int(target_argmax[j]))
            return torch.tensor(emitted, dtype=torch.long), j
    emitted.append(int(target_argmax[k]))
    return torch.tensor(emitted, dtype=torch.long), k
