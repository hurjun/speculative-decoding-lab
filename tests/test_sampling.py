"""Tests for the model-independent acceptance/rejection primitives.

The headline test is :func:`test_emitted_token_matches_target_distribution`,
which verifies by Monte Carlo the central guarantee of speculative decoding:
the emitted token is distributed exactly as the target ``p``, regardless of the
draft ``q``.
"""

import torch

from specdec.sampling import (
    logits_to_probs,
    residual_distribution,
    sample_from_probs,
    speculative_verify,
)


def _random_dist(vocab: int, generator: torch.Generator) -> torch.Tensor:
    logits = torch.randn(vocab, generator=generator)
    return torch.softmax(logits, dim=-1)


def test_logits_to_probs_normalized_and_shaped():
    logits = torch.randn(5, 17)
    probs = logits_to_probs(logits, temperature=0.8)
    assert probs.shape == (5, 17)
    assert torch.allclose(probs.sum(-1), torch.ones(5), atol=1e-5)
    assert (probs >= 0).all()


def test_top_k_restricts_support():
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0])
    probs = logits_to_probs(logits, temperature=1.0, top_k=2)
    assert (probs[2:] == 0).all()
    assert probs[0] > 0 and probs[1] > 0
    assert abs(float(probs.sum()) - 1.0) < 1e-6


def test_top_p_keeps_nucleus():
    logits = torch.tensor([5.0, 4.0, 0.0, -5.0])
    probs = logits_to_probs(logits, temperature=1.0, top_p=0.9)
    # The two long-tail entries must be removed; the kept mass renormalizes to 1.
    assert probs[3] == 0
    assert abs(float(probs.sum()) - 1.0) < 1e-6


def test_residual_distribution_basic():
    p = torch.tensor([0.5, 0.3, 0.2])
    q = torch.tensor([0.1, 0.6, 0.3])
    r = residual_distribution(p, q)
    # max(0, p - q) = [0.4, 0, 0] -> normalized to [1, 0, 0].
    assert torch.allclose(r, torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)


def test_residual_distribution_degenerate_falls_back_to_p():
    p = torch.tensor([0.2, 0.5, 0.3])
    r = residual_distribution(p, p)  # p == q -> zero residual -> fall back to p
    assert torch.allclose(r, p, atol=1e-6)


def test_sample_from_probs_respects_support():
    probs = torch.tensor([0.0, 0.0, 1.0, 0.0])
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        assert int(sample_from_probs(probs, gen)) == 2


def test_q_equals_p_always_accepts():
    gen = torch.Generator().manual_seed(0)
    vocab, k = 8, 5
    p = _random_dist(vocab, torch.Generator().manual_seed(7))
    target_probs = p.unsqueeze(0).repeat(k + 1, 1)
    draft_probs = p.unsqueeze(0).repeat(k, 1)
    for _ in range(50):
        draft_tokens = sample_from_probs(p.unsqueeze(0).repeat(k, 1), gen)
        emitted, n_accepted = speculative_verify(target_probs, draft_probs, draft_tokens, gen)
        assert n_accepted == k
        assert emitted.shape[0] == k + 1


def test_emitted_token_matches_target_distribution():
    """Monte Carlo proof that the emitted token is distributed as the target p.

    With k = 1 the emitted token at position 0 is either the accepted draft
    token or the residual-resampled token. Its empirical distribution over many
    trials must match the target distribution p, independent of the draft q.
    """
    vocab = 6
    p = _random_dist(vocab, torch.Generator().manual_seed(1))
    q = _random_dist(vocab, torch.Generator().manual_seed(2))
    # Make q meaningfully different from p so rejection happens often.
    assert float((p - q).abs().sum()) > 0.3

    target_probs = p.unsqueeze(0).repeat(2, 1)  # (k+1, vocab), k=1
    draft_probs = q.unsqueeze(0)  # (k, vocab)

    gen = torch.Generator().manual_seed(12345)
    n_trials = 40000
    counts = torch.zeros(vocab)
    for _ in range(n_trials):
        draft_token = sample_from_probs(q, gen).view(1)
        emitted, _ = speculative_verify(target_probs, draft_probs, draft_token, gen)
        counts[int(emitted[0])] += 1

    empirical = counts / n_trials
    max_abs_err = float((empirical - p).abs().max())
    # 5-sigma sampling slack for n_trials and vocab this size is ~0.018.
    assert max_abs_err < 0.025, f"empirical={empirical.tolist()} target={p.tolist()}"
