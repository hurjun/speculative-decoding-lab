"""Tests tying the acceptance-sampling theory to the implementation.

These complement ``tests/test_sampling.py`` by checking the two closed-form
results derived in ``notes/derivation-acceptance-sampling.md`` against Monte
Carlo estimates produced by :mod:`specdec.sampling`:

- the marginal probability that a single draft token is accepted equals the
  distribution overlap ``sum_x min(p(x), q(x)) = 1 - TV(p, q)``, and
- the expected number of confirmed tokens per round equals
  ``(1 - alpha**(k + 1)) / (1 - alpha)`` under the i.i.d. acceptance model, where
  ``alpha`` is that same overlap.

The generators are seeded, so the trials are deterministic for a given torch
version; the tolerances cover the residual sampling slack with margin.
"""

import torch

from specdec.sampling import sample_from_probs, speculative_verify


def _dist(seed: int, vocab: int) -> torch.Tensor:
    return torch.softmax(torch.randn(vocab, generator=torch.Generator().manual_seed(seed)), dim=-1)


def test_marginal_acceptance_probability_equals_overlap():
    """Empirical P(accept) for k = 1 matches sum_x min(p(x), q(x))."""
    vocab = 7
    p = _dist(3, vocab)
    q = _dist(8, vocab)
    overlap = float(torch.minimum(p, q).sum())  # = 1 - TV(p, q)
    # The pair must be far enough apart that acceptance is clearly below 1.
    assert overlap < 0.9

    target_probs = p.unsqueeze(0).repeat(2, 1)  # (k + 1, vocab), k = 1
    draft_probs = q.unsqueeze(0)  # (k, vocab)
    gen = torch.Generator().manual_seed(2025)
    n_trials = 60000
    accepted = 0
    for _ in range(n_trials):
        draft_token = sample_from_probs(q, gen).view(1)
        _, n_accepted = speculative_verify(target_probs, draft_probs, draft_token, gen)
        accepted += int(n_accepted)  # 0 or 1 when k = 1
    empirical = accepted / n_trials
    assert abs(empirical - overlap) < 0.01, f"empirical={empirical:.4f} overlap={overlap:.4f}"


def test_expected_tokens_per_round_matches_formula():
    """Empirical E[tokens/round] matches (1 - alpha**(k+1)) / (1 - alpha)."""
    vocab = 5
    p = _dist(11, vocab)
    q = _dist(21, vocab)
    alpha = float(torch.minimum(p, q).sum())
    # A mid-range acceptance rate makes the k dependence visible.
    assert 0.3 < alpha < 0.9

    gen = torch.Generator().manual_seed(99)
    n_trials = 40000
    for k in (1, 2, 4, 6):
        target_probs = p.unsqueeze(0).repeat(k + 1, 1)  # every position shares p
        draft_probs = q.unsqueeze(0).repeat(k, 1)  # every position shares q
        total_emitted = 0
        for _ in range(n_trials):
            draft_tokens = sample_from_probs(q.unsqueeze(0).repeat(k, 1), gen)
            emitted, _ = speculative_verify(target_probs, draft_probs, draft_tokens, gen)
            total_emitted += int(emitted.shape[0])
        empirical = total_emitted / n_trials
        expected = (1 - alpha ** (k + 1)) / (1 - alpha)
        assert abs(empirical - expected) < 0.05, f"k={k}: empirical={empirical:.3f} expected={expected:.3f}"
