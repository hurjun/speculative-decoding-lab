"""Tests for K-token verification logic, shapes, and the greedy code path."""

import torch

from specdec.sampling import speculative_verify, speculative_verify_greedy


def _one_hot_logits(tokens, vocab):
    """Logits whose argmax at row j is tokens[j]."""
    logits = torch.full((len(tokens), vocab), -10.0)
    for j, t in enumerate(tokens):
        logits[j, t] = 10.0
    return logits


def test_greedy_all_accepted_emits_drafts_plus_bonus():
    vocab = 16
    draft_tokens = torch.tensor([3, 7, 1])
    # Target argmax agrees with every draft token; bonus argmax is 9.
    target_logits = _one_hot_logits([3, 7, 1, 9], vocab)
    emitted, n_accepted = speculative_verify_greedy(target_logits, draft_tokens)
    assert n_accepted == 3
    assert emitted.tolist() == [3, 7, 1, 9]


def test_greedy_rejection_truncates_and_corrects():
    vocab = 16
    draft_tokens = torch.tensor([3, 7, 1])
    # Target agrees on token 0 (3) but wants 5 at position 1 -> reject there.
    target_logits = _one_hot_logits([3, 5, 1, 9], vocab)
    emitted, n_accepted = speculative_verify_greedy(target_logits, draft_tokens)
    assert n_accepted == 1
    assert emitted.tolist() == [3, 5]


def test_greedy_first_token_rejected():
    vocab = 16
    draft_tokens = torch.tensor([3, 7])
    target_logits = _one_hot_logits([8, 7, 1], vocab)
    emitted, n_accepted = speculative_verify_greedy(target_logits, draft_tokens)
    assert n_accepted == 0
    assert emitted.tolist() == [8]


def test_sampling_verify_output_shapes_and_bounds():
    vocab, k = 12, 4
    gen = torch.Generator().manual_seed(0)
    for _ in range(100):
        target_probs = torch.softmax(torch.randn(k + 1, vocab, generator=gen), dim=-1)
        draft_probs = torch.softmax(torch.randn(k, vocab, generator=gen), dim=-1)
        draft_tokens = torch.randint(0, vocab, (k,), generator=gen)
        emitted, n_accepted = speculative_verify(target_probs, draft_probs, draft_tokens, gen)
        assert 0 <= n_accepted <= k
        assert emitted.shape[0] == n_accepted + 1
        assert emitted.dtype == torch.long
        # Every accepted token must equal the corresponding draft token.
        assert emitted[:n_accepted].tolist() == draft_tokens[:n_accepted].tolist()
