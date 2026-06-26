"""Integration tests on a tiny, randomly-initialized GPT-2 (no network access).

These exercise the full KV-cache-managed generation loop end to end:

- Greedy speculative decoding must be bit-for-bit identical to greedy
  autoregressive decoding (this validates position/cache bookkeeping).
- Under sampling, the first emitted token must follow the target distribution
  (a second, model-level Monte Carlo check on top of the primitive test).
"""

import pytest
import torch

from specdec.generate import CachedModel, autoregressive_generate, speculative_generate
from specdec.models import build_toy_gpt2


@pytest.fixture(scope="module")
def toy_pair():
    target = CachedModel(
        build_toy_gpt2(seed=1, vocab_size=24, n_layer=2, n_head=2, n_embd=32, n_positions=64)
    )
    draft = CachedModel(build_toy_gpt2(seed=2, vocab_size=24, n_layer=2, n_head=2, n_embd=32, n_positions=64))
    return target, draft


@pytest.mark.parametrize("k", [1, 2, 3, 5])
@pytest.mark.parametrize("prompt", [[1, 5, 7, 3], [2, 2, 9, 4, 8, 1]])
def test_greedy_speculative_matches_autoregressive(toy_pair, k, prompt):
    target, draft = toy_pair
    prompt_ids = torch.tensor([prompt])
    ar, _ = autoregressive_generate(target, prompt_ids, max_new_tokens=24, do_sample=False)
    sp, stats = speculative_generate(target, draft, prompt_ids, max_new_tokens=24, k=k, do_sample=False)
    assert torch.equal(ar, sp), f"k={k}: {ar.tolist()} != {sp.tolist()}"
    assert stats.new_tokens == 24


def test_speculative_respects_token_budget(toy_pair):
    target, draft = toy_pair
    prompt_ids = torch.tensor([[1, 2, 3]])
    for n in (1, 7, 13):
        sp, stats = speculative_generate(target, draft, prompt_ids, max_new_tokens=n, k=4, do_sample=False)
        assert sp.shape[1] == n
        assert stats.new_tokens == n


def test_sampling_first_token_follows_target_distribution():
    vocab = 12
    target = CachedModel(
        build_toy_gpt2(seed=3, vocab_size=vocab, n_layer=2, n_head=2, n_embd=16, n_positions=32)
    )
    draft = CachedModel(
        build_toy_gpt2(seed=4, vocab_size=vocab, n_layer=2, n_head=2, n_embd=16, n_positions=32)
    )
    prompt = torch.tensor([[2, 5, 1, 7]])

    # True target next-token distribution at the prompt (temperature 1.0).
    target.reset()
    with torch.no_grad():
        logits = target.forward_suffix(prompt)[0, -1, :].float()
    p_true = torch.softmax(logits, dim=-1)

    gen = torch.Generator().manual_seed(2024)
    n_trials = 2500
    counts = torch.zeros(vocab)
    for _ in range(n_trials):
        generated, _ = speculative_generate(
            target, draft, prompt, max_new_tokens=1, k=3, do_sample=True, temperature=1.0, generator=gen
        )
        counts[int(generated[0, 0])] += 1

    empirical = counts / n_trials
    max_abs_err = float((empirical - p_true).abs().max())
    assert max_abs_err < 0.05, f"empirical={empirical.tolist()} target={p_true.tolist()}"
