"""Autoregressive and speculative generation loops with KV-cache management.

Both loops share a thin :class:`CachedModel` wrapper that owns a HuggingFace KV
cache (either a ``Cache`` object or a legacy ``(key, value)`` tuple) and only
ever feeds the suffix of the sequence that the cache does not yet cover. The
speculative loop additionally crops the target and
draft caches back to the length of the confirmed prefix after each round, which
is what makes a round cost a single target forward pass over ``K + 1`` positions
regardless of how many draft tokens were rejected.

The design keeps a clean invariant: at the start of every round each cache
covers exactly ``len(confirmed) - 1`` tokens, so the target verification pass
re-reads only the last confirmed token plus the ``K`` draft tokens, and the last
``K + 1`` logit rows of that pass are exactly the distributions needed to verify
the draft and to draw the bonus token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from specdec.sampling import (
    logits_to_probs,
    sample_from_probs,
    speculative_verify,
    speculative_verify_greedy,
)


def _cache_seq_length(cache) -> int:
    """Length covered by a KV cache, for either a ``Cache`` object or a legacy tuple."""
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    return int(cache[0][0].shape[-2])


def _crop_cache(cache, length: int):
    """Shrink a KV cache to ``length`` tokens, for either cache format."""
    if cache is None:
        return None
    if hasattr(cache, "crop"):
        cache.crop(length)
        return cache
    return tuple((k[..., :length, :], v[..., :length, :]) for (k, v) in cache)


class CachedModel:
    """Wraps a causal LM with a growable/croppable KV cache.

    The wrapper is agnostic to the cache representation: HuggingFace models that
    return a :class:`~transformers.cache_utils.Cache` object and those that
    still return a legacy tuple of ``(key, value)`` pairs (e.g. GPT-2 in
    transformers 4.49) are both handled. Only batch size 1 is supported, which
    keeps cache cropping unambiguous and is all the lab's CPU/MPS benchmarks
    need.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.reset()

    def reset(self) -> None:
        self.cache = None
        self.processed = 0

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward_suffix(self, full_ids: torch.Tensor) -> torch.Tensor:
        """Run the model over the tokens of ``full_ids`` not yet in the cache.

        Args:
            full_ids: Long tensor of shape ``(1, seq)`` holding the full
                sequence the cache should cover after this call.

        Returns:
            Logits of shape ``(1, n_new, vocab)`` for the newly processed
            positions; the final row is the next-token distribution for the
            full sequence.
        """
        if full_ids.shape[0] != 1:
            raise ValueError("CachedModel supports batch size 1 only")
        new_ids = full_ids[:, self.processed :]
        if new_ids.shape[1] == 0:
            raise ValueError("forward_suffix called with no new tokens to process")
        out = self.model(input_ids=new_ids, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.processed = full_ids.shape[1]
        return out.logits

    def crop(self, length: int) -> None:
        """Shrink the cache so it covers at most ``length`` tokens."""
        length = min(length, _cache_seq_length(self.cache))
        self.cache = _crop_cache(self.cache, length)
        self.processed = min(self.processed, length)


@dataclass
class GenerationStats:
    """Per-run statistics collected during generation."""

    new_tokens: int = 0
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    elapsed_s: float = 0.0
    per_round_accepted: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed draft tokens that were accepted (alpha)."""
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def mean_accepted_per_round(self) -> float:
        return self.accepted / self.rounds if self.rounds else 0.0

    @property
    def mean_tokens_per_round(self) -> float:
        return self.new_tokens / self.rounds if self.rounds else 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.new_tokens / self.elapsed_s if self.elapsed_s else 0.0


def _next_token(
    logits_row: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator | None,
) -> tuple[int, torch.Tensor | None]:
    """Return ``(token, probs)`` for one position; ``probs`` is ``None`` if greedy."""
    row = logits_row.to(torch.float32).cpu()
    if do_sample:
        probs = logits_to_probs(row, temperature, top_k, top_p)
        return int(sample_from_probs(probs, generator)), probs
    return int(row.argmax()), None


@torch.no_grad()
def autoregressive_generate(
    model: CachedModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, GenerationStats]:
    """Standard one-token-at-a-time decoding, used as the speedup baseline."""
    model.reset()
    device = model.device
    tokens = input_ids.to(device)
    start_len = tokens.shape[1]
    stats = GenerationStats()
    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if start is not None:
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        import time

        t0 = time.perf_counter()

    for _ in range(max_new_tokens):
        logits = model.forward_suffix(tokens)
        token, _ = _next_token(logits[:, -1, :].squeeze(0), do_sample, temperature, top_k, top_p, generator)
        tokens = torch.cat([tokens, torch.tensor([[token]], device=device)], dim=1)

    if start is not None:
        end.record()
        torch.cuda.synchronize()
        stats.elapsed_s = start.elapsed_time(end) / 1000.0
    else:
        if device.type == "mps":
            torch.mps.synchronize()
        stats.elapsed_s = time.perf_counter() - t0

    stats.new_tokens = tokens.shape[1] - start_len
    stats.rounds = stats.new_tokens
    return tokens[:, start_len:], stats


@torch.no_grad()
def speculative_generate(
    target: CachedModel,
    draft: CachedModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    k: int = 4,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, GenerationStats]:
    """Speculative decoding (Leviathan et al., 2023; Chen et al., 2023).

    Each round the draft proposes ``k`` tokens autoregressively, the target
    verifies them in a single forward pass, and accepted tokens plus one
    corrected/bonus token are appended. With greedy decoding the output is
    bit-for-bit identical to :func:`autoregressive_generate`; with sampling the
    emitted distribution is exactly the target's.

    Returns the newly generated tokens (trimmed to ``max_new_tokens``) and the
    run statistics.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    target.reset()
    draft.reset()
    device = target.device
    tokens = input_ids.to(device)
    start_len = tokens.shape[1]
    stats = GenerationStats()

    import time

    t0 = time.perf_counter()

    while tokens.shape[1] - start_len < max_new_tokens:
        prefix_len = tokens.shape[1]

        # --- Draft proposes k tokens autoregressively. ---
        draft_tokens: list[int] = []
        draft_probs_rows: list[torch.Tensor | None] = []
        cur = tokens
        for _ in range(k):
            logits = draft.forward_suffix(cur)
            token, probs = _next_token(
                logits[:, -1, :].squeeze(0), do_sample, temperature, top_k, top_p, generator
            )
            draft_tokens.append(token)
            draft_probs_rows.append(probs)
            cur = torch.cat([cur, torch.tensor([[token]], device=device)], dim=1)
        draft_tok = torch.tensor(draft_tokens, dtype=torch.long)

        # --- Target verifies all k drafts in one forward pass. ---
        target_logits = target.forward_suffix(cur)
        rows = target_logits[0, -(k + 1) :, :].to(torch.float32).cpu()  # (k+1, vocab)

        if do_sample:
            target_probs = logits_to_probs(rows, temperature, top_k, top_p)
            draft_probs = torch.stack([p for p in draft_probs_rows])  # type: ignore[arg-type]
            emitted, n_accepted = speculative_verify(target_probs, draft_probs, draft_tok, generator)
        else:
            emitted, n_accepted = speculative_verify_greedy(rows, draft_tok)

        tokens = torch.cat([tokens, emitted.to(device).view(1, -1)], dim=1)

        # Drop the speculative KV entries beyond the confirmed prefix.
        target.crop(tokens.shape[1] - 1)
        draft.crop(tokens.shape[1] - 1)

        stats.rounds += 1
        stats.proposed += k
        stats.accepted += n_accepted
        stats.per_round_accepted.append(n_accepted)
        # Re-anchor prefix_len for clarity (number actually emitted this round).
        _ = prefix_len

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    stats.elapsed_s = time.perf_counter() - t0

    generated = tokens[:, start_len : start_len + max_new_tokens]
    stats.new_tokens = generated.shape[1]
    return generated, stats
