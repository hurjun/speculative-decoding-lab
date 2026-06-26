"""Model construction helpers.

Two paths are provided:

- :func:`build_toy_gpt2` builds a small, randomly-initialized GPT-2 with no
  network access. It is used by the unit tests and lets the full
  cache-managed generation code path run offline.
- :func:`load_hf_pair` downloads pretrained checkpoints (e.g. ``distilgpt2`` as
  draft and ``gpt2`` as target). It is only ever called from the benchmark
  script and never from the tests or CI.
"""

from __future__ import annotations

import torch


def select_device(prefer: str = "auto") -> torch.device:
    """Pick a torch device, preferring MPS/CUDA when available."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_toy_gpt2(
    seed: int,
    vocab_size: int = 128,
    n_layer: int = 3,
    n_head: int = 4,
    n_embd: int = 64,
    n_positions: int = 256,
    device: torch.device | str = "cpu",
) -> torch.nn.Module:
    """Build a tiny, randomly-initialized GPT-2 for offline testing.

    Different seeds produce genuinely different next-token distributions, which
    is what makes the draft/target acceptance rate fall strictly between 0 and 1
    in tests.
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_positions=n_positions,
        n_ctx=n_positions,
        bos_token_id=0,
        eos_token_id=vocab_size - 1,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model.to(device)


def load_hf_pair(
    target_name: str,
    draft_name: str,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """Load a pretrained (target, draft) pair and a shared tokenizer.

    Both models must share a tokenizer/vocabulary for the acceptance test to be
    well defined; ``distilgpt2`` and the GPT-2 family satisfy this.

    Returns:
        ``(target_model, draft_model, tokenizer)``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(target_name)
    target = AutoModelForCausalLM.from_pretrained(target_name, torch_dtype=dtype).eval().to(device)
    draft = AutoModelForCausalLM.from_pretrained(draft_name, torch_dtype=dtype).eval().to(device)
    if target.config.vocab_size != draft.config.vocab_size:
        raise ValueError(
            f"target vocab {target.config.vocab_size} != draft vocab {draft.config.vocab_size}; "
            "draft and target must share a vocabulary"
        )
    return target, draft, tokenizer
