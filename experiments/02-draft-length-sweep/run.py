"""Sweep the draft length k and measure how acceptance behaves on real GPT-2.

This experiment isolates the *model-level* quantities of speculative decoding --
the acceptance rate alpha and the mean number of confirmed tokens per round --
as a function of the number of draft tokens proposed per round, k. Greedy
decoding makes both quantities deterministic, so a single pass per k reproduces
these numbers exactly and the result is independent of machine load (unlike the
wall-clock speedup measured in experiment 01).

The script never reaches the network: it sets the HuggingFace offline flags and
relies on the checkpoints already being in the local cache. Run from the
repository root with the project virtual environment active:

    python experiments/02-draft-length-sweep/run.py
"""

from __future__ import annotations

import argparse
import os

import torch

from specdec.generate import CachedModel, autoregressive_generate, speculative_generate
from specdec.models import load_hf_pair, select_device

DEFAULT_PROMPT = (
    "Speculative decoding accelerates large language model inference by letting"
    " a small draft model propose tokens that a larger target model verifies in"
    " parallel. In this report we"
)


def sweep(target, draft, prompt_ids, ks, max_new_tokens):
    """Return one record per draft length k.

    The autoregressive baseline is generated once and reused as the reference
    for the per-k losslessness check (greedy speculative output must equal it
    token for token regardless of k).
    """
    ar_tokens, _ = autoregressive_generate(target, prompt_ids, max_new_tokens=max_new_tokens, do_sample=False)
    rows = []
    for k in ks:
        sp_tokens, stats = speculative_generate(
            target, draft, prompt_ids, max_new_tokens=max_new_tokens, k=k, do_sample=False
        )
        rows.append(
            {
                "k": k,
                "alpha": stats.acceptance_rate,
                "accepted_per_round": stats.mean_accepted_per_round,
                "tokens_per_round": stats.mean_tokens_per_round,
                "rounds": stats.rounds,
                "lossless": bool(torch.equal(ar_tokens, sp_tokens)),
            }
        )
    return rows


def format_markdown(rows) -> str:
    header = "| k | alpha (accepted/proposed) | accepted/round | tokens/round | rounds | lossless |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['k']} | {r['alpha']:.3f} | {r['accepted_per_round']:.2f} | "
            f"{r['tokens_per_round']:.2f} | {r['rounds']} | {'yes' if r['lossless'] else 'NO'} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--target", default="gpt2", help="HF target model name")
    parser.add_argument("--draft", default="distilgpt2", help="HF draft model name")
    parser.add_argument("--device", default="cpu", help="auto | cpu | mps | cuda")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Keep the run fully offline; the checkpoints must already be cached.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    target_model, draft_model, tokenizer = load_hf_pair(args.target, args.draft, device=device)
    target = CachedModel(target_model)
    draft = CachedModel(draft_model)
    prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)

    rows = sweep(target, draft, prompt_ids, args.ks, args.max_new_tokens)

    print(
        f"device={device} target={args.target} draft={args.draft} "
        f"max_new_tokens={args.max_new_tokens} decoding=greedy"
    )
    print(format_markdown(rows))


if __name__ == "__main__":
    main()
