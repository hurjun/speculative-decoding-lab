"""Benchmark speculative decoding against autoregressive decoding.

The script reports only measured numbers on the machine it runs on:

- the acceptance rate ``alpha`` and mean accepted tokens per round,
- a correctness check (greedy speculative output must equal greedy
  autoregressive output, token for token), and
- the wall-clock speedup of speculative over autoregressive decoding.

By default it uses tiny, randomly-initialized toy models and downloads nothing.
Pass ``--backend hf`` to download and benchmark real pretrained checkpoints
(e.g. ``distilgpt2`` as draft and ``gpt2`` as target); this is the only code
path that touches the network.

Examples:
    # Offline structural check on toy models (no download):
    python benchmark/benchmark.py --backend toy

    # Real measured speedup (downloads distilgpt2 and gpt2 on first run):
    python benchmark/benchmark.py --backend hf --target gpt2 --draft distilgpt2 \\
        --max-new-tokens 128 --k 4 --runs 3
"""

from __future__ import annotations

import argparse
import statistics

import torch

from specdec.generate import CachedModel, autoregressive_generate, speculative_generate
from specdec.models import build_toy_gpt2, load_hf_pair, select_device

DEFAULT_PROMPT = (
    "Speculative decoding accelerates large language model inference by"
    " letting a small draft model propose tokens that a larger target model"
    " verifies in parallel. In this report we"
)


def build_models(args, device):
    """Return ``(target, draft, tokenizer_or_none, prompt_ids)``."""
    if args.backend == "toy":
        target = CachedModel(
            build_toy_gpt2(seed=1, vocab_size=256, n_layer=4, n_head=4, n_embd=128, device=device)
        )
        draft = CachedModel(
            build_toy_gpt2(seed=2, vocab_size=256, n_layer=2, n_head=4, n_embd=128, device=device)
        )
        torch.manual_seed(0)
        prompt_ids = torch.randint(0, 256, (1, 16), device=device)
        return target, draft, None, prompt_ids

    target_model, draft_model, tokenizer = load_hf_pair(args.target, args.draft, device=device)
    target = CachedModel(target_model)
    draft = CachedModel(draft_model)
    prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    return target, draft, tokenizer, prompt_ids


def correctness_check(target, draft, prompt_ids, k, max_new_tokens):
    """Greedy speculative output must match greedy autoregressive output exactly."""
    ar, _ = autoregressive_generate(target, prompt_ids, max_new_tokens=max_new_tokens, do_sample=False)
    sp, stats = speculative_generate(
        target, draft, prompt_ids, max_new_tokens=max_new_tokens, k=k, do_sample=False
    )
    matches = bool(torch.equal(ar, sp))
    return matches, ar, stats


def timed_runs(fn, runs):
    times, accept_rates, mean_tpr = [], [], []
    for _ in range(runs):
        _, stats = fn()
        times.append(stats.elapsed_s)
        accept_rates.append(stats.acceptance_rate)
        mean_tpr.append(stats.mean_tokens_per_round)
    return times, accept_rates, mean_tpr


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backend", choices=["toy", "hf"], default="toy", help="model backend (default: toy, no download)"
    )
    parser.add_argument("--target", default="gpt2", help="HF target model name (--backend hf)")
    parser.add_argument("--draft", default="distilgpt2", help="HF draft model name (--backend hf)")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--k", type=int, default=4, help="draft tokens proposed per round")
    parser.add_argument("--runs", type=int, default=3, help="timed repetitions (after one warmup)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    print(f"device: {device}  backend: {args.backend}  k: {args.k}  max_new_tokens: {args.max_new_tokens}")
    if args.backend == "hf":
        print(f"target: {args.target}  draft: {args.draft}")

    target, draft, _, prompt_ids = build_models(args, device)

    # Warmup (kernels, lazy allocations).
    autoregressive_generate(target, prompt_ids, max_new_tokens=8, do_sample=False)
    speculative_generate(target, draft, prompt_ids, max_new_tokens=8, k=args.k, do_sample=False)

    matches, _, _ = correctness_check(target, draft, prompt_ids, args.k, args.max_new_tokens)

    ar_times, _, _ = timed_runs(
        lambda: autoregressive_generate(
            target, prompt_ids, max_new_tokens=args.max_new_tokens, do_sample=False
        ),
        args.runs,
    )
    sp_times, sp_accept, sp_tpr = timed_runs(
        lambda: speculative_generate(
            target, draft, prompt_ids, max_new_tokens=args.max_new_tokens, k=args.k, do_sample=False
        ),
        args.runs,
    )

    ar_t = statistics.median(ar_times)
    sp_t = statistics.median(sp_times)
    n = args.max_new_tokens

    print()
    print(f"greedy correctness (speculative == autoregressive): {'PASS' if matches else 'FAIL'}")
    print(f"acceptance rate alpha:          {statistics.mean(sp_accept):.3f}")
    print(f"mean accepted tokens / round:   {statistics.mean(sp_tpr) - 1:.2f}")
    print(f"mean emitted tokens / round:    {statistics.mean(sp_tpr):.2f}  (= 1 + accepted)")
    print()
    print(f"{'method':<18}{'median s':>12}{'tokens/s':>12}")
    print(f"{'autoregressive':<18}{ar_t:>12.3f}{n / ar_t:>12.1f}")
    print(f"{'speculative':<18}{sp_t:>12.3f}{n / sp_t:>12.1f}")
    print(f"{'speedup':<18}{ar_t / sp_t:>12.2f}x")


if __name__ == "__main__":
    main()
