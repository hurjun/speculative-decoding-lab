"""Offline acceptance study on the toy backend: draft length k and temperature.

This experiment reproduces the *model-level* behavior of speculative decoding --
the acceptance rate alpha and the mean number of confirmed tokens per round --
entirely offline, using the same tiny randomly-initialized GPT-2 pair as the
unit tests (no network, no checkpoint download). It complements
[experiment 02](../02-draft-length-sweep/), which measures the same quantities on
real GPT-2 but needs the checkpoints cached locally. Because everything here runs
on synthetic weights, anyone can rerun it from a clean checkout in a few seconds,
and the greedy half is bit-for-bit deterministic.

Two sweeps are run:

1. **Greedy draft-length sweep** (deterministic). For k = 1..8 it measures the
   harness acceptance estimate ``accepted / proposed``, the confirmed tokens per
   round, and a token-for-token losslessness check, then compares the measured
   tokens per round against the geometric model ``(1 - alpha**(k+1)) / (1 - alpha)``
   derived in [`notes/derivation-acceptance-sampling.md`](../../notes/derivation-acceptance-sampling.md),
   using the k = 1 acceptance as the per-position alpha.
2. **Sampling temperature sweep** (fixed k). For a range of softmax temperatures
   it measures alpha averaged over several generator seeds. This illustrates the
   identity ``alpha = sum_x min(p, q) = 1 - TV(p, q)`` and its two limits: as
   ``T -> 0`` both distributions collapse onto their argmaxes and alpha tends to
   the greedy argmax-agreement rate, while as ``T -> inf`` both flatten toward
   uniform and alpha tends to 1. Between the limits the curve is U-shaped, dipping
   where the temperature-scaled distributions disagree the most.

Run from the repository root with the project virtual environment active:

    python experiments/03-toy-acceptance-sweep/run.py            # tables + figure
    python experiments/03-toy-acceptance-sweep/run.py --no-plot  # tables only

The figure additionally requires ``matplotlib`` (``pip install matplotlib``); the
measurement path itself depends only on the pinned project requirements.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import torch

from specdec.generate import CachedModel, autoregressive_generate, speculative_generate
from specdec.models import build_toy_gpt2


def build_toy_pair(device: torch.device | str, vocab: int):
    """Build the same toy target/draft pair and prompt the benchmark uses.

    The target is a 4-layer model and the draft a 2-layer model over a shared
    256-token vocabulary; different seeds give genuinely different next-token
    distributions, so the acceptance rate falls strictly between 0 and 1.
    """
    target = CachedModel(
        build_toy_gpt2(seed=1, vocab_size=vocab, n_layer=4, n_head=4, n_embd=128, device=device)
    )
    draft = CachedModel(
        build_toy_gpt2(seed=2, vocab_size=vocab, n_layer=2, n_head=4, n_embd=128, device=device)
    )
    torch.manual_seed(0)
    prompt_ids = torch.randint(0, vocab, (1, 16), device=device)
    return target, draft, prompt_ids


def predicted_tokens_per_round(alpha: float, k: int) -> float:
    """Geometric model ``(1 - alpha**(k+1)) / (1 - alpha)`` for confirmed tokens."""
    if alpha >= 1.0:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def predicted_accept_per_proposed(alpha: float, k: int) -> float:
    """Harness acceptance estimate ``(1/k) * sum_{j=1}^{k} alpha**j`` under constant alpha."""
    return sum(alpha**j for j in range(1, k + 1)) / k


def greedy_k_sweep(target, draft, prompt_ids, ks, max_new_tokens):
    """Deterministic greedy sweep over draft length k.

    The autoregressive baseline is generated once and reused as the reference for
    the per-k losslessness check. The k = 1 acceptance rate is taken as the
    per-position alpha that drives the geometric predictions in the other rows.
    """
    ar_tokens, _ = autoregressive_generate(target, prompt_ids, max_new_tokens=max_new_tokens, do_sample=False)
    rows = []
    alpha1 = None
    for k in ks:
        sp_tokens, stats = speculative_generate(
            target, draft, prompt_ids, max_new_tokens=max_new_tokens, k=k, do_sample=False
        )
        if alpha1 is None:
            alpha1 = stats.acceptance_rate
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
    for r in rows:
        r["pred_alpha"] = predicted_accept_per_proposed(alpha1, r["k"])
        r["pred_tokens_per_round"] = predicted_tokens_per_round(alpha1, r["k"])
    return rows, alpha1


def temperature_sweep(target, draft, prompt_ids, temps, k, seeds, max_new_tokens):
    """Sampling sweep over softmax temperature at fixed k, averaged over seeds.

    Acceptance under sampling is stochastic, so each temperature is averaged over
    ``seeds`` independent generator seeds; the standard deviation across seeds is
    reported as a measure of run-to-run spread.
    """
    rows = []
    for temp in temps:
        alphas, tprs = [], []
        for s in range(seeds):
            generator = torch.Generator().manual_seed(1000 + s)
            _, stats = speculative_generate(
                target,
                draft,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                k=k,
                do_sample=True,
                temperature=temp,
                generator=generator,
            )
            alphas.append(stats.acceptance_rate)
            tprs.append(stats.mean_tokens_per_round)
        rows.append(
            {
                "temperature": temp,
                "alpha_mean": statistics.mean(alphas),
                "alpha_std": statistics.pstdev(alphas),
                "tokens_per_round_mean": statistics.mean(tprs),
                "seeds": seeds,
            }
        )
    return rows


def format_greedy_table(rows) -> str:
    header = "| k | alpha (acc/prop) | model alpha | tokens/round | model tokens/round | rounds | lossless |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['k']} | {r['alpha']:.3f} | {r['pred_alpha']:.3f} | "
            f"{r['tokens_per_round']:.3f} | {r['pred_tokens_per_round']:.3f} | "
            f"{r['rounds']} | {'yes' if r['lossless'] else 'NO'} |"
        )
    return "\n".join(lines)


def format_temperature_table(rows) -> str:
    header = "| temperature | alpha (mean) | alpha (std) | tokens/round | seeds |"
    sep = "|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['temperature']:.2f} | {r['alpha_mean']:.3f} | {r['alpha_std']:.3f} | "
            f"{r['tokens_per_round_mean']:.3f} | {r['seeds']} |"
        )
    return "\n".join(lines)


def write_csv(path: Path, rows, fields) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({key: r[key] for key in fields})


def make_figure(greedy_rows, temp_rows, alpha1, temp_k, out_path: Path) -> None:
    """Render the three-panel summary figure from the measured rows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [r["k"] for r in greedy_rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))

    # Panel (a): greedy tokens per round vs the geometric model.
    ax = axes[0]
    ax.plot(ks, [r["tokens_per_round"] for r in greedy_rows], "o", color="C0", label="measured")
    ax.plot(
        ks,
        [r["pred_tokens_per_round"] for r in greedy_rows],
        "-",
        color="C0",
        alpha=0.6,
        label=r"$(1-\alpha^{k+1})/(1-\alpha)$",
    )
    ax.set_xlabel("draft length k")
    ax.set_ylabel("confirmed tokens / round")
    ax.set_title(f"(a) greedy tokens/round\n(model alpha = {alpha1:.3f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (b): greedy acceptance estimate vs the constant-alpha prediction.
    ax = axes[1]
    ax.plot(ks, [r["alpha"] for r in greedy_rows], "s", color="C1", label="measured")
    ax.plot(
        ks,
        [r["pred_alpha"] for r in greedy_rows],
        "-",
        color="C1",
        alpha=0.6,
        label=r"$\frac{1}{k}\sum_{j=1}^{k}\alpha^{j}$",
    )
    ax.set_xlabel("draft length k")
    ax.set_ylabel("acceptance (accepted / proposed)")
    ax.set_title("(b) greedy acceptance vs k")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (c): sampling acceptance vs temperature (U-shaped between two limits).
    ax = axes[2]
    temps = [r["temperature"] for r in temp_rows]
    means = [r["alpha_mean"] for r in temp_rows]
    stds = [r["alpha_std"] for r in temp_rows]
    ax.errorbar(temps, means, yerr=stds, fmt="^-", color="C2", capsize=3, label="measured alpha")
    ax.axhline(1.0, color="gray", ls=":", lw=1, label=r"$T\to\infty$ limit ($\alpha=1$)")
    ax.axhline(alpha1, color="C0", ls="--", lw=1, label=rf"$T\to 0$ limit (argmax agree {alpha1:.2f})")
    ax.set_xscale("log")
    ax.set_xlabel("sampling temperature T (log scale)")
    ax.set_ylabel(r"acceptance $\alpha = 1 - \mathrm{TV}(p, q)$")
    ax.set_title(f"(c) sampling acceptance vs T\n(k = {temp_k} fixed)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument(
        "--temps",
        type=float,
        nargs="+",
        default=[0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0],
    )
    parser.add_argument("--temp-k", type=int, default=4, help="draft length used in the temperature sweep")
    parser.add_argument("--temp-seeds", type=int, default=8, help="generator seeds averaged per temperature")
    parser.add_argument("--no-plot", action="store_true", help="skip the figure (no matplotlib needed)")
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parent / "results"),
        help="directory for the CSV tables and figure",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    target, draft, prompt_ids = build_toy_pair(device, args.vocab)

    greedy_rows, alpha1 = greedy_k_sweep(target, draft, prompt_ids, args.ks, args.max_new_tokens)
    temp_rows = temperature_sweep(
        target, draft, prompt_ids, args.temps, args.temp_k, args.temp_seeds, args.max_new_tokens
    )

    print(f"device={device} backend=toy vocab={args.vocab} max_new_tokens={args.max_new_tokens}")
    print(f"\ngreedy draft-length sweep (per-position alpha from k=1: {alpha1:.3f})")
    print(format_greedy_table(greedy_rows))
    print(f"\nsampling temperature sweep (k={args.temp_k}, {args.temp_seeds} seeds averaged)")
    print(format_temperature_table(temp_rows))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(
        outdir / "greedy_k_sweep.csv",
        greedy_rows,
        [
            "k",
            "alpha",
            "pred_alpha",
            "accepted_per_round",
            "tokens_per_round",
            "pred_tokens_per_round",
            "rounds",
            "lossless",
        ],
    )
    write_csv(
        outdir / "temperature_sweep.csv",
        temp_rows,
        ["temperature", "alpha_mean", "alpha_std", "tokens_per_round_mean", "seeds"],
    )
    if not args.no_plot:
        make_figure(greedy_rows, temp_rows, alpha1, args.temp_k, outdir / "acceptance_sweep.png")
        print(f"\nwrote tables and figure to {outdir}")
    else:
        print(f"\nwrote tables to {outdir}")


if __name__ == "__main__":
    main()
