#!/usr/bin/env bash
# Reproduce experiment 01: speculative decoding speedup on the GPT-2 family.
# Run from the repository root with the project virtual environment active and
# dependencies installed (see the top-level README). Checkpoints are downloaded
# from HuggingFace on first use.
set -euo pipefail

cd "$(dirname "$0")/../.."

BENCH="python benchmark/benchmark.py --backend hf"

echo "== gpt2-large / distilgpt2 (MPS) =="
$BENCH --target gpt2-large  --draft distilgpt2 --device mps --max-new-tokens 96  --k 4 --runs 3

echo "== gpt2-medium / distilgpt2 (MPS) =="
$BENCH --target gpt2-medium --draft distilgpt2 --device mps --max-new-tokens 128 --k 4 --runs 5

echo "== gpt2-large / gpt2 (MPS) =="
$BENCH --target gpt2-large  --draft gpt2       --device mps --max-new-tokens 96  --k 4 --runs 3

echo "== gpt2 / distilgpt2 (MPS) =="
$BENCH --target gpt2        --draft distilgpt2 --device mps --max-new-tokens 128 --k 4 --runs 5

echo "== gpt2-medium / distilgpt2 (CPU) =="
$BENCH --target gpt2-medium --draft distilgpt2 --device cpu --max-new-tokens 96  --k 4 --runs 3
