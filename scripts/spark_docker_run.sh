#!/usr/bin/env bash
set -euo pipefail

# Run any Python / training command inside DGX Spark's NVIDIA Grace Blackwell PyTorch container
# using the persistent `/workspace/.venv` (inherits NVIDIA GB10 PyTorch 2.7 + HuggingFace/PEFT stack).
#
# Usage (on DGX Spark):
#   bash scripts/spark_docker_run.sh python -m src.train_stage1 --config configs/spark_8b_lora.yaml

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${PROJECT_DIR}/.hf_cache"

exec docker run --rm --gpus all --ipc=host --network=host \
  --user "$(id -u):$(id -g)" \
  -v "${PROJECT_DIR}:/workspace" \
  -v "${PROJECT_DIR}/.hf_cache:/tmp/hf_cache" \
  -w /workspace \
  -e HF_HOME="/tmp/hf_cache" \
  -e PATH="/workspace/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  -e PYTHONPATH="/workspace" \
  nvcr.io/nvidia/pytorch:25.04-py3 \
  "$@"
