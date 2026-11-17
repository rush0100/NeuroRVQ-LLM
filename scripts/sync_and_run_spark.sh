#!/usr/bin/env bash
set -euo pipefail

# Sync Paper 1 workspace to NVIDIA DGX Spark and optionally run the smoke test remotely.
# Usage:
#   bash scripts/sync_and_run_spark.sh [SPARK_SSH_TARGET]
# Example:
#   bash scripts/sync_and_run_spark.sh user@your-dgx-spark-host
#   bash scripts/sync_and_run_spark.sh dgx-spark.local

SPARK_HOST="${1:-${SPARK_SSH_HOST:-dgx-spark}}"
REMOTE_DIR="${2:-~/Research/bci_eeg_llm}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== [1/3] Checking SSH connectivity to DGX Spark (${SPARK_HOST}) ==="
ssh -o ConnectTimeout=10 "${SPARK_HOST}" "uname -a && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true"

echo "=== [2/3] Syncing Paper 1 codebase to ${SPARK_HOST}:${REMOTE_DIR} ==="
ssh "${SPARK_HOST}" "mkdir -p ${REMOTE_DIR}"
rsync -avz --progress \
  --exclude '._*' \
  --exclude '.DS_Store' \
  --exclude '__pycache__' \
  --exclude 'datasets/zuco_raw' \
  --exclude 'checkpoints' \
  "${LOCAL_DIR}/" "${SPARK_HOST}:${REMOTE_DIR}/"

echo "=== [3/3] Remote DGX Spark Ready! ==="
cat <<EOF
Run on DGX Spark (${SPARK_HOST}):
  ssh ${SPARK_HOST}
  cd ${REMOTE_DIR}

  # Optional: Clear Linux buffer cache on DGX Spark Unified Memory Architecture (UMA) before 32B/70B runs
  # sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

  # A) 2-Minute End-to-End Smoke Test (Synthetic ZuCo -> Stage 1 CET-RVQ -> Stage 2 LLM -> Noise Sanity Check)
  python3 -m src.data.zuco_preprocess --synthetic --output-dir ./datasets/synthetic_zuco --max-eeg-len 32
  python3 -m src.train_stage1 --config configs/smoke_test_tiny.yaml
  python3 -m src.train_stage2_llm --config configs/smoke_test_tiny.yaml

  # B) Fast Real-EEG Prototype (Download 3 subjects from ZuCo 1.0 OSF -> Train Qwen-7B/8B BF16 LoRA)
  python3 scripts/download_zuco_osf.py --quick-subset 3 --output-root ./datasets/zuco_raw
  python3 -m src.data.zuco_preprocess --raw-root ./datasets/zuco_raw --output-dir ./datasets/processed_zuco
  python3 -m src.train_stage1 --config configs/spark_8b_lora.yaml
  python3 -m src.train_stage2_llm --config configs/spark_8b_lora.yaml
EOF
