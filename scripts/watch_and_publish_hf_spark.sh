#!/usr/bin/env bash
set -euo pipefail

# Background Watcher on NVIDIA DGX Spark:
# Waits cleanly for `bci_paper1_stage2_full` to finish writing `test_evaluation_report.json`,
# then automatically publishes the final checkpoints + LoRA adapter + Model Card to:
#   https://huggingface.co/RUSH0100/RUSH-EEG2Text-Qwen2.5-7B

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
REPORT_FILE="${PROJECT_DIR}/logs/rush_eeg2text_full4task_champion_qwen7b/test_evaluation_report.json"

echo "[HF Watcher] Monitoring Stage 2 completion (waiting for ${REPORT_FILE})..."
while [[ ! -s "${REPORT_FILE}" ]]; do
  sleep 45
done

# Give container 10s to finish chown & exit cleanly
sleep 10
echo "[HF Watcher] Detected completed test_evaluation_report.json! Uploading to RUSH0100/RUSH-EEG2Text-Qwen2.5-7B..."

HF_TOKEN="$(cat ~/.cache/huggingface/token)"
docker run --rm --network=host \
  -v "${PROJECT_DIR}:/workspace" \
  -v "${PROJECT_DIR}/.hf_cache:/root/.cache/huggingface" \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e HF_HUB_DISABLE_XET=1 \
  nvcr.io/nvidia/pytorch:25.04-py3 \
  /workspace/.venv/bin/python -u scripts/publish_to_huggingface.py --repo-id RUSH0100/RUSH-EEG2Text-Qwen2.5-7B

echo "[HF Watcher] Upload to https://huggingface.co/RUSH0100/RUSH-EEG2Text-Qwen2.5-7B COMPLETE!"
