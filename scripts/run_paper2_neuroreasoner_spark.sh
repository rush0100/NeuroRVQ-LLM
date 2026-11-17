#!/usr/bin/env bash
set -euo pipefail

# Master Paper 2 (`NeuroReasoner`) Benchmark & Ablation Pipeline on NVIDIA DGX Spark (GB10 128 GB UMA):
#   Run 1 (Main Proposed): Privacy-Preserving `federated_lora` + `EvidenceGroundingBridge` + Local DP (`dp_noise_std=0.001`)
#   Run 2 (Ablation A):    Naive EEG-LLM (`federated_lora` WITHOUT Neural Evidence Grounding `--disable-grounding`)
#   Run 3 (Ablation B):    Centralized Pooled Upper-Bound (`--paradigm centralized`)

PROJECT_DIR="${PROJECT_DIR:-$HOME/Research/bci_eeg_llm}"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/neuroreasoner_multicenter_cot_qwen7b" "${PROJECT_DIR}/checkpoints/neuroreasoner_multicenter_cot_qwen7b"

docker rm -f bci_paper2_neuroreasoner_master 2>/dev/null || true

docker run --rm --gpus all --ipc=host --network=host \
  --name bci_paper2_neuroreasoner_master \
  -v "${PROJECT_DIR}:/workspace" \
  -v "${PROJECT_DIR}/.hf_cache:/root/.cache/huggingface" \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  -e PYTHONUNBUFFERED=1 \
  nvcr.io/nvidia/pytorch:25.04-py3 \
  bash -c '
    set -euo pipefail

    echo "================================================================================"
    echo "[Paper 2 | Run 1/3] Proposed NeuroReasoner (Federated-LoRA + CoT Evidence Grounding)"
    echo "================================================================================"
    if [[ -f "checkpoints/neuroreasoner_multicenter_cot_qwen7b/neuroreasoner_privacy_state_best.pt" ]]; then
      echo "Resuming Continual Fine-Tuning from existing 12-round checkpoint with 430-token CoT budget (4 rounds)..."
      /workspace/.venv/bin/python -u -m src.train_neuroreasoner \
        --config configs/spark_paper2_neuroreasoner_qwen7b.yaml \
        --resume-from-checkpoint checkpoints/neuroreasoner_multicenter_cot_qwen7b/neuroreasoner_privacy_state_best.pt \
        --epochs 4 \
        --lr-lora 3e-5 \
        --max-target-tokens 430 \
        --max-new-tokens 430
    else
      /workspace/.venv/bin/python -u -m src.train_neuroreasoner \
        --config configs/spark_paper2_neuroreasoner_qwen7b.yaml \
        --paradigm federated_lora
    fi

    echo "================================================================================"
    echo "[Paper 2 | Run 2/3] Ablation: Naive Multimodal EEG-LLM (w/o Evidence Grounding)"
    echo "================================================================================"
    /workspace/.venv/bin/python -u -m src.train_neuroreasoner \
      --config configs/spark_paper2_neuroreasoner_qwen7b.yaml \
      --paradigm federated_lora \
      --disable-grounding \
      --exp-suffix ablate_no_grounding

    echo "================================================================================"
    echo "[Paper 2 | Run 3/3] Upper-Bound Reference: Centralized Multi-Hospital Pooling"
    echo "================================================================================"
    /workspace/.venv/bin/python -u -m src.train_neuroreasoner \
      --config configs/spark_paper2_neuroreasoner_qwen7b.yaml \
      --paradigm centralized \
      --exp-suffix centralized_upperbound

    chown -R "$(stat -c "%u:%g" /workspace)" /workspace/checkpoints /workspace/logs
    echo "=== ALL 3 PAPER 2 NEUROREASONER RUNS COMPLETED SUCCESSFULLY! ==="
  ' 2>&1 | tee "${PROJECT_DIR}/logs/neuroreasoner_multicenter_cot_qwen7b/train_paper2_master.log"
