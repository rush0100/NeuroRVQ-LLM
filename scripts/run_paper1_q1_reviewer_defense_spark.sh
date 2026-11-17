#!/usr/bin/env bash
set -euo pipefail

# Master Q1 Reviewer Defense Pipeline (Remaining Blocks: Action 3 -> Action 1c -> Action 2 -> Paper 2 NeuroReasoner)
# Skips already-completed Action 4 & Action 1a/1b!

PROJECT_DIR="/home/rush0100/Research/bci_eeg_llm"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/q1_reviewer_defense"

docker rm -f bci_q1_defense_master 2>/dev/null || true
docker rm -f bci_paper2_neuroreasoner_master 2>/dev/null || true

docker run -d --gpus all --ipc=host --network=host \
  --name bci_q1_defense_master \
  -v "${PROJECT_DIR}:/workspace" \
  -v "${PROJECT_DIR}/.hf_cache:/root/.cache/huggingface" \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  -e PYTHONUNBUFFERED=1 \
  nvcr.io/nvidia/pytorch:25.04-py3 \
  bash -c '
    set -euo pipefail
    exec > >(tee -a /workspace/logs/q1_reviewer_defense/master_q1_defense.log) 2>&1

    echo "================================================================================"
    echo "[Action 3 | ~10 min] Fair Backbone & Single-Task Controlled Baseline (NeuroRVQ + BART-large 406M on task1-SR)"
    echo "================================================================================"
    if [[ ! -f "checkpoints/q1_action3_neurorvq_bart_large_singletask/stage1_cet_rvq_best.pt" ]]; then
      /workspace/.venv/bin/python -u -m src.train_stage1 \
        --config configs/q1_action3_bart_large_singletask.yaml
    else
      echo "Stage 1 checkpoint already exists, proceeding directly to Stage 2 BART-large fine-tuning!"
    fi
    /workspace/.venv/bin/python -u -m src.train_stage2_llm \
      --config configs/q1_action3_bart_large_singletask.yaml

    echo "================================================================================"
    echo "[Action 1c | ~7 min] Full 4-Task Eye-Tracking-Free Mode (eeg_only_no_oculomotor across 30 subjects, 1,313 sentences)"
    echo "================================================================================"
    if [[ ! -f "checkpoints/q1_action1c_eeg_only_full4task/stage1_cet_rvq_best.pt" ]]; then
      /workspace/.venv/bin/python -u -m src.train_stage1 \
        --config configs/q1_action1c_eeg_only_full4task.yaml
    fi

    echo "================================================================================"
    echo "[Action 2 | ~25 min] Strict Double Zero-Shot (Unseen Subject + Unseen Sentence Pool) Stage 1 & Stage 2"
    echo "================================================================================"
    if [[ ! -f "checkpoints/q1_action2_double_zeroshot/stage1_cet_rvq_best.pt" ]]; then
      /workspace/.venv/bin/python -u -m src.train_stage1 \
        --config configs/q1_action2_unseen_subject_and_sentence.yaml
    fi
    /workspace/.venv/bin/python -u -m src.train_stage2_llm \
      --config configs/q1_action2_unseen_subject_and_sentence.yaml

    chown -R "$(stat -c "%u:%g" /workspace)" /workspace/checkpoints /workspace/logs
    echo "=== ALL 4 Q1 REVIEWER DEFENSE EXPERIMENTS COMPLETED 100%! ==="

    echo "================================================================================"
    echo "[Auto-Chain -> Paper 2] Launching Full NeuroReasoner Federated-LoRA Suite"
    echo "================================================================================"
    /workspace/.venv/bin/python -u -m src.train_neuroreasoner \
      --config configs/spark_paper2_neuroreasoner_qwen7b.yaml \
      --paradigm federated_lora \
      2>&1 | tee -a /workspace/logs/neuroreasoner_multicenter_cot_qwen7b/train_paper2_master.log

    /workspace/.venv/bin/python -u -m src.train_neuroreasoner \
      --config configs/spark_paper2_neuroreasoner_qwen7b.yaml \
      --paradigm federated_lora \
      --disable-grounding \
      --exp-suffix ablate_no_grounding \
      2>&1 | tee -a /workspace/logs/neuroreasoner_multicenter_cot_qwen7b/train_paper2_master.log

    /workspace/.venv/bin/python -u -m src.train_neuroreasoner \
      --config configs/spark_paper2_neuroreasoner_qwen7b.yaml \
      --paradigm centralized \
      --exp-suffix centralized_upperbound \
      2>&1 | tee -a /workspace/logs/neuroreasoner_multicenter_cot_qwen7b/train_paper2_master.log

    chown -R "$(stat -c "%u:%g" /workspace)" /workspace/checkpoints /workspace/logs
    echo "=== BOTH PAPER 1 Q1 DEFENSE SUITE AND PAPER 2 NEUROREASONER COMPLETED 100%! ==="
  '

echo "Launched persistent container bci_q1_defense_master!"
