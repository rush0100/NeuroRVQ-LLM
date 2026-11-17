#!/usr/bin/env bash
set -euo pipefail

# Paper 1 Ablation Row 2 Launcher on DGX Spark (128 GB Unified Memory):
#   - Preprocesses all completed ZuCo 1.0 `task1-SR` subjects (10–11 human subjects) into `./datasets/processed_zuco_10sub`
#   - Trains Stage 1 with upgraded L2-Hypersphere EMA-RVQ + Dead-Code Revival (high codebook perplexity, bounded RVQ loss)
#   - Trains Stage 2 with Joint EEG-RVQ + Qwen2.5-7B BF16 LoRA & 50% Scheduled Target Dropout
#   - Evaluates on the EXACT same unseen test subject (`ZMG`) + Gaussian Noise Sanity Check to compare directly against Row 1!

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/datasets/processed_zuco_10sub"

# Wait up to 90s if the 11th subject (resultsZKW_SR.mat) is on its final few megabytes
for _ in $(seq 1 18); do
  if [[ ! -f "./datasets/zuco_raw/task1-SR/Matlab_files/resultsZKW_SR.mat.aria2" ]]; then
    break
  fi
  echo "[Waiting 5s for 11th subject resultsZKW_SR.mat to finish final chunk...]"
  sleep 5
done

docker run --rm --gpus all --ipc=host --network=host \
  --name bci_paper1_emavq_10sub \
  -v "${PROJECT_DIR}:/workspace" \
  -v "${PROJECT_DIR}/.hf_cache:/root/.cache/huggingface" \
  -w /workspace \
  -e PYTHONUNBUFFERED=1 \
  nvcr.io/nvidia/pytorch:25.04-py3 \
  bash -c '
    set -euo pipefail
    echo "============================================================"
    echo "[Row 2 | Step 1/3] Preprocessing 10-11 Real Human Subjects (task1-SR)..."
    echo "============================================================"
    /workspace/.venv/bin/python -u - << "PYEOF"
import os, pickle
from glob import glob
from tqdm import tqdm
from src.data.zuco_preprocess import process_zuco_v1_mat

mat_dir = "./datasets/zuco_raw/task1-SR/Matlab_files"
mat_files = sorted(mf for mf in glob(os.path.join(mat_dir, "*.mat")) if not os.path.exists(mf + ".aria2"))
print(f"Found {len(mat_files)} complete task1-SR subject MAT files:", [os.path.basename(m) for m in mat_files])

samples = []
for mf in tqdm(mat_files, desc="task1-SR (10+ subs)"):
    samples.extend(process_zuco_v1_mat(mf, task_name="task1-SR", eeg_type="GD", max_eeg_len=58))

os.makedirs("./datasets/processed_zuco_10sub", exist_ok=True)
out_pkl = "./datasets/processed_zuco_10sub/task1-SR.pkl"
with open(out_pkl, "wb") as f:
    pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved {len(samples)} real EEG-text samples across {len(mat_files)} subjects -> {out_pkl}")
PYEOF

    echo "============================================================"
    echo "[Row 2 | Step 2/3] Stage 1: Training Upgraded Hypersphere EMA-RVQ..."
    echo "============================================================"
    /workspace/.venv/bin/python -u -m src.train_stage1 --config configs/spark_8b_emavq_10sub.yaml

    echo "============================================================"
    echo "[Row 2 | Step 3/3] Stage 2: Joint Brain-to-Qwen7B LoRA + Sanity Eval..."
    echo "============================================================"
    /workspace/.venv/bin/python -u -m src.train_stage2_llm --config configs/spark_8b_emavq_10sub.yaml

    chown -R "$(stat -c "%u:%g" /workspace)" /workspace/checkpoints /workspace/logs /workspace/datasets/processed_zuco_10sub
    echo "=== ABLATION ROW 2 (EMA-RVQ + 10-11 SUBJECTS) COMPLETED SUCCESSFULLY! ==="
  '
