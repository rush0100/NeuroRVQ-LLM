#!/usr/bin/env bash
set -euo pipefail

# Master Full-Scale Paper 1 Pipeline on NVIDIA DGX Spark (128 GB Unified Memory):
#   1. Parallel 16-20 Core Connectome + 8-Metadata Preprocessing across ALL 54 `.mat` files (61.5 GB)
#      (`task1-SR`, `task2-NR`, `task3-TSR` from ZuCo 1.0 + `task2-NR-2.0` from ZuCo 2.0)
#   2. Stage 1 Champion Training (`dual_late_fusion`: 3D 12-ROI + Late Connectome Bottleneck + 6-Book EMA-RVQ)
#   3. Stage 2 Full-Scale Joint Brain-to-Qwen2.5-7B BF16 LoRA Training + Strict Teacher-Forcing-Free & Noise Sanity Eval

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
rm -f "${PROJECT_DIR}/logs/HOLD_BEFORE_STAGE2"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/datasets/processed_zuco_connectome_full"

docker rm -f bci_paper1_full4task_master 2>/dev/null || true

docker run --rm --gpus all --ipc=host --network=host \
  --name bci_paper1_full4task_master \
  -v "${PROJECT_DIR}:/workspace" \
  -v "${PROJECT_DIR}/.hf_cache:/root/.cache/huggingface" \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  -e PYTHONUNBUFFERED=1 \
  nvcr.io/nvidia/pytorch:25.04-py3 \
  bash -c '
    set -euo pipefail

    echo "================================================================================"
    echo "[Step 1/3] 20-Core Parallel Connectome + 8-Metadata Preprocessing (All 4 Tasks)"
    echo "================================================================================"
    /workspace/.venv/bin/python -u -m src.data.zuco_preprocess \
      --raw-root ./datasets/zuco_raw \
      --metadata-dir ./datasets/zuco_metadata \
      --output-dir ./datasets/processed_zuco_connectome_full \
      --tasks task1-SR task2-NR task3-TSR task2-NR-2.0 \
      --workers 16

    /workspace/.venv/bin/python -u - << "PYEOF"
import glob, os, pickle
total = 0
all_subs = set()
for p in sorted(glob.glob("./datasets/processed_zuco_connectome_full/*.pkl")):
    if "llm_stage1_cache" in p:
        continue
    data = pickle.load(open(p, "rb"))
    subs = sorted({x["subject_id"] for x in data})
    all_subs.update(subs)
    total += len(data)
    print(f"  [Verified PKL] {os.path.basename(p):<20} -> {len(data):5d} samples | {len(subs):2d} subjects: {subs}")
print(f"  >>> GRAND TOTAL: {total} Dual-Stream Connectome samples across {len(all_subs)} unique human subjects! <<<")
PYEOF

    echo "================================================================================"
    echo "[Step 2/3] Full-Scale Stage 1 Champion Training (4-Task Connectome + EMA-RVQ)"
    echo "================================================================================"
    /workspace/.venv/bin/python -u -m src.train_stage1 \
      --config configs/spark_full4task_champion_qwen7b.yaml

    echo "================================================================================"
    echo "[Step 3/3] Full-Scale Stage 2 Joint Qwen2.5-7B BF16 LoRA + Noise Sanity Eval"
    echo "================================================================================"
    /workspace/.venv/bin/python -u -m src.train_stage2_llm \
      --config configs/spark_full4task_champion_qwen7b.yaml

    chown -R "$(stat -c "%u:%g" /workspace)" \
      /workspace/datasets/processed_zuco_connectome_full \
      /workspace/checkpoints \
      /workspace/logs
    echo "=== FULL-SCALE 4-TASK PAPER 1 PIPELINE COMPLETED SUCCESSFULLY! ==="
  '
