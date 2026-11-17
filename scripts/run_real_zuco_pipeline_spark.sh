#!/usr/bin/env bash
set -euo pipefail

# Autonomous Background Pipeline on NVIDIA DGX Spark (128 GB Unified Memory):
#   1. Verifies / downloads 3 real human EEG subjects from ZuCo 1.0 `task1-SR`
#   2. Downloads `Qwen/Qwen2.5-7B-Instruct` via ModelScope China CDN (bypasses huggingface.co 0B hang!)
#   3. Preprocesses `.mat` -> `.pkl` inside NVIDIA GB10 PyTorch container
#   4. Trains Stage 1 (`CET-RVQ`) + Stage 2 (`Qwen2.5-7B-Instruct` BF16 LoRA) on NVIDIA GB10
#   5. Runs Strict Teacher-Forcing-Free + Gaussian Noise Sanity Evaluation & writes JSON report

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/.hf_cache" "${PROJECT_DIR}/datasets/zuco_raw" "${PROJECT_DIR}/models/Qwen2.5-7B-Instruct"

aria2_resumable_fetch() {
  local out_dir="$1"
  local out_name="$2"
  local url="$3"
  local dest="${out_dir}/${out_name}"

  for attempt in $(seq 1 100); do
    if [[ -f "${dest}" && ! -f "${dest}.aria2" && -s "${dest}" ]]; then
      echo "  [Verified Complete] ${out_name} ($(du -h "${dest}" | cut -f1))"
      return 0
    fi
    echo "  [aria2c pass ${attempt}] Fetching ${out_name}..."
    if aria2c --continue=true \
      --max-connection-per-server=8 \
      --split=8 \
      --min-split-size=5M \
      --connect-timeout=15 \
      --timeout=30 \
      --retry-wait=3 \
      --max-tries=5 \
      --summary-interval=15 \
      --console-log-level=warn \
      --dir "${out_dir}" \
      --out "${out_name}" \
      "${url}"; then
      if [[ -f "${dest}" && ! -f "${dest}.aria2" ]]; then
        echo "  [Completed] ${out_name} ($(du -h "${dest}" | cut -f1))"
        return 0
      fi
    fi
    echo "  [Refreshing presigned URL] Resuming ${out_name} in 2s..."
    sleep 2
  done

  echo "ERROR: Exceeded max resume attempts for ${out_name}" >&2
  return 1
}

echo "============================================================"
echo "[Step 1/4] Verifying 3 Real ZuCo 1.0 Subjects..."
echo "============================================================"
ZUCO_OUT="./datasets/zuco_raw/task1-SR/Matlab_files"
mkdir -p "${ZUCO_OUT}"
EXISTING_MATS=$(find "${ZUCO_OUT}" -maxdepth 1 -name '*.mat' ! -name '*.aria2' | wc -l)
if [[ "${EXISTING_MATS}" -ge 3 ]]; then
  echo "  [All 3 Real ZuCo MAT files already complete!]"
  ls -lh "${ZUCO_OUT}"/*.mat
else
  mapfile -t ZUCO_ITEMS < <(python3 - << 'PYEOF'
import json, urllib.request

def list_dir(url):
    req = urllib.request.urlopen(url, timeout=45)
    return json.loads(req.read().decode())["data"]

root = list_dir("https://api.osf.io/v2/nodes/q3zws/files/osfstorage/")
mat_files = []
for item in root:
    if "task1" in item["attributes"]["name"].lower():
        for sub in list_dir(item["relationships"]["files"]["links"]["related"]["href"]):
            if "matlab" in sub["attributes"]["name"].lower():
                for m in list_dir(sub["relationships"]["files"]["links"]["related"]["href"]):
                    if m["attributes"]["name"].endswith(".mat"):
                        mat_files.append({
                            "name": m["attributes"]["name"],
                            "url": m["links"]["download"],
                            "size": int(m["attributes"]["size"] or 0),
                        })

for mf in sorted(mat_files, key=lambda x: x["size"])[:3]:
    print(f"{mf['name']}|{mf['url']}|{mf['size']}")
PYEOF
  )
  for entry in "${ZUCO_ITEMS[@]}"; do
    IFS='|' read -r fname furl fsize <<< "${entry}"
    aria2_resumable_fetch "${ZUCO_OUT}" "${fname}" "${furl}"
  done
fi

echo "============================================================"
echo "[Step 2/4] Downloading Qwen/Qwen2.5-7B-Instruct via ModelScope CDN..."
echo "============================================================"
MODEL_DIR="./models/Qwen2.5-7B-Instruct"
MS_API="https://www.modelscope.cn/api/v1/models/Qwen/Qwen2.5-7B-Instruct/repo?Revision=master&FilePath="

SMALL_FILES=(
  "config.json"
  "generation_config.json"
  "tokenizer_config.json"
  "vocab.json"
  "merges.txt"
  "tokenizer.json"
  "model.safetensors.index.json"
)
for fn in "${SMALL_FILES[@]}"; do
  dest="${MODEL_DIR}/${fn}"
  rm -f "${dest}.aria2"
  if [[ -f "${dest}" && -s "${dest}" ]]; then
    echo "  [Verified Complete] ${fn} ($(du -h "${dest}" | cut -f1))"
  else
    echo "  [curl] Fetching ${fn} from ModelScope..."
    curl -fL --retry 5 --connect-timeout 15 -o "${dest}" "${MS_API}${fn}"
    echo "  [Completed] ${fn} ($(du -h "${dest}" | cut -f1))"
  fi
done

SHARD_FILES=(
  "model-00001-of-00004.safetensors"
  "model-00002-of-00004.safetensors"
  "model-00003-of-00004.safetensors"
  "model-00004-of-00004.safetensors"
)
for fn in "${SHARD_FILES[@]}"; do
  aria2_resumable_fetch "${MODEL_DIR}" "${fn}" "${MS_API}${fn}"
done

echo "============================================================"
echo "[Step 3 & 4] Preprocessing + Training Stage 1 & Stage 2 on NVIDIA GB10..."
echo "============================================================"
docker run --rm --gpus all --ipc=host --network=host \
  --name bci_paper1_real3sub \
  -v "${PROJECT_DIR}:/workspace" \
  -v "${PROJECT_DIR}/.hf_cache:/root/.cache/huggingface" \
  -w /workspace \
  -e PYTHONUNBUFFERED=1 \
  nvcr.io/nvidia/pytorch:25.04-py3 \
  bash -c '
    set -euo pipefail
    /workspace/.venv/bin/python -u -m src.data.zuco_preprocess \
      --raw-root ./datasets/zuco_raw \
      --output-dir ./datasets/processed_zuco \
      --eeg-type GD \
      --max-eeg-len 58

    /workspace/.venv/bin/python -u -m src.train_stage1 --config configs/spark_8b_real_subset3.yaml
    /workspace/.venv/bin/python -u -m src.train_stage2_llm --config configs/spark_8b_real_subset3.yaml

    chown -R "$(stat -c "%u:%g" /workspace)" /workspace
    echo "=== ALL PAPER 1 REAL-SUBSET PIPELINE STEPS COMPLETED SUCCESSFULLY! ==="
  '
