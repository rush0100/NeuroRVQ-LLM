#!/usr/bin/env bash
set -euo pipefail

# Autonomous Parallel Full-ZuCo Downloader + Full Paper 1 Benchmark Queue on DGX Spark:
#   Phase A (Runs IMMEDIATELY in parallel with current Stage 2 GPU run — uses zero GPU memory!):
#     - Downloads ALL remaining ZuCo 1.0 `.mat` files (`task1-SR`, `task2-NR`, `task3-TSR`)
#     - Downloads ALL ZuCo 2.0 `.mat` files (`task1-NR` -> `task2-NR-2.0`)
#     - Uses presigned-URL auto-refreshing `aria2c` so OSF 60s expiry never aborts the job
#   Phase B (Automatically waits for `bci_paper1_real3sub` to finish cleanly):
#     - Preprocesses ALL downloaded ZuCo 1.0 + 2.0 `.mat` files into `./datasets/processed_zuco/*.pkl`
#     - Launches the upgraded EMA-RVQ + Dead-Code Revival Full Benchmark (`configs/spark_8b_lora.yaml`)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/datasets/zuco_raw"

aria2_resumable_fetch() {
  local out_dir="$1"
  local out_name="$2"
  local url="$3"
  local expected_bytes="${4:-0}"
  local dest="${out_dir}/${out_name}"
  mkdir -p "${out_dir}"

  for attempt in $(seq 1 200); do
    if [[ -f "${dest}" && ! -f "${dest}.aria2" && -s "${dest}" ]]; then
      if [[ "${expected_bytes}" -eq 0 ]] || [[ "$(stat -c '%s' "${dest}")" -eq "${expected_bytes}" ]]; then
        echo "  [Verified Complete] ${out_dir}/${out_name} ($(du -h "${dest}" | cut -f1))"
        return 0
      fi
    fi
    echo "  [aria2c pass ${attempt}] Fetching ${out_dir}/${out_name}..."
    if aria2c --continue=true \
      --max-connection-per-server=8 \
      --split=8 \
      --min-split-size=5M \
      --connect-timeout=15 \
      --timeout=30 \
      --retry-wait=3 \
      --max-tries=5 \
      --summary-interval=30 \
      --console-log-level=warn \
      --dir "${out_dir}" \
      --out "${out_name}" \
      "${url}"; then
      if [[ -f "${dest}" && ! -f "${dest}.aria2" ]]; then
        echo "  [Completed] ${out_dir}/${out_name} ($(du -h "${dest}" | cut -f1))"
        return 0
      fi
    fi
    echo "  [Presigned URL refreshed] Resuming ${out_name} in 2s..."
    sleep 2
  done

  echo "ERROR: Exceeded max resume attempts for ${out_name}" >&2
  return 1
}

echo "=================================================================="
echo "[Phase A] Discovering ALL ZuCo 1.0 & ZuCo 2.0 MAT files on OSF..."
echo "=================================================================="
MANIFEST_FILE="${PROJECT_DIR}/logs/zuco_full_manifest.txt"
python3 -u - << 'PYEOF' > "${MANIFEST_FILE}"
import json, re, urllib.request

def norm_key(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

def list_dir(url):
    items = []
    nxt = url
    while nxt:
        req = urllib.request.urlopen(nxt, timeout=45)
        payload = json.loads(req.read().decode())
        items.extend(payload.get("data", []))
        nxt = payload.get("links", {}).get("next")
    return items

def collect_project(node_id, task_map):
    rows = []
    root = list_dir(f"https://api.osf.io/v2/nodes/{node_id}/files/osfstorage/")
    for item in root:
        attrs = item.get("attributes", {})
        k = norm_key(attrs.get("name", ""))
        if attrs.get("kind") == "folder" and k in task_map:
            local_task = task_map[k]
            sub_url = item["relationships"]["files"]["links"]["related"]["href"]
            for sub in list_dir(sub_url):
                s_attrs = sub.get("attributes", {})
                if s_attrs.get("kind") == "folder" and "matlab" in s_attrs.get("name", "").lower():
                    m_url = sub["relationships"]["files"]["links"]["related"]["href"]
                    for m in sorted(list_dir(m_url), key=lambda x: x["attributes"]["name"]):
                        m_attrs = m.get("attributes", {})
                        if m_attrs.get("kind") == "file" and m_attrs.get("name", "").endswith(".mat"):
                            rows.append((local_task, m_attrs["name"], m["links"]["download"], int(m_attrs.get("size") or 0)))
    return rows

# Priority order matching CET-MAE 3-task benchmark first (task1-SR, task2-NR, task2-NR-2.0), then task3-TSR
v1_rows = collect_project("q3zws", {"task1sr": "task1-SR", "task2nr": "task2-NR", "task3tsr": "task3-TSR"})
v2_rows = collect_project("2urht", {"task1nr": "task2-NR-2.0"})

ordered = (
    [r for r in v1_rows if r[0] == "task1-SR"]
    + [r for r in v1_rows if r[0] == "task2-NR"]
    + v2_rows
    + [r for r in v1_rows if r[0] == "task3-TSR"]
)
for task, name, url, size in ordered:
    print(f"{task}|{name}|{url}|{size}")
PYEOF

TOTAL_FILES=$(wc -l < "${MANIFEST_FILE}")
echo "[Phase A] Queued ${TOTAL_FILES} total MAT files across ZuCo 1.0 & ZuCo 2.0."

IDX=0
while IFS='|' read -r ltask fname furl fsize; do
  IDX=$((IDX + 1))
  echo "------------------------------------------------------------------"
  echo "[File ${IDX}/${TOTAL_FILES}] Task=${ltask} | ${fname} ($((fsize / 1048576)) MB)"
  echo "------------------------------------------------------------------"
  aria2_resumable_fetch "./datasets/zuco_raw/${ltask}/Matlab_files" "${fname}" "${furl}" "${fsize}"
done < "${MANIFEST_FILE}"

echo "=================================================================="
echo "[Phase B] All ZuCo 1.0 + 2.0 MAT Downloads Finished!"
echo "Waiting for current 3-subject GPU run (bci_paper1_real3sub) if still active..."
echo "=================================================================="
while docker ps --format '{{.Names}}' | grep -Eqx 'bci_paper1_real3sub|bci_paper1_emavq_10sub|bci_stage1_opt_loop'; do
  echo "  [Waiting 60s for active Paper 1 GPU run to complete...]"
  sleep 60
done

if [[ -f "${PROJECT_DIR}/logs/HOLD_BEFORE_STAGE2" ]]; then
  echo "[Phase B Complete] All ZuCo 1.0 + 2.0 MAT files downloaded! Holding auto-launch per user Stage-1-first policy."
  exit 0
fi

echo "=================================================================="
echo "[Phase C] Preprocessing Full ZuCo 1.0 + 2.0 & Launching Upgraded EMA-RVQ Full Run!"
echo "=================================================================="
docker run --rm --gpus all --ipc=host --network=host \
  --name bci_paper1_full_zuco \
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

    /workspace/.venv/bin/python -u -m src.train_stage1 --config configs/spark_8b_lora.yaml
    /workspace/.venv/bin/python -u -m src.train_stage2_llm --config configs/spark_8b_lora.yaml

    chown -R "$(stat -c "%u:%g" /workspace)" /workspace
    echo "=== FULL ZUCO PAPER 1 BENCHMARK COMPLETED SUCCESSFULLY! ==="
  '
