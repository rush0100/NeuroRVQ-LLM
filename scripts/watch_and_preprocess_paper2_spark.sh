#!/usr/bin/env bash
# Automated Watcher & 20-Core Preprocessor for Paper 2 (`NeuroReasoner`) on DGX Spark
set -euo pipefail

CONTAINER_ID=$(docker ps -q | head -n 1)
if [[ -z "${CONTAINER_ID}" ]]; then
    echo "ERROR: No active PyTorch NGC container found on DGX Spark!"
    exit 1
fi

RAW_ROOT="${CLINICAL_RAW_ROOT:-./datasets/clinical_paper2}"
OUT_ROOT="${CLINICAL_OUT_ROOT:-./datasets/processed_clinical_neuroreasoner}"
mkdir -p "${OUT_ROOT}"

echo "[$(date '+%F %T')] Step 1: Running immediate 16-core preprocessing on currently ready files (ds004504 88 sub + UIowa subset + MODMA 3ch 55 sub)..."
docker exec -w /workspace "${CONTAINER_ID}" python3 -m src.data.clinical_neuroreasoner_preprocess \
    --raw-root ./datasets/clinical_paper2 \
    --output-dir ./datasets/processed_clinical_neuroreasoner \
    --workers 16

C2_FULL_PROCESSED=0
C3_128_PROCESSED=0

echo "[$(date '+%F %T')] Step 2: Watching Cohort 2 (UNM 54 + UIowa 28) & Cohort 3 (MODMA 128ch 53 sub) downloads to auto-extract & preprocess 100%..."
while true; do
    # Check if all 5 Zenodo zips finished downloading (no .aria2 files left)
    Z_ARIA_COUNT=$(find "${RAW_ROOT}/02_crosscenter_unm_uiowa_macs/archives" -maxdepth 1 -name "*.zip.aria2" 2>/dev/null | wc -l)
    Z_ZIP_COUNT=$(find "${RAW_ROOT}/02_crosscenter_unm_uiowa_macs/archives" -maxdepth 1 -name "*.zip" 2>/dev/null | wc -l)

    if [[ "${Z_ZIP_COUNT}" -eq 5 ]] && [[ "${Z_ARIA_COUNT}" -eq 0 ]] && [[ "${C2_FULL_PROCESSED}" -eq 0 ]]; then
        echo "[$(date '+%F %T')] All 5 UNM + UIowa archives downloaded! Ensuring extraction & running full 82-subject preprocessing..."
        mkdir -p "${RAW_ROOT}/02_crosscenter_unm_uiowa_macs/extracted/Iowa_Data" "${RAW_ROOT}/02_crosscenter_unm_uiowa_macs/extracted/UNM_Data"
        for zf in UI-001.zip UI-002.zip; do
            unzip -q -o "${RAW_ROOT}/02_crosscenter_unm_uiowa_macs/archives/${zf}" -d "${RAW_ROOT}/02_crosscenter_unm_uiowa_macs/extracted/Iowa_Data"
        done
        for zf in UNM-001.zip UNM-002.zip UNM-003.zip; do
            unzip -q -o "${RAW_ROOT}/02_crosscenter_unm_uiowa_macs/archives/${zf}" -d "${RAW_ROOT}/02_crosscenter_unm_uiowa_macs/extracted/UNM_Data"
        done
        docker exec -w /workspace "${CONTAINER_ID}" python3 -m src.data.clinical_neuroreasoner_preprocess \
            --raw-root ./datasets/clinical_paper2 \
            --output-dir ./datasets/processed_clinical_neuroreasoner \
            --workers 16
        C2_FULL_PROCESSED=1
    fi

    # Check if MODMA 128-ch Resting archive finished downloading
    REST128_ZIP="${RAW_ROOT}/03_affective_mdd_modma/archives/854301_EEG_128Channels_Resting_Lanzhou_2015.zip"
    if [[ -f "${REST128_ZIP}" ]] && [[ ! -f "${REST128_ZIP}.aria2" ]] && [[ "${C3_128_PROCESSED}" -eq 0 ]]; then
        echo "[$(date '+%F %T')] MODMA 128-ch Resting archive finished! Extracting & running full preprocessing..."
        mkdir -p "${RAW_ROOT}/03_affective_mdd_modma/extracted/128ch_resting_53sub"
        unzip -q -o "${REST128_ZIP}" -d "${RAW_ROOT}/03_affective_mdd_modma/extracted/128ch_resting_53sub"
        docker exec -w /workspace "${CONTAINER_ID}" python3 -m src.data.clinical_neuroreasoner_preprocess \
            --raw-root ./datasets/clinical_paper2 \
            --output-dir ./datasets/processed_clinical_neuroreasoner \
            --workers 16
        C3_128_PROCESSED=1
    fi

    if [[ "${C2_FULL_PROCESSED}" -eq 1 ]] && [[ "${C3_128_PROCESSED}" -eq 1 ]]; then
        echo "[$(date '+%F %T')] ALL 3 PUBLIC CLINICAL DATASETS (ds004504 88 + UNM 54 + UIowa 28 + MODMA 128ch 53 + MODMA 3ch 55) ARE 100% READY FOR NEUROREASONER!"
        docker exec -w /workspace "${CONTAINER_ID}" python3 -c "
from src.data.clinical_dataset import load_neuroreasoner_splits
for proto in ['multitask_cot', 'eeg_macs_crosscenter', 'ds004504_5fold', 'modma_crossdevice']:
    sp = load_neuroreasoner_splits('./datasets/processed_clinical_neuroreasoner', protocol=proto)
    print(f'Protocol [{proto}]:', {k: len(v) for k, v in sp.items()})
"
        break
    fi

    sleep 25
done
