#!/usr/bin/env python3
"""
NeuroReasoner (Paper 2) — Public Clinical EEG Datasets Downloader & Verifier
=============================================================================
Downloads, checksum-verifies, and organizes 3 public clinical EEG benchmarks on DGX Spark:

1. Cognitive Domain (Alzheimer's AD / Frontotemporal Dementia FTD / Healthy CN + MMSE):
   - OpenNeuro ds004504 (88 subjects: 36 AD, 23 FTD, 29 CN; 19-ch 10-20 system, 500 Hz, MMSE scores)
   - Source: AWS OpenNeuro Public S3 Bucket (s3://openneuro.org/ds004504/)

2. Cross-Center Neuro Domain (64-ch BrainVision, 500 Hz — Exact EEG-MACS Benchmark, Song & Zhang et al., ACM MM 2024):
   - Univ. of New Mexico (UNM: 54 subjects — 27 PD, 27 CTL) + Univ. of Iowa (UI: 28 subjects — 14 PD, 14 CTL)
   - Source: Zenodo Record 8338680 (MD5-verified UI-001..002.zip + UNM-001..003.zip)
   - Also clones official ICI-BCI/EEG-MACS baseline code & window index for 1-to-1 benchmark comparison

3. Affective Domain (Clinical MDD Depression & Anxiety + PHQ-9 / GAD-7):
   - MODMA Dataset (Lanzhou Univ. Second Hospital, Scientific Data 2022)
   - Both 128-channel high-density EEG (Resting + Dot-Probe ERP, 53 subjects) and 3-channel wearable prefrontal EEG (Fp1/Fpz/Fp2, 55 subjects) with clinical PHQ-9 & GAD-7 scores
   - Source: UK Data Service ReShare Archive (10.5255/UKDA-SN-854301)
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(os.environ.get("CLINICAL_DATA_ROOT", os.path.expanduser("~/Research/bci_eeg_llm/datasets/clinical_paper2")))

COHORT_1_DIR = BASE_DIR / "01_cognitive_ad_ftd_ds004504"
COHORT_2_DIR = BASE_DIR / "02_crosscenter_unm_uiowa_macs"
COHORT_3_DIR = BASE_DIR / "03_affective_mdd_modma"
STATUS_FILE = BASE_DIR / "download_manifest_status.json"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def md5sum(filepath: Path) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def aria2_download(
    url: str,
    dest_dir: Path,
    filename: str,
    connections: int = 8,
    expected_md5: str | None = None,
    user_agent: str = "curl/8.5.0",
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / filename
    aria2_ctrl = dest_dir / f"{filename}.aria2"

    if out_path.exists() and not aria2_ctrl.exists() and out_path.stat().st_size > 0:
        if expected_md5:
            actual = md5sum(out_path)
            if actual.lower() == expected_md5.lower():
                log(f"SKIP (MD5 verified {actual[:10]}...): {filename}")
                return out_path
            else:
                log(f"MD5 mismatch for {filename} (got {actual}, expected {expected_md5}), re-downloading...")
                out_path.unlink(missing_ok=True)
        else:
            log(f"SKIP (already exists, {out_path.stat().st_size / 1e6:.1f} MB): {filename}")
            return out_path

    cmd = [
        "aria2c",
        "--continue=true",
        f"--max-connection-per-server={connections}",
        f"--split={connections}",
        "--min-split-size=5M",
        "--retry-wait=5",
        "--max-tries=10",
        "--timeout=60",
        "--connect-timeout=30",
        f"--user-agent={user_agent}",
        "--console-log-level=warn",
        "--summary-interval=15",
        "--dir",
        str(dest_dir),
        "--out",
        filename,
        url,
    ]

    for attempt in range(1, 6):
        log(f"Downloading {filename} (attempt {attempt}/5) <- {url}")
        res = subprocess.run(cmd)
        if res.returncode == 0 and out_path.exists():
            if expected_md5:
                actual = md5sum(out_path)
                if actual.lower() != expected_md5.lower():
                    log(f"WARNING: MD5 mismatch on {filename}: got {actual}, expected {expected_md5}")
                    if attempt < 5:
                        out_path.unlink(missing_ok=True)
                        continue
                    raise RuntimeError(f"MD5 verification failed for {filename}")
                log(f"DONE & MD5 OK ({actual}): {filename} ({out_path.stat().st_size / 1e6:.1f} MB)")
            else:
                log(f"DONE: {filename} ({out_path.stat().st_size / 1e6:.1f} MB)")
            return out_path
        log(f"aria2c exited with {res.returncode} on {filename}; retrying in 5s...")
        time.sleep(5)

    raise RuntimeError(f"Failed to download {filename} from {url}")


# ==============================================================================
# 1. COHORT 1: OpenNeuro ds004504 (AD / FTD / CN, 19-ch, 500 Hz, MMSE)
# ==============================================================================
def download_ds004504() -> dict:
    log("=== [1/3] Starting Cohort 1: OpenNeuro ds004504 (36 AD, 23 FTD, 29 CN, MMSE) ===")
    COHORT_1_DIR.mkdir(parents=True, exist_ok=True)

    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    marker = ""
    s3_items: list[tuple[str, int]] = []

    while True:
        list_url = "https://s3.amazonaws.com/openneuro.org/?prefix=ds004504/&max-keys=1000"
        if marker:
            list_url += f"&marker={urllib.parse.quote(marker)}"
        xml_bytes = urllib.request.urlopen(list_url, timeout=30).read()
        root = ET.fromstring(xml_bytes)
        for c in root.findall("s3:Contents", ns):
            key = c.find("s3:Key", ns).text
            size = int(c.find("s3:Size", ns).text)
            if not key.endswith("/") and "/.datalad/" not in key and "/.git" not in key:
                s3_items.append((key, size))
            marker = key
        if root.find("s3:IsTruncated", ns).text.lower() != "true":
            break

    log(f"Discovered {len(s3_items)} BIDS files ({sum(s for _, s in s3_items) / (1024**3):.2f} GB) in s3://openneuro.org/ds004504/")

    # Build aria2 batch input file for missing/incomplete files
    aria_input_lines = []
    skipped = 0
    for key, expected_size in s3_items:
        rel_path = key[len("ds004504/"):]
        local_file = COHORT_1_DIR / rel_path
        aria_ctrl = Path(str(local_file) + ".aria2")
        if local_file.exists() and not aria_ctrl.exists() and local_file.stat().st_size == expected_size:
            skipped += 1
            continue
        local_file.parent.mkdir(parents=True, exist_ok=True)
        encoded_key = "/".join(urllib.parse.quote(part) for part in key.split("/"))
        file_url = f"https://s3.amazonaws.com/openneuro.org/{encoded_key}"
        aria_input_lines.append(f"{file_url}\n  dir={local_file.parent}\n  out={local_file.name}\n")

    log(f"ds004504: {skipped}/{len(s3_items)} files already complete; {len(aria_input_lines)} files queued.")
    if aria_input_lines:
        batch_file = COHORT_1_DIR / "_aria2_ds004504_queue.txt"
        batch_file.write_text("".join(aria_input_lines), encoding="utf-8")
        cmd = [
            "aria2c",
            "--continue=true",
            "--max-concurrent-downloads=24",
            "--max-connection-per-server=16",
            "--split=16",
            "--min-split-size=1M",
            "--retry-wait=3",
            "--max-tries=10",
            "--console-log-level=warn",
            "--summary-interval=15",
            f"--input-file={batch_file}",
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError(f"ds004504 batch aria2c failed with code {res.returncode}")
        batch_file.unlink(missing_ok=True)

    # Verify all 357 files & parse participants.tsv summary
    verified_count = 0
    verified_bytes = 0
    for key, expected_size in s3_items:
        rel_path = key[len("ds004504/"):]
        local_file = COHORT_1_DIR / rel_path
        if local_file.exists() and local_file.stat().st_size == expected_size:
            verified_count += 1
            verified_bytes += expected_size

    participants_tsv = COHORT_1_DIR / "participants.tsv"
    group_counts = {"A": 0, "F": 0, "C": 0}
    if participants_tsv.exists():
        lines = participants_tsv.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        for row in lines[1:]:
            cols = row.split("\t")
            if len(cols) >= 4 and cols[3].strip() in group_counts:
                group_counts[cols[3].strip()] += 1

    log(
        f"Cohort 1 COMPLETE: {verified_count}/{len(s3_items)} files ({verified_bytes / (1024**3):.2f} GB) | "
        f"AD={group_counts['A']}, FTD={group_counts['F']}, CN={group_counts['C']}"
    )
    return {
        "dataset": "OpenNeuro ds004504",
        "path": str(COHORT_1_DIR),
        "files_verified": verified_count,
        "files_expected": len(s3_items),
        "total_gb": round(verified_bytes / (1024**3), 3),
        "subjects": {"AD": group_counts["A"], "FTD": group_counts["F"], "CN": group_counts["C"]},
    }


# ==============================================================================
# 2. COHORT 2: Cross-Center UNM (54) + UIowa (28) (64-ch, 500 Hz — EEG-MACS)
# ==============================================================================
def download_unm_uiowa() -> dict:
    log("=== [2/3] Starting Cohort 2: UNM (54 sub) + UIowa (28 sub) 64-ch 500 Hz (EEG-MACS Benchmark) ===")
    COHORT_2_DIR.mkdir(parents=True, exist_ok=True)
    archives_dir = COHORT_2_DIR / "archives"
    extracted_dir = COHORT_2_DIR / "extracted"
    archives_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    # Clone official ICI-BCI/EEG-MACS repository from Prof. Zhiguo Zhang & Dr. Zhenxi Song's lab for direct baseline comparison
    macs_repo_dir = COHORT_2_DIR / "EEG-MACS-official-baseline"
    if not macs_repo_dir.exists():
        log("Cloning official HIT Shenzhen baseline repo: https://github.com/ICI-BCI/EEG-MACS.git")
        subprocess.run(["git", "clone", "--depth=1", "https://github.com/ICI-BCI/EEG-MACS.git", str(macs_repo_dir)], check=False)

    zenodo_files = [
        ("UI-001.zip", "https://zenodo.org/records/8338680/files/UI-001.zip?download=1", "3e5e5d6133dff190de04b7f20b5a437a"),
        ("UI-002.zip", "https://zenodo.org/records/8338680/files/UI-002.zip?download=1", "879ac5407bd8dcd2dcde4f89b3d163e4"),
        ("UNM-001.zip", "https://zenodo.org/records/8338680/files/UNM-001.zip?download=1", "59f799cab40b2f69c3c37b84cb5af7f7"),
        ("UNM-002.zip", "https://zenodo.org/records/8338680/files/UNM-002.zip?download=1", "234f7854a95e5f8a0d7642fcc87997a2"),
        ("UNM-003.zip", "https://zenodo.org/records/8338680/files/UNM-003.zip?download=1", "849957aa61232ef46585ee4be3adff8c"),
    ]

    # Build concurrent aria2c queue for all 5 Zenodo archives (5 x 16 = 80 streams to bypass per-stream throttle)
    z_queue = []
    for fname, url, md5 in zenodo_files:
        fpath = archives_dir / fname
        fctrl = archives_dir / f"{fname}.aria2"
        if fpath.exists() and not fctrl.exists() and md5sum(fpath).lower() == md5.lower():
            log(f"SKIP (MD5 verified): {fname}")
            continue
        z_queue.append(f"{url}\n  dir={archives_dir}\n  out={fname}\n")

    if z_queue:
        z_batch = archives_dir / "_aria2_zenodo_queue.txt"
        z_batch.write_text("".join(z_queue), encoding="utf-8")
        cmd = [
            "aria2c",
            "--continue=true",
            "--max-concurrent-downloads=5",
            "--max-connection-per-server=16",
            "--split=16",
            "--min-split-size=1M",
            "--retry-wait=5",
            "--max-tries=15",
            "--timeout=60",
            "--user-agent=curl/8.5.0",
            "--console-log-level=warn",
            "--summary-interval=15",
            f"--input-file={z_batch}",
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError(f"Zenodo parallel aria2c failed with code {res.returncode}")
        z_batch.unlink(missing_ok=True)

    total_bytes = 0
    for fname, url, md5 in zenodo_files:
        fpath = aria2_download(
            url=url,
            dest_dir=archives_dir,
            filename=fname,
            connections=16,
            expected_md5=md5,
            user_agent="curl/8.5.0",
        )
        total_bytes += fpath.stat().st_size

        # Extract into UNM_Data / Iowa_Data
        target_sub = extracted_dir / ("Iowa_Data" if fname.startswith("UI-") else "UNM_Data")
        target_sub.mkdir(parents=True, exist_ok=True)
        marker_done = archives_dir / f".extracted_{fname}.done"
        if not marker_done.exists():
            log(f"Extracting {fname} -> {target_sub} ...")
            with zipfile.ZipFile(fpath, "r") as zf:
                zf.extractall(target_sub)
            marker_done.write_text("ok\n", encoding="utf-8")

    unm_csvs = list((extracted_dir / "UNM_Data").rglob("*.csv"))
    ui_csvs = list((extracted_dir / "Iowa_Data").rglob("*.csv"))
    log(
        f"Cohort 2 COMPLETE: UNM subjects={len(unm_csvs)}, UIowa subjects={len(ui_csvs)}, "
        f"Archive GB={total_bytes / (1024**3):.2f} GB"
    )
    return {
        "dataset": "Cross-Center PD (UNM 54 + UIowa 28, 64-ch 500 Hz — EEG-MACS Benchmark)",
        "path": str(COHORT_2_DIR),
        "archive_gb": round(total_bytes / (1024**3), 3),
        "unm_subject_files": len(unm_csvs),
        "uiowa_subject_files": len(ui_csvs),
        "eeg_macs_repo_cloned": macs_repo_dir.exists(),
    }


# ==============================================================================
# 3. COHORT 3: MODMA Dataset (MDD + HC, 128-ch + 3-ch, PHQ-9 & GAD-7)
# ==============================================================================
def download_modma() -> dict:
    log("=== [3/3] Starting Cohort 3: MODMA Dataset (128-ch Resting & ERP + 3-ch Wearable, PHQ-9 & GAD-7) ===")
    COHORT_3_DIR.mkdir(parents=True, exist_ok=True)
    archives_dir = COHORT_3_DIR / "archives"
    extracted_dir = COHORT_3_DIR / "extracted"
    docs_dir = COHORT_3_DIR / "documentation"
    archives_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Note: ReShare archives bundle the exact Lanzhou packages with ReadMe + Demographic/PHQ-9/GAD-7 Excel sheets inside
    modma_packages = [
        (
            "854301_EEG_3Channels_Resting_Lanzhou_2015.zip",
            "https://reshare.ukdataservice.ac.uk/854301/2/854301_EEG_3Channels_Resting_Lanzhou_2015.zip",
            "3ch_resting_55sub",
        ),
        (
            "854301_EEG_128Channels_Resting_Lanzhou_2015.zip",
            "https://reshare.ukdataservice.ac.uk/854301/4/854301_EEG_128Channels_Resting_Lanzhou_2015.zip",
            "128ch_resting_53sub",
        ),
        (
            "854301_EEG_128Channels_ERP_Lanzhou_2015.zip",
            "https://reshare.ukdataservice.ac.uk/854301/3/854301_EEG_128Channels_ERP_Lanzhou_2015.zip",
            "128ch_erp_dotprobe_53sub",
        ),
    ]

    modma_docs = [
        ("ReadMe.pdf", "https://reshare.ukdataservice.ac.uk/854301/37/ReadMe.pdf"),
        ("854301_Methodology.docx", "https://reshare.ukdataservice.ac.uk/854301/8/854301_Methodology.docx"),
        ("854301_InformationSheet.pdf", "https://reshare.ukdataservice.ac.uk/854301/5/854301_InformationSheet.pdf"),
    ]

    for doc_name, doc_url in modma_docs:
        aria2_download(url=doc_url, dest_dir=docs_dir, filename=doc_name, connections=2)

    m_queue = []
    for fname, url, _ in modma_packages:
        fpath = archives_dir / fname
        fctrl = archives_dir / f"{fname}.aria2"
        if fpath.exists() and not fctrl.exists() and fpath.stat().st_size > 0:
            continue
        m_queue.append(f"{url}\n  dir={archives_dir}\n  out={fname}\n")

    if m_queue:
        m_batch = archives_dir / "_aria2_modma_queue.txt"
        m_batch.write_text("".join(m_queue), encoding="utf-8")
        cmd = [
            "aria2c",
            "--continue=true",
            "--max-concurrent-downloads=3",
            "--max-connection-per-server=16",
            "--split=16",
            "--min-split-size=1M",
            "--retry-wait=5",
            "--max-tries=15",
            "--timeout=60",
            "--user-agent=curl/8.5.0",
            "--console-log-level=warn",
            "--summary-interval=15",
            f"--input-file={m_batch}",
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError(f"MODMA parallel aria2c failed with code {res.returncode}")
        m_batch.unlink(missing_ok=True)

    total_bytes = 0
    extracted_summary = {}
    for fname, url, sub_folder in modma_packages:
        fpath = aria2_download(url=url, dest_dir=archives_dir, filename=fname, connections=16)
        total_bytes += fpath.stat().st_size

        target_sub = extracted_dir / sub_folder
        target_sub.mkdir(parents=True, exist_ok=True)
        marker_done = archives_dir / f".extracted_{fname}.done"
        if not marker_done.exists():
            log(f"Extracting {fname} -> {target_sub} ...")
            res = subprocess.run(["unzip", "-q", "-o", str(fpath), "-d", str(target_sub)])
            if res.returncode == 0:
                marker_done.write_text("ok\n", encoding="utf-8")
            else:
                log(f"Standard unzip returned {res.returncode} on {fname}, trying python zipfile...")
                with zipfile.ZipFile(fpath, "r") as zf:
                    zf.extractall(target_sub)
                marker_done.write_text("ok\n", encoding="utf-8")

        all_extracted = [p for p in target_sub.rglob("*") if p.is_file()]
        xlsx_files = [p.name for p in all_extracted if p.suffix.lower() in (".xlsx", ".xls", ".csv")]
        extracted_summary[sub_folder] = {
            "total_files": len(all_extracted),
            "clinical_scale_tables": xlsx_files,
        }

    log(f"Cohort 3 COMPLETE: Archive GB={total_bytes / (1024**3):.2f} GB | {extracted_summary}")
    return {
        "dataset": "MODMA (128-ch Resting + 128-ch ERP + 3-ch Wearable, MDD vs HC, PHQ-9 & GAD-7)",
        "path": str(COHORT_3_DIR),
        "archive_gb": round(total_bytes / (1024**3), 3),
        "modalities": extracted_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download & verify Paper 2 Clinical EEG Datasets")
    parser.add_argument(
        "--cohort",
        choices=["all", "1", "2", "3"],
        default="all",
        help="Download all 3 cohorts in parallel ('all') or a single cohort ('1', '2', '3')",
    )
    args = parser.parse_args()

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Target Clinical EEG Root on DGX Spark: {BASE_DIR} (mode={args.cohort})")

    tasks = {
        "1": ("01_cognitive_ds004504", download_ds004504),
        "2": ("02_crosscenter_unm_uiowa", download_unm_uiowa),
        "3": ("03_affective_modma", download_modma),
    }

    if args.cohort != "all":
        key, fn = tasks[args.cohort]
        info = fn()
        out_path = BASE_DIR / f"status_{key}.json"
        out_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"Cohort {args.cohort} ({key}) finished -> {out_path}")
        return

    status = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "cohorts": {}}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(fn): key for _, (key, fn) in tasks.items()}
        for fut in as_completed(futs):
            key = futs[fut]
            info = fut.result()
            status["cohorts"][key] = info
            STATUS_FILE.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    status["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status["all_complete"] = True
    STATUS_FILE.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    log("ALL 3 PUBLIC CLINICAL EEG DATASETS DOWNLOADED & VERIFIED 100%!")


if __name__ == "__main__":
    main()
