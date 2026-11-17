"""
Automated Open-Science OSF Downloader for ZuCo 1.0 (osf.io/q3zws) and ZuCo 2.0 (osf.io/2urht).
Supports:
  - `--quick-subset 3`: Download only 3 subjects (~2.0 GB) from ZuCo 1.0 `task1-SR` for rapid real-EEG experiments!
  - Multi-connection `aria2c` acceleration with automatic resume (`-c -x 16 -s 16`) if `aria2c` is installed,
    falling back to resumable HTTP Range streaming via `requests`.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List

import requests
from tqdm import tqdm

OSF_API = "https://api.osf.io/v2/nodes"
ZUCO_PROJECTS = {
    "v1": {
        "node_id": "q3zws",
        "tasks": {
            "task1sr": "task1-SR",
            "task2nr": "task2-NR",
            "task3tsr": "task3-TSR",
        },
    },
    "v2": {
        "node_id": "2urht",
        "tasks": {
            "task1nr": "task2-NR-2.0",
            "task2tsr": "task3-TSR-2.0",
        },
    },
}


def norm_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def list_osf_folder(url: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    next_url: str | None = url
    while next_url:
        resp = requests.get(next_url, timeout=45)
        resp.raise_for_status()
        payload = resp.json()
        items.extend(payload.get("data", []))
        next_url = payload.get("links", {}).get("next")
    return items


def find_mat_files_in_matlab_subfolder(task_folder_url: str) -> List[Dict[str, Any]]:
    """Only descend into `Matlab files` (skips Raw data / Preprocessed so we never pull unwanted 100GB raw folders)."""
    found: List[Dict[str, Any]] = []
    for entry in list_osf_folder(task_folder_url):
        attrs = entry.get("attributes", {})
        kind = attrs.get("kind")
        name = attrs.get("name", "")
        if kind == "file" and name.endswith(".mat"):
            dl_url = entry.get("links", {}).get("download")
            if dl_url:
                found.append({"name": name, "url": dl_url, "size": int(attrs.get("size") or 0)})
        elif kind == "folder" and "matlab" in name.lower():
            sub_url = entry.get("relationships", {}).get("files", {}).get("links", {}).get("related", {}).get("href")
            if sub_url:
                for sub_entry in list_osf_folder(sub_url):
                    s_attrs = sub_entry.get("attributes", {})
                    if s_attrs.get("kind") == "file" and s_attrs.get("name", "").endswith(".mat"):
                        dl_url = sub_entry.get("links", {}).get("download")
                        if dl_url:
                            found.append(
                                {
                                    "name": s_attrs["name"],
                                    "url": dl_url,
                                    "size": int(s_attrs.get("size") or 0),
                                }
                            )
    return found


def download_file_resumable(url: str, dest_path: str, expected_size: int = 0) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        cur_size = os.path.getsize(dest_path)
        if expected_size > 0 and cur_size == expected_size:
            print(f"  [Complete] {dest_path} ({cur_size / 1e6:.1f} MB)")
            return
        if expected_size == 0 and cur_size > 10 * 1024 * 1024:
            print(f"  [Skip existing] {dest_path} ({cur_size / 1e6:.1f} MB)")
            return

    aria2 = shutil.which("aria2c")
    if aria2:
        # OSF & HuggingFace redirect to S3/OCI presigned URLs with `X-Amz-Expires=60`.
        # When a chunk reconnects after 60s, aria2c hits HTTP 403 (exit code 22) on the expired redirect URL.
        # Re-invoking aria2c with `--continue=true` against the original `url` fetches a fresh presigned URL
        # and resumes seamlessly from the `.aria2` control bitmap!
        for attempt in range(1, 100):
            if os.path.exists(dest_path) and not os.path.exists(dest_path + ".aria2"):
                if expected_size == 0 or os.path.getsize(dest_path) == expected_size:
                    print(f"  [Complete] {os.path.basename(dest_path)}")
                    return
            cmd = [
                aria2,
                "--continue=true",
                "--max-connection-per-server=8",
                "--split=8",
                "--min-split-size=5M",
                "--retry-wait=3",
                "--max-tries=5",
                "--summary-interval=15",
                "--console-log-level=warn",
                "--dir",
                os.path.dirname(dest_path),
                "--out",
                os.path.basename(dest_path),
                url,
            ]
            print(f"  [aria2c 8x | pass {attempt}] Downloading {os.path.basename(dest_path)} ({expected_size / 1e6:.1f} MB)...")
            res = subprocess.run(cmd, check=False)
            if res.returncode == 0 and os.path.exists(dest_path) and not os.path.exists(dest_path + ".aria2"):
                print(f"  [Complete] {os.path.basename(dest_path)}")
                return
            print(f"  [Presigned URL refreshed] aria2c exited with code {res.returncode}; resuming from .aria2 state...")
        raise RuntimeError(f"Failed to complete {dest_path} after 100 resumable passes.")

    tmp_path = dest_path + ".part"
    resume_pos = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}
    mode = "ab" if resume_pos > 0 else "wb"

    print(f"  [HTTP Stream] Downloading {os.path.basename(dest_path)} (resume={resume_pos / 1e6:.1f} MB / {expected_size / 1e6:.1f} MB)...")
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if r.status_code == 200 and resume_pos > 0:
            # Server ignored Range; restart cleanly
            resume_pos = 0
            mode = "wb"
        r.raise_for_status()
        total = expected_size or (int(r.headers.get("content-length", 0)) + resume_pos)
        with open(tmp_path, mode) as f, tqdm(
            total=total, initial=resume_pos, unit="B", unit_scale=True, desc=os.path.basename(dest_path)
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    os.replace(tmp_path, dest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ZuCo 1.0 / 2.0 `.mat` files directly from OSF.")
    parser.add_argument("--output-root", type=str, default="./datasets/zuco_raw")
    parser.add_argument(
        "--quick-subset",
        type=int,
        default=0,
        help="If > 0 (e.g. 3), download the N smallest `.mat` subject files from ZuCo v1 task1-SR (~1.6 GB) for fast real-EEG prototyping.",
    )
    parser.add_argument("--include-v2", action="store_true", help="Also download ZuCo 2.0 (62 GB HDF5 MAT files)")
    args = parser.parse_args()

    v1_root_url = f"{OSF_API}/{ZUCO_PROJECTS['v1']['node_id']}/files/osfstorage/"
    print(f"[OSF] Querying ZuCo 1.0 storage ({v1_root_url})...")
    for item in list_osf_folder(v1_root_url):
        attrs = item.get("attributes", {})
        key = norm_key(attrs.get("name", ""))
        if attrs.get("kind") == "folder" and key in ZUCO_PROJECTS["v1"]["tasks"]:
            local_task = ZUCO_PROJECTS["v1"]["tasks"][key]
            if args.quick_subset > 0 and local_task != "task1-SR":
                continue
            sub_url = item["relationships"]["files"]["links"]["related"]["href"]
            print(f"[OSF] Scanning `Matlab files` in ZuCo 1.0 `{attrs.get('name')}` -> `{local_task}`...")
            mat_files = find_mat_files_in_matlab_subfolder(sub_url)
            if args.quick_subset > 0:
                # Pick the 3 smallest complete subject files first so real-EEG training starts as fast as possible!
                mat_files = sorted(mat_files, key=lambda x: x["size"] or 10**12)[: args.quick_subset]
            else:
                mat_files = sorted(mat_files, key=lambda x: x["name"])
            for mf in mat_files:
                dest = os.path.join(args.output_root, local_task, "Matlab_files", mf["name"])
                download_file_resumable(mf["url"], dest, expected_size=mf["size"])

    if args.include_v2 and args.quick_subset == 0:
        v2_root_url = f"{OSF_API}/{ZUCO_PROJECTS['v2']['node_id']}/files/osfstorage/"
        print(f"[OSF] Querying ZuCo 2.0 storage ({v2_root_url})...")
        for item in list_osf_folder(v2_root_url):
            attrs = item.get("attributes", {})
            key = norm_key(attrs.get("name", ""))
            if attrs.get("kind") == "folder" and key in ZUCO_PROJECTS["v2"]["tasks"]:
                local_task = ZUCO_PROJECTS["v2"]["tasks"][key]
                sub_url = item["relationships"]["files"]["links"]["related"]["href"]
                mat_files = sorted(find_mat_files_in_matlab_subfolder(sub_url), key=lambda x: x["name"])
                for mf in mat_files:
                    dest = os.path.join(args.output_root, local_task, "Matlab_files", mf["name"])
                    download_file_resumable(mf["url"], dest, expected_size=mf["size"])

    print("[OSF Download Complete] Next step: `python -m src.data.zuco_preprocess`")


if __name__ == "__main__":
    main()
