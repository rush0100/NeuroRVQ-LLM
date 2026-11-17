"""
One-Command Exporter & Uploader to Hugging Face Model Hub (`rush0100/RUSH-EEG2Text-Qwen2.5-7B`).

Run this AFTER `train_stage2_llm.py` finishes its final Test + Gaussian Noise Sanity Evaluation:
  HF_TOKEN="hf_..." python3 scripts/publish_to_huggingface.py --repo-id rush0100/RUSH-EEG2Text-Qwen2.5-7B

Uploads:
  1. Auto-generated IEEE/HF Model Card (`README.md`) populated with exact Stage 1 & Stage 2 Test + Noise Sanity metrics
  2. `stage1_cet_rvq_best.pt` (3D 12-ROI + Late Connectome Bottleneck + 6-Book EMA-RVQ Pretrained Tokenizer)
  3. `stage2_cet_rvq_best.pt` & `stage2_projector_best.pt` (Jointly fine-tuned Stage 2 Encoder & MLP Bridge)
  4. `stage2_lora_adapter/` (`adapter_config.json` + `adapter_model.safetensors` for `Qwen/Qwen2.5-7B-Instruct`)
  5. `config.yaml`, `stage1_summary.json`, `stage2_history.json`, and `test_evaluation_report.json`
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import tempfile
from typing import Any, Dict

from huggingface_hub import HfApi, create_repo

try:
    import urllib3.util.connection as urllib3_cn
    urllib3_cn.allowed_gai_family = lambda: socket.AF_INET
except ImportError:
    pass


def build_model_card(
    repo_id: str,
    s1_summary: Dict[str, Any],
    test_report: Dict[str, Any],
) -> str:
    s1_test = s1_summary["test_unseen_subject_real_eeg"]["post_rvq"]
    s1_noise = s1_summary["test_unseen_subject_gaussian_noise"]["post_rvq"]
    s2_real = test_report["real_eeg_tf_free_metrics"]
    s2_noise = test_report.get("gaussian_noise_metrics", {})
    gap_b1 = test_report.get("sanity_gap_bleu1", 0.0)
    gap_b4 = test_report.get("sanity_gap_bleu4", 0.0)

    examples_md = ""
    for ex in test_report.get("decoded_examples", [])[:5]:
        examples_md += (
            f"| `{ex['subject']}` | {ex['reference']} | {ex['decoded']} |\n"
        )

    return f"""---
license: mit
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
tags:
  - brain-computer-interface
  - eeg
  - eeg-to-text
  - neural-decoding
  - residual-vector-quantization
  - riemannian-geometry
  - qwen2.5
  - lora
datasets:
  - zuco-1.0
  - zuco-2.0
language:
  - en
pipeline_tag: text-generation
---

# RUSH-EEG2Text-Qwen2.5-7B

**Riemannian Connectome-Guided Hierarchical Quantization for Teacher-Forcing-Free Cross-Subject EEG-to-Text Decoding**

- **Project Status:** Pre-Submission Working Draft & Academic Collaboration Proposal (KMITL Overseas Research Mobility Grant)
- **GitHub Repository (Code, Preprocessing, & LaTeX Draft):** [github.com/rush0100/RUSH-EEG2Text](https://github.com/rush0100/RUSH-EEG2Text)
- **Conceptual Video Explainer:** [Watch on GitHub](https://github.com/rush0100/RUSH-EEG2Text/raw/main/assets/rush-eeg2text-explained-simply.mp4)
- **Primary Authors:** Ratchaphon Khaemphukhiao (Ph.D. Candidate, KMITL) & Assoc. Prof. Pattarapong Phasukkit, D.Eng. (Principal Advisor, KMITL)
- **Invited Co-Author Slots (Open for Joint Collaboration):**
  - `[Invited Co-Author Slot: Dr. Zhenxi Song, Harbin Institute of Technology, Shenzhen]`
  - `[Invited Co-Author & Co-Corresponding Author Slot: Prof. Zhiguo Zhang, Harbin Institute of Technology, Shenzhen]`
- **Base Causal LLM:** `Qwen/Qwen2.5-7B-Instruct` (adapted via Rank-16 BF16 LoRA + 50% Scheduled Target-Word Dropout)
- **Training Corpus:** All 4 Naturalistic Reading Tasks of **ZuCo 1.0 & ZuCo 2.0** (`task1-SR`, `task2-NR`, `task3-TSR`, `task2-NR-2.0` — **18,276 trials across 30 unique subjects**)
- **Evaluation Protocol:** Strictly **Subject-Independent** (4 completely unseen test subjects: `ZMG`, `ZPH`, `YTL`, `YSL` — 2,717 test trials) & **100% Teacher-Forcing-Free** Beam Search (`num_beams=4`) verified against a Pure Gaussian Noise ($\\mathcal{{N}}(0, \\mathbf{{I}})$) Sanity Audit.

---

## 1. Benchmark Results on Unseen Human Test Subjects

### Stage 1: Zero-Shot 1,313-Sentence Retrieval (`ZMG`, `ZPH`, `YTL`, `YSL`)
| Input Condition | R@1 (%) | R@5 (%) | R@10 (%) | MRR (%) | Median Rank (out of 1,313) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Real Unseen-Subject EEG** | **{s1_test['R@1']:.2f}%** | **{s1_test['R@5']:.2f}%** | **{s1_test['R@10']:.2f}%** | **{s1_test['MRR']:.2f}%** | **{s1_test['MedR']:.1f}** |
| Pure Gaussian Noise ($\\mathcal{{N}}(0, \\mathbf{{I}})$) | {s1_noise['R@1']:.2f}% | {s1_noise['R@5']:.2f}% | {s1_noise['R@10']:.2f}% | {s1_noise['MRR']:.2f}% | {s1_noise['MedR']:.1f} |
| **Neural Sanity Gap ($\\Delta$)** | **+{s1_test['R@1'] - s1_noise['R@1']:.2f}%** | **+{s1_test['R@5'] - s1_noise['R@5']:.2f}%** | **+{s1_summary['test_sanity_gap_r10']:.2f}%** | **+{s1_summary['test_sanity_gap_mrr']:.2f}%** | — |

### Stage 2: Strict Teacher-Forcing-Free Open-Vocabulary Generation (`Qwen2.5-7B-Instruct` LoRA)
| Input Condition | BLEU-1 (%) | BLEU-2 (%) | BLEU-3 (%) | BLEU-4 (%) | ROUGE-1 F1 (%) | WER (%) $\\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Real Unseen-Subject EEG (TF-Free)** | **{s2_real['BLEU-1']:.2f}%** | **{s2_real['BLEU-2']:.2f}%** | **{s2_real['BLEU-3']:.2f}%** | **{s2_real['BLEU-4']:.2f}%** | **{s2_real['ROUGE-1_F1']:.2f}%** | **{s2_real['WER']:.2f}%** |
| Pure Gaussian Noise ($\\mathcal{{N}}(0, \\mathbf{{I}})$) | {s2_noise.get('BLEU-1', 0.0):.2f}% | {s2_noise.get('BLEU-2', 0.0):.2f}% | {s2_noise.get('BLEU-3', 0.0):.2f}% | {s2_noise.get('BLEU-4', 0.0):.2f}% | {s2_noise.get('ROUGE-1_F1', 0.0):.2f}% | {s2_noise.get('WER', 0.0):.2f}% |
| **Neural Sanity Gap ($\\Delta$)** | **+{gap_b1:.2f}%** | — | — | **+{gap_b4:.2f}%** | **+{s2_real['ROUGE-1_F1'] - s2_noise.get('ROUGE-1_F1', 0.0):.2f}%** | — |

---

## 2. Included Checkpoints & Files

- `stage1_cet_rvq_best.pt`: Pretrained Stage 1 3D 12-ROI + Late Connectome Bottleneck + 6-Book Euclidean EMA-RVQ checkpoint.
- `stage2_cet_rvq_best.pt`: End-to-end Stage 2 fine-tuned neural encoder + RVQ weights.
- `stage2_projector_best.pt`: 2-layer `BrainToLLMProjector` (`768 -> 3584`) mapping discrete brain tokens into `Qwen2.5-7B` embedding space.
- `stage2_lora_adapter/`: PEFT LoRA adapter (`r=16, alpha=32`) for `Qwen/Qwen2.5-7B-Instruct`.
- `spark_full4task_champion_qwen7b.yaml`: Exact training & evaluation configuration.
- `stage1_summary.json`, `stage2_history.json`, `test_evaluation_report.json`: Full machine-readable benchmark reports.

## 3. Sample Unseen-Subject Teacher-Forcing-Free Outputs

| Unseen Subject | Ground-Truth Sentence | RUSH-EEG2Text Decoded Output (Zero Teacher Forcing) |
| :---: | :--- | :--- |
{examples_md}

## Acknowledgment
This work is supported by King Mongkut's Institute of Technology Ladkrabang.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish completed RUSH-EEG2Text checkpoints & Model Card to Hugging Face Hub.")
    parser.add_argument("--repo-id", type=str, default="RUSH0100/RUSH-EEG2Text-Qwen2.5-7B")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints/rush_eeg2text_full4task_champion_qwen7b")
    parser.add_argument("--log-dir", type=str, default="./logs/rush_eeg2text_full4task_champion_qwen7b")
    parser.add_argument("--config-path", type=str, default="./configs/spark_full4task_champion_qwen7b.yaml")
    parser.add_argument("--private", action="store_true", help="Create private model repo instead of public")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        token_file = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_file):
            with open(token_file, "r", encoding="utf-8") as tf:
                token = tf.read().strip()
    if not token:
        raise RuntimeError("Please set `HF_TOKEN=hf_...` or store token in `~/.cache/huggingface/token`.")

    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir

    report_path = os.path.join(log_dir, "test_evaluation_report.json")
    s1_path = os.path.join(log_dir, "stage1_summary.json")
    if not os.path.exists(report_path):
        raise FileNotFoundError(
            f"Final Stage 2 report `{report_path}` not found yet. Wait for Stage 2 training & noise sanity evaluation to finish!"
        )

    with open(s1_path, "r", encoding="utf-8") as f:
        s1_summary = json.load(f)
    with open(report_path, "r", encoding="utf-8") as f:
        test_report = json.load(f)

    api = HfApi(token=token)
    create_repo(repo_id=args.repo_id, token=token, repo_type="model", private=args.private, exist_ok=True)

    with tempfile.TemporaryDirectory() as staging:
        # 1. Write rich Model Card
        readme_md = build_model_card(args.repo_id, s1_summary, test_report)
        with open(os.path.join(staging, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_md)

        # 2. Copy config & JSON logs
        shutil.copy2(args.config_path, os.path.join(staging, os.path.basename(args.config_path)))
        for fn in ("stage1_summary.json", "stage2_history.json", "test_evaluation_report.json"):
            src = os.path.join(log_dir, fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(staging, fn))

        # 3. Copy checkpoints & PEFT LoRA adapter
        for pt_name in ("stage1_cet_rvq_best.pt", "stage2_cet_rvq_best.pt", "stage2_projector_best.pt"):
            src = os.path.join(ckpt_dir, pt_name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(staging, pt_name))

        lora_src = os.path.join(ckpt_dir, "stage2_lora_adapter")
        if os.path.isdir(lora_src):
            shutil.copytree(lora_src, os.path.join(staging, "stage2_lora_adapter"))
            # Remove auto-generated blank PEFT README inside subfolder if present
            sub_readme = os.path.join(staging, "stage2_lora_adapter", "README.md")
            if os.path.exists(sub_readme):
                os.remove(sub_readme)

        print(f"[HuggingFace] Uploading bundle to https://huggingface.co/{args.repo_id} ...")
        api.upload_folder(
            folder_path=staging,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Release RUSH-EEG2Text-Qwen2.5-7B (Full 4-Task ZuCo 30-Subject Checkpoints & Sanity Audit)",
        )

    print(f"[Success] Published public model repository -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
