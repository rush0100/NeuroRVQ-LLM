#!/usr/bin/env python3
"""
Summarizer & IEEE LaTeX Row Generator for the 4-Point Q1 Reviewer Defense Suite.
Run locally or on DGX Spark whenever `bci_q1_defense_master` finishes any block:
  python3 scripts/summarize_q1_defense_results.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "logs"

FILES = {
    "Action 4 (Stage 1 Per-Subject Breakdown)": ROOT / "rush_eeg2text_full4task_champion_qwen7b" / "stage1_per_subject_eval.json",
    "Action 4 (Stage 2 Per-Subject & Multi-Sub Examples)": ROOT / "rush_eeg2text_full4task_champion_qwen7b" / "test_evaluation_report_per_subject.json",
    "Action 1a (EEG-Only, Zero Oculomotor Metadata)": ROOT / "q1_action1a_eeg_only_no_oculomotor" / "stage1_summary.json",
    "Action 1b (100% Fixation-Free Holistic Locked-In Mode)": ROOT / "q1_action1b_fixation_free_holistic" / "stage1_summary.json",
    "Action 3 (Fair Backbone: RUSH-EEG2Text + BART-large Single-Task)": ROOT / "q1_action3_rush_eeg2text_bart_large_singletask" / "test_evaluation_report.json",
    "Action 2 (Double Zero-Shot: Unseen Subj + Unseen Sent S1)": ROOT / "q1_action2_double_zeroshot" / "stage1_summary.json",
    "Action 2 (Double Zero-Shot: Unseen Subj + Unseen Sent S2)": ROOT / "q1_action2_double_zeroshot" / "test_evaluation_report.json",
}


def main() -> None:
    print("=" * 88)
    print("PAPER 1 (`RUSH-EEG2Text`) — Q1 REVIEWER DEFENSE SUITE STATUS & METRICS")
    print("=" * 88)
    for title, p in FILES.items():
        if not p.exists():
            print(f"[RUNNING / QUEUED] {title} -> {p.relative_to(ROOT.parent)}")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        print(f"\n[COMPLETED] {title}")
        if "per_subject" in data and "mean_std" in data:
            print("  Mean +/- SD:", data["mean_std"])
            for s, m in data["per_subject"].items():
                print(f"    {s} ({m['corpus']}, N={m['n_trials']}): R@10={m['real_eeg']['R@10']}% MRR={m['real_eeg']['MRR']}% (Delta MRR=+{m['sanity_gap_mrr']}%)")
        elif "per_subject_summary_table" in data:
            print("  Overall Real EEG:", data["real_eeg_tf_free_metrics"])
            print("  Mean +/- SD:", data.get("per_subject_mean_std"))
            for s, m in data["per_subject_summary_table"].items():
                print(f"    {s} (N={m['n_trials']}): BLEU-1={m['BLEU-1']}% BLEU-4={m['BLEU-4']}% ROUGE-1={m['ROUGE-1_F1']}% SemSim={m['Semantic_CosSim']}% (Delta B4=+{m['Delta_BLEU-4']}%)")
        elif "test_unseen_subject_real_eeg" in data:
            r = data["test_unseen_subject_real_eeg"]["post_rvq"]
            print(f"  Test Real EEG: R@1={r['R@1']}% R@5={r['R@5']}% R@10={r['R@10']}% MRR={r['MRR']}% (Sanity Gap=+{data['test_sanity_gap_mrr']}%)")
            if data.get("unseen_sentence_pool_eval"):
                u = data["unseen_sentence_pool_eval"]
                print(f"  Unseen Sentence Pool ({u['candidate_pool_size']} sents): R@10={u['real_eeg']['R@10']}% MRR={u['real_eeg']['MRR']}% SentAcc={u['sentiment_acc']}%")
        elif "real_eeg_tf_free_metrics" in data:
            r = data["real_eeg_tf_free_metrics"]
            print(f"  Stage 2 Real EEG: BLEU-1={r['BLEU-1']}% BLEU-4={r['BLEU-4']}% ROUGE-1={r['ROUGE-1_F1']}% SemSim={r.get('Semantic_CosSim')}% (Delta B1=+{data.get('sanity_gap_bleu1')}%, Delta B4=+{data.get('sanity_gap_bleu4')}%)")


if __name__ == "__main__":
    main()
