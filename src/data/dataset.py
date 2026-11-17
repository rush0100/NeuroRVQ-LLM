"""Dataset splits & Dual-Stream + 8-Metadata Collation for ZuCo 1.0 & ZuCo 2.0."""

from __future__ import annotations

import hashlib
import os
import pickle
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from src.data.neuro_connectome import COG_LOAD_DIM, NODE_FEAT_DIM, SENT_CONNECTOME_DIM, TASK_NAME_TO_ID, WORD_EDGE_DIM


class ZuCoEEGTextDataset(Dataset):
    """
    PyTorch Dataset yielding Dual-Stream Neuro-Connectome + 8-Metadata tensors.
    Supports `eeg_only_mode` for Q1 Clinical Applicability (Locked-In / Eye-Tracking-Free) ablations:
      - `"none"`: Full NeuroRVQ-LLM mode
      - `"eeg_only_no_oculomotor"`: Zeros out all pupil/fixation/saccade metadata & scanpath re-ordering
      - `"fixation_free_holistic"`: Pure whole-sentence continuous EEG + 500-Hz Connectome ONLY (`seq_len=1`, zero word fixations!)
    """

    def __init__(self, samples: List[Dict[str, Any]], eeg_only_mode: str = "none") -> None:
        self.samples = samples
        self.eeg_only_mode = eeg_only_mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        eeg_feat = item["eeg_features"].float().clone()
        attn_mask = item["eeg_attn_mask"].long().clone()
        max_len = int(eeg_feat.size(0))
        seq_len = int(item["seq_len"])

        if "node_features" in item:
            node_feat = item["node_features"].float().clone()
            word_edges = item["word_coupled_edges"].float().clone()
            sent_conn = item["sent_connectome"].float().clone()
            rel_pos = item["rel_pos"].float().clone()
            lex_pos = item["lex_pos"].float().clone()
            salience = item["word_salience"].float().clone()
            cog_load = item["cog_load"].float().clone()
            task_id = int(item.get("task_id", TASK_NAME_TO_ID.get(item["task_name"], 1)))
            sent_lbl = int(item.get("sentiment_label", -100))
            rel_lbl = int(item.get("relation_label", -100))
        else:
            # Automatic fallback if loading an older legacy 840-d pickle
            node_feat = torch.zeros((max_len, NODE_FEAT_DIM), dtype=torch.float32)
            word_edges = torch.zeros((max_len, WORD_EDGE_DIM), dtype=torch.float32)
            sent_conn = torch.zeros((SENT_CONNECTOME_DIM,), dtype=torch.float32)
            rel_pos = torch.zeros((max_len,), dtype=torch.float32)
            if seq_len > 0:
                rel_pos[:seq_len] = torch.linspace(0.0, 1.0, steps=seq_len)
            lex_pos = rel_pos.clone()
            salience = attn_mask.float()
            cog_load = torch.zeros((COG_LOAD_DIM,), dtype=torch.float32)
            task_id = TASK_NAME_TO_ID.get(item["task_name"], 1)
            sent_lbl = -100
            rel_lbl = -100

        if self.eeg_only_mode == "eeg_only_no_oculomotor":
            # Strip all 8-d word oculomotor/pupil features & 6-d sentence cognitive-load features;
            # replace scanpath order & fixation salience with uniform non-oculomotor progression
            node_feat[:, 480:] = 0.0
            cog_load.zero_()
            rel_pos.zero_()
            if seq_len > 1:
                rel_pos[: seq_len - 1] = torch.linspace(0.0, 0.92, steps=seq_len - 1)
                rel_pos[seq_len - 1] = 1.0
            elif seq_len == 1:
                rel_pos[0] = 1.0
            lex_pos = rel_pos.clone()
            salience = attn_mask.float()

        elif self.eeg_only_mode == "fixation_free_holistic":
            # Strictly Fixation-Free Locked-In Mode: discard ALL eye-tracked word segments!
            # Keep ONLY the holistic sentence-level EEG slot (`seq_len - 1`) + continuous 500-Hz connectome!
            hol_idx = max(0, seq_len - 1)
            hol_node = node_feat[hol_idx].clone()
            hol_node[384 - 96 : 384] = 0.0  # zero out TRT-FFD fixation reanalysis delta
            hol_node[480:] = 0.0            # zero out all 8-d oculomotor/pupil metadata
            hol_edge = word_edges[hol_idx].clone()
            hol_leg = eeg_feat[hol_idx].clone()

            node_feat.zero_()
            word_edges.zero_()
            eeg_feat.zero_()
            attn_mask.zero_()
            rel_pos.zero_()
            lex_pos.zero_()
            salience.zero_()
            cog_load.zero_()

            node_feat[0] = hol_node
            word_edges[0] = hol_edge
            eeg_feat[0] = hol_leg
            attn_mask[0] = 1
            rel_pos[0] = 1.0
            lex_pos[0] = 1.0
            salience[0] = 1.0
            seq_len = 1

        return {
            "eeg_features": eeg_feat,
            "node_features": node_feat,
            "word_coupled_edges": word_edges,
            "sent_connectome": sent_conn,
            "eeg_attn_mask": attn_mask,
            "rel_pos": rel_pos,
            "lex_pos": lex_pos,
            "word_salience": salience,
            "cog_load": cog_load,
            "task_id": task_id,
            "sentiment_label": sent_lbl,
            "relation_label": rel_lbl,
            "target_text": item["target_text"],
            "subject_id": item["subject_id"],
            "task_name": item["task_name"],
            "seq_len": seq_len,
        }


def collate_eeg_text(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "eeg_features": torch.stack([b["eeg_features"] for b in batch], dim=0),
        "node_features": torch.stack([b["node_features"] for b in batch], dim=0),
        "word_coupled_edges": torch.stack([b["word_coupled_edges"] for b in batch], dim=0),
        "sent_connectome": torch.stack([b["sent_connectome"] for b in batch], dim=0),
        "eeg_attn_mask": torch.stack([b["eeg_attn_mask"] for b in batch], dim=0),
        "rel_pos": torch.stack([b["rel_pos"] for b in batch], dim=0),
        "lex_pos": torch.stack([b["lex_pos"] for b in batch], dim=0),
        "word_salience": torch.stack([b["word_salience"] for b in batch], dim=0),
        "cog_load": torch.stack([b["cog_load"] for b in batch], dim=0),
        "task_id": torch.tensor([b["task_id"] for b in batch], dtype=torch.long),
        "sentiment_label": torch.tensor([b["sentiment_label"] for b in batch], dtype=torch.long),
        "relation_label": torch.tensor([b["relation_label"] for b in batch], dtype=torch.long),
        "target_text": [b["target_text"] for b in batch],
        "subject_id": [b["subject_id"] for b in batch],
        "task_name": [b["task_name"] for b in batch],
        "seq_len": torch.tensor([b["seq_len"] for b in batch], dtype=torch.long),
    }


def load_and_split_zuco(
    dataset_root: str,
    tasks: List[str],
    split_strategy: str = "subject_independent",
    held_out_test_subjects: List[str] | None = None,
    held_out_val_subjects: List[str] | None = None,
    eeg_only_mode: str = "none",
) -> Tuple[ZuCoEEGTextDataset, ZuCoEEGTextDataset, ZuCoEEGTextDataset]:
    """Load preprocessed `.pkl` files and partition into train/val/test without data leakage."""
    all_samples: List[Dict[str, Any]] = []
    for task in tasks:
        pkl_path = os.path.join(dataset_root, f"{task}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"Missing preprocessed task file: {pkl_path}. "
                f"Run `python -m src.data.zuco_preprocess` first."
            )
        with open(pkl_path, "rb") as f:
            all_samples.extend(pickle.load(f))

    if not all_samples:
        raise RuntimeError(f"Loaded 0 samples from {dataset_root} for tasks={tasks}.")

    train_list: List[Dict[str, Any]] = []
    val_list: List[Dict[str, Any]] = []
    test_list: List[Dict[str, Any]] = []

    if split_strategy == "unseen_subject_and_sentence":
        # Strict Double Zero-Shot (Action 2): BOTH human subjects AND sentence pools are 100% mutually disjoint!
        # - Train: train_subjects reading 80% train sentences (`h < 8`)
        # - Val:   val_subjects   reading 10% unseen val sentences (`h == 8`)
        # - Test:  test_subjects  reading 10% unseen test sentences (`h == 9`) (or 20% `h >= 8` when val_subs != test_subs & separate)
        test_subs = set(held_out_test_subjects or [])
        val_subs = set(held_out_val_subjects or [])
        available_subs = sorted({s["subject_id"] for s in all_samples})
        if not (test_subs & set(available_subs)) and len(available_subs) >= 3:
            test_subs = {available_subs[-1]}
            val_subs = {available_subs[-2]}

        for s in all_samples:
            norm_txt = " ".join(s["target_text"].lower().strip().split())
            h = int(hashlib.md5(norm_txt.encode("utf-8")).hexdigest(), 16) % 10
            sub = s["subject_id"]
            if sub not in test_subs and sub not in val_subs and h < 8:
                train_list.append(s)
            elif sub in val_subs and h == 8:
                val_list.append(s)
            elif sub in test_subs and h == 9:
                test_list.append(s)

    elif split_strategy == "subject_independent":
        test_subs = set(held_out_test_subjects or [])
        val_subs = set(held_out_val_subjects or [])
        available_subs = sorted({s["subject_id"] for s in all_samples})

        if not (test_subs & set(available_subs)) and len(available_subs) >= 3:
            test_subs = {available_subs[-1]}
            val_subs = {available_subs[-2]}
        elif len(available_subs) < 3:
            split_strategy = "sentence_independent"

        if split_strategy == "subject_independent":
            for s in all_samples:
                if s["subject_id"] in test_subs:
                    test_list.append(s)
                elif s["subject_id"] in val_subs:
                    val_list.append(s)
                else:
                    train_list.append(s)

    if split_strategy == "sentence_independent":
        for s in all_samples:
            h = int(hashlib.md5(s["target_text"].encode("utf-8")).hexdigest(), 16) % 10
            if h < 8:
                train_list.append(s)
            elif h == 8:
                val_list.append(s)
            else:
                test_list.append(s)

    elif split_strategy == "within_subject_811":
        by_sub_task: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for s in all_samples:
            by_sub_task.setdefault((s["subject_id"], s["task_name"]), []).append(s)
        for _, items in by_sub_task.items():
            items = sorted(items, key=lambda x: x["sentence_idx"])
            n = len(items)
            n_train = max(1, int(0.8 * n))
            n_val = max(1, int(0.1 * n))
            train_list.extend(items[:n_train])
            val_list.extend(items[n_train : n_train + n_val])
            test_list.extend(items[n_train + n_val :])

    return (
        ZuCoEEGTextDataset(train_list, eeg_only_mode=eeg_only_mode),
        ZuCoEEGTextDataset(val_list, eeg_only_mode=eeg_only_mode),
        ZuCoEEGTextDataset(test_list, eeg_only_mode=eeg_only_mode),
    )
