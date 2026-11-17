"""
Leakage-Free Clinical Dataset & Split Engine for Paper 2 (`NeuroReasoner`).

Supports 4 rigorous Q1 evaluation protocols:
  1. `multitask_cot`: Unified Cognitive (AD/FTD/CN + MMSE) + Cross-Center PD (UNM) + Affective (MODMA 128ch & 3ch + PHQ-9/GAD-7)
     with `UIowa_Center` (28 subjects) strictly held out as an Unseen External Hospital Cohort!
  2. `eeg_macs_crosscenter`: Direct 1-to-1 benchmark against `EEG-MACS` (Song & Zhang et al., ACM MM 2024) —
     3-Fold Subject-Independent CV on `UNM` (54 subjects) + Zero-Shot Cross-Center Transfer to `UIowa` (28 subjects).
  3. `ds004504_5fold`: 5-Fold Stratified Subject-Independent CV on OpenNeuro `ds004504` (88 subjects: 36 AD, 23 FTD, 29 CN).
  4. `modma_crossdevice`: Patient-disjoint High-Density (128-ch) -> Pervasive Wearable (3-ch Fp1/Fpz/Fp2) transfer
     (strictly groups `MODMA128ch_0201xxxx` and `MODMA3ch_0201xxxx` by canonical patient ID so the same outpatient
     never leaks across train and test!).
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def canonical_patient_key(record: Dict[str, Any]) -> str:
    """
    Map recording-level `subject_id` to a strict patient-level identity.
    Crucially links `MODMA128ch_02010002` and `MODMA3ch_02010002` to `MODMA_02010002`
    to guarantee zero patient leakage between 128-channel and 3-channel splits.
    """
    sid = str(record["subject_id"])
    m_modma = re.search(r"(020[123]\d{4})", sid)
    if m_modma and "modma" in str(record.get("cohort_name", "")).lower():
        return f"MODMA_{m_modma.group(1)}"
    return sid


class NeuroReasonerClinicalDataset(Dataset):
    """PyTorch Dataset yielding Dual-Stream Connectome + CoT Clinical Grounding batches."""

    def __init__(self, samples: List[Dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        return {
            "eeg_features": item["eeg_features"].float(),
            "node_features": item["node_features"].float(),
            "word_coupled_edges": item["word_coupled_edges"].float(),
            "sent_connectome": item["sent_connectome"].float(),
            "roi_coverage_mask": item["roi_coverage_mask"].float(),
            "eeg_attn_mask": item["eeg_attn_mask"].long(),
            "rel_pos": item["rel_pos"].float(),
            "lex_pos": item["lex_pos"].float(),
            "word_salience": item["word_salience"].float(),
            "cog_load": item["cog_load"].float(),
            "task_id": int(item["task_id"]),
            "numeric_label": int(item["numeric_label"]),
            "sentiment_label": int(item["numeric_label"]),
            "relation_label": -100,
            "clinical_prompt": item["clinical_prompt"],
            "target_text": item["target_text"],
            "subject_id": item["subject_id"],
            "patient_key": canonical_patient_key(item),
            "cohort_name": item["cohort_name"],
            "domain": item["domain"],
            "center_name": item["center_name"],
            "label_code": item["label_code"],
            "clinical_scores": item["clinical_scores"],
            "biomarkers": item["biomarkers"],
            "seq_len": int(item["seq_len"]),
        }


def collate_clinical_neuroreasoner(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "eeg_features": torch.stack([b["eeg_features"] for b in batch], dim=0),
        "node_features": torch.stack([b["node_features"] for b in batch], dim=0),
        "word_coupled_edges": torch.stack([b["word_coupled_edges"] for b in batch], dim=0),
        "sent_connectome": torch.stack([b["sent_connectome"] for b in batch], dim=0),
        "roi_coverage_mask": torch.stack([b["roi_coverage_mask"] for b in batch], dim=0),
        "eeg_attn_mask": torch.stack([b["eeg_attn_mask"] for b in batch], dim=0),
        "rel_pos": torch.stack([b["rel_pos"] for b in batch], dim=0),
        "lex_pos": torch.stack([b["lex_pos"] for b in batch], dim=0),
        "word_salience": torch.stack([b["word_salience"] for b in batch], dim=0),
        "cog_load": torch.stack([b["cog_load"] for b in batch], dim=0),
        "task_id": torch.tensor([b["task_id"] for b in batch], dtype=torch.long),
        "numeric_label": torch.tensor([b["numeric_label"] for b in batch], dtype=torch.long),
        "sentiment_label": torch.tensor([b["sentiment_label"] for b in batch], dtype=torch.long),
        "relation_label": torch.tensor([b["relation_label"] for b in batch], dtype=torch.long),
        "clinical_prompt": [b["clinical_prompt"] for b in batch],
        "target_text": [b["target_text"] for b in batch],
        "subject_id": [b["subject_id"] for b in batch],
        "patient_key": [b["patient_key"] for b in batch],
        "cohort_name": [b["cohort_name"] for b in batch],
        "domain": [b["domain"] for b in batch],
        "center_name": [b["center_name"] for b in batch],
        "label_code": [b["label_code"] for b in batch],
        "clinical_scores": [b["clinical_scores"] for b in batch],
        "biomarkers": [b["biomarkers"] for b in batch],
        "seq_len": torch.tensor([b["seq_len"] for b in batch], dtype=torch.long),
    }


def _stratified_patient_folds(
    records: List[Dict[str, Any]],
    num_folds: int = 5,
    seed: int = 42,
) -> List[set[str]]:
    """Deterministic stratified K-fold assignment at the canonical patient level."""
    patient_to_stratum: Dict[str, str] = {}
    for r in records:
        pk = canonical_patient_key(r)
        if pk not in patient_to_stratum:
            patient_to_stratum[pk] = f"{r['domain']}::{r['label_code']}"

    strata_buckets: Dict[str, List[str]] = defaultdict(list)
    for pk, st in patient_to_stratum.items():
        strata_buckets[st].append(pk)

    folds: List[set[str]] = [set() for _ in range(num_folds)]
    for st, pks in sorted(strata_buckets.items()):
        pks_sorted = sorted(
            pks,
            key=lambda k: hashlib.sha256(f"{seed}:{st}:{k}".encode("utf-8")).hexdigest(),
        )
        for i, pk in enumerate(pks_sorted):
            folds[i % num_folds].add(pk)
    return folds


def load_neuroreasoner_splits(
    processed_root: str = "./datasets/processed_clinical_neuroreasoner",
    protocol: str = "multitask_cot",
    fold_idx: int = 0,
    seed: int = 42,
) -> Dict[str, NeuroReasonerClinicalDataset]:
    """
    Load processed `.pkl` clinical cohorts and construct zero-leakage splits.

    Returns dict containing `"train"`, `"val"`, `"test"`, and (when available) `"external_crosscenter_test"`.
    """
    def _load_pkl(name: str) -> List[Dict[str, Any]]:
        p = os.path.join(processed_root, f"{name}.pkl")
        if not os.path.exists(p):
            return []
        with open(p, "rb") as f:
            return pickle.load(f)

    if protocol == "eeg_macs_crosscenter":
        unm = _load_pkl("unm_pd_crosscenter")
        uiowa = _load_pkl("uiowa_pd_crosscenter")
        folds = _stratified_patient_folds(unm, num_folds=3, seed=seed)
        val_pks = folds[fold_idx % 3]
        train_recs = [r for r in unm if canonical_patient_key(r) not in val_pks]
        val_recs = [r for r in unm if canonical_patient_key(r) in val_pks]
        return {
            "train": NeuroReasonerClinicalDataset(train_recs),
            "val": NeuroReasonerClinicalDataset(val_recs),
            "test": NeuroReasonerClinicalDataset(uiowa),  # Direct Zero-Shot Cross-Center Test on UIowa!
        }

    if protocol == "ds004504_5fold":
        ds = _load_pkl("ds004504_cognitive")
        folds = _stratified_patient_folds(ds, num_folds=5, seed=seed)
        test_pks = folds[fold_idx % 5]
        val_pks = folds[(fold_idx + 1) % 5]
        return {
            "train": NeuroReasonerClinicalDataset([r for r in ds if canonical_patient_key(r) not in (test_pks | val_pks)]),
            "val": NeuroReasonerClinicalDataset([r for r in ds if canonical_patient_key(r) in val_pks]),
            "test": NeuroReasonerClinicalDataset([r for r in ds if canonical_patient_key(r) in test_pks]),
        }

    if protocol == "modma_crossdevice":
        m128 = _load_pkl("modma_128ch_affective")
        m3 = _load_pkl("modma_3ch_affective")
        folds = _stratified_patient_folds(m128 + m3, num_folds=5, seed=seed)
        test_pks = folds[fold_idx % 5]
        val_pks = folds[(fold_idx + 1) % 5]
        train_pool = [r for r in (m128 + m3) if canonical_patient_key(r) not in (test_pks | val_pks)]
        val_pool = [r for r in m3 if canonical_patient_key(r) in val_pks]
        test_wearable = [r for r in m3 if canonical_patient_key(r) in test_pks]
        return {
            "train": NeuroReasonerClinicalDataset(train_pool),
            "val": NeuroReasonerClinicalDataset(val_pool),
            "test": NeuroReasonerClinicalDataset(test_wearable),
        }

    # Default: `multitask_cot` across all 3 clinical domains + Unseen External Hospital (`UIowa`)
    in_center_pool = (
        _load_pkl("ds004504_cognitive")
        + _load_pkl("unm_pd_crosscenter")
        + _load_pkl("modma_128ch_affective")
        + _load_pkl("modma_3ch_affective")
    )
    external_uiowa = _load_pkl("uiowa_pd_crosscenter")
    folds = _stratified_patient_folds(in_center_pool, num_folds=5, seed=seed)
    test_pks = folds[fold_idx % 5]
    val_pks = folds[(fold_idx + 1) % 5]

    return {
        "train": NeuroReasonerClinicalDataset([r for r in in_center_pool if canonical_patient_key(r) not in (test_pks | val_pks)]),
        "val": NeuroReasonerClinicalDataset([r for r in in_center_pool if canonical_patient_key(r) in val_pks]),
        "test": NeuroReasonerClinicalDataset([r for r in in_center_pool if canonical_patient_key(r) in test_pks]),
        "external_crosscenter_test": NeuroReasonerClinicalDataset(external_uiowa),
    }
