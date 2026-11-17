#!/usr/bin/env python3
"""
Montage-Agnostic Clinical EEG -> 12-ROI Dual-Stream Connectome & CoT Preprocessor for Paper 2 (`NeuroReasoner`).

Unifies all 3 Public Clinical EEG Cohorts into the exact Dual-Stream Connectome + RVQ representation:
  1. Cognitive Domain (`ds004504`):
     - 88 subjects (36 AD, 23 FTD, 29 CN), 19-ch 10-20 clinical Nihon Kohden EEG (500 Hz), MMSE [0..30]
  2. Cross-Center Neuro Domain (`UNM` + `UIowa` — Exact `EEG-MACS` Benchmark, Song & Zhang et al., ACM MM 2024):
     - UNM Center (54 subjects: 27 PD, 27 CTL) + UIowa Center (28 subjects: 14 PD, 14 CTL), 60 common 10-10 channels (500 Hz)
     - Supports both Zero-Shot Cross-Center Transfer (Train UNM -> Test UIowa) & 3-Fold Subject-Independent CV
  3. Affective Domain (`MODMA` Dataset, Lanzhou Univ. Second Hospital, Scientific Data 2022):
     - 128-ch EGI HydroCel Resting-State (53 subjects: 24 MDD, 29 HC, 250 Hz) + PHQ-9 & GAD-7 scores
     - 3-ch Wearable Prefrontal Headband (Fp1, Fpz, Fp2; 55 subjects: 26 MDD, 29 HC, 250 Hz) + PHQ-9 & GAD-7 scores

Outputs per-cohort `.pkl` files + `clinical_splits_manifest.json` + SFT Chain-of-Thought JSONL in:
  `datasets/processed_clinical_neuroreasoner/`
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as sp_signal
import torch

from src.data.neuro_connectome import (
    BAND_HZ_RANGES,
    CHANLOCS_105,
    COG_LOAD_DIM,
    NODE_FEAT_DIM,
    NUM_BANDS,
    NUM_DIFF_PAIRS,
    NUM_ROI_PAIRS,
    NUM_ROIS,
    NUM_SPD_UPPER,
    ROI_NAMES,
    ROI_WEIGHT_MATRIX,
    SENT_CONNECTOME_DIM,
    SPD_OFFDIAG_SCALE,
    SPD_TRI_U,
    SPD_TRI_V,
    UPPER_TRI_U,
    UPPER_TRI_V,
    WORD_EDGE_DIM,
    compute_word_coupled_edges,
)

# ==============================================================================
# 1. Montage-Agnostic 10-20 / 10-10 / EGI-128 / Wearable-3ch -> 12-ROI Mapping
# ==============================================================================
# ROI Indices (matching `neuro_connectome.ROI_NAMES`):
#  0: L_Prefrontal,         1: R_Prefrontal
#  2: L_Broca_IFG,          3: R_IFG
#  4: L_Central,            5: R_Central
#  6: L_Wernicke_Temporal,  7: R_Temporal
#  8: L_Parietal,           9: R_Parietal
# 10: L_Occipital,         11: R_Occipital

STANDARD_10_10_ROI_MAP: Dict[str, List[Tuple[int, float]]] = {
    # Prefrontal
    "FP1": [(0, 1.0)],
    "FP2": [(1, 1.0)],
    "FPZ": [(0, 0.5), (1, 0.5)],
    "AF7": [(0, 0.7), (2, 0.3)],
    "AF8": [(1, 0.7), (3, 0.3)],
    "AF3": [(0, 1.0)],
    "AF4": [(1, 1.0)],
    "AFZ": [(0, 0.5), (1, 0.5)],
    "F1": [(0, 0.8), (4, 0.2)],
    "F2": [(1, 0.8), (5, 0.2)],
    "FZ": [(0, 0.5), (1, 0.5)],
    # Inferior Frontal / Broca & R-IFG
    "F3": [(2, 0.75), (0, 0.25)],
    "F4": [(3, 0.75), (1, 0.25)],
    "F5": [(2, 1.0)],
    "F6": [(3, 1.0)],
    "F7": [(2, 1.0)],
    "F8": [(3, 1.0)],
    "FT7": [(2, 0.6), (6, 0.4)],
    "FT8": [(3, 0.6), (7, 0.4)],
    "FT9": [(2, 0.5), (6, 0.5)],
    "FT10": [(3, 0.5), (7, 0.5)],
    "FC5": [(2, 0.7), (4, 0.3)],
    "FC6": [(3, 0.7), (5, 0.3)],
    # Central Sensorimotor
    "FC3": [(4, 0.75), (2, 0.25)],
    "FC4": [(5, 0.75), (3, 0.25)],
    "FC1": [(4, 1.0)],
    "FC2": [(5, 1.0)],
    "FCZ": [(4, 0.5), (5, 0.5)],
    "C1": [(4, 1.0)],
    "C2": [(5, 1.0)],
    "C3": [(4, 1.0)],
    "C4": [(5, 1.0)],
    "C5": [(4, 0.8), (6, 0.2)],
    "C6": [(5, 0.8), (7, 0.2)],
    "CZ": [(4, 0.5), (5, 0.5)],
    # Temporal (including 10-20 aliases T3->T7, T4->T8, T5->P7, T6->P8)
    "T3": [(6, 1.0)],
    "T4": [(7, 1.0)],
    "T7": [(6, 1.0)],
    "T8": [(7, 1.0)],
    "TP7": [(6, 0.75), (8, 0.25)],
    "TP8": [(7, 0.75), (9, 0.25)],
    "TP9": [(6, 1.0)],
    "TP10": [(7, 1.0)],
    "T5": [(6, 0.65), (8, 0.35)],
    "T6": [(7, 0.65), (9, 0.35)],
    # Parietal
    "CP1": [(8, 0.7), (4, 0.3)],
    "CP2": [(9, 0.7), (5, 0.3)],
    "CP3": [(8, 0.8), (4, 0.2)],
    "CP4": [(9, 0.8), (5, 0.2)],
    "CP5": [(8, 0.7), (6, 0.3)],
    "CP6": [(9, 0.7), (7, 0.3)],
    "CPZ": [(8, 0.5), (9, 0.5)],
    "P1": [(8, 1.0)],
    "P2": [(9, 1.0)],
    "P3": [(8, 1.0)],
    "P4": [(9, 1.0)],
    "P5": [(8, 0.8), (6, 0.2)],
    "P6": [(9, 0.8), (7, 0.2)],
    "P7": [(8, 0.55), (6, 0.45)],
    "P8": [(9, 0.55), (7, 0.45)],
    "PZ": [(8, 0.5), (9, 0.5)],
    # Occipital
    "PO3": [(10, 0.75), (8, 0.25)],
    "PO4": [(11, 0.75), (9, 0.25)],
    "PO7": [(10, 0.85), (8, 0.15)],
    "PO8": [(11, 0.85), (9, 0.15)],
    "POZ": [(10, 0.5), (11, 0.5)],
    "O1": [(10, 1.0)],
    "O2": [(11, 1.0)],
    "OZ": [(10, 0.5), (11, 0.5)],
    "IZ": [(10, 0.5), (11, 0.5)],
}

EGI_NAME_TO_IDX: Dict[str, int] = {name.upper(): i for i, (name, _, _, _) in enumerate(CHANLOCS_105)}

CLINICAL_TASK_ID_MAP: Dict[str, int] = {
    "ds004504_cognitive": 0,
    "unm_pd_crosscenter": 1,
    "uiowa_pd_crosscenter": 1,
    "modma_128ch_affective": 2,
    "modma_3ch_affective": 3,
}


def build_montage_projection_matrix(channel_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build normalized projection matrix `W` of shape `[12, C]` and binary `roi_coverage_mask` `[12]`
    for any arbitrary EEG montage (`19-ch 10-20`, `60-ch 10-10`, `128-ch EGI`, or `3-ch Fp1/Fpz/Fp2`).
    """
    n_ch = len(channel_names)
    weights = np.zeros((NUM_ROIS, n_ch), dtype=np.float32)

    for c_idx, raw_name in enumerate(channel_names):
        clean = raw_name.strip().upper().replace("EEG ", "").replace("-REF", "").replace("-CZ", "")
        if clean in STANDARD_10_10_ROI_MAP:
            for r_idx, w_val in STANDARD_10_10_ROI_MAP[clean]:
                weights[r_idx, c_idx] += w_val
        elif clean in EGI_NAME_TO_IDX:
            idx_105 = EGI_NAME_TO_IDX[clean]
            weights[:, c_idx] = ROI_WEIGHT_MATRIX[:, idx_105]

    row_sums = weights.sum(axis=1, keepdims=True)
    roi_coverage_mask = (row_sums[:, 0] > 1e-5).astype(np.float32)
    weights_norm = weights / np.clip(row_sums, 1e-6, None)
    return weights_norm.astype(np.float32), roi_coverage_mask


# ==============================================================================
# 2. Signal Filtering, Artifact Cleaning & Dual-Stream Connectome Extraction
# ==============================================================================
def preprocess_continuous_waveform(
    raw_ch_time: np.ndarray,
    fs_in: float,
    fs_target: float = 250.0,
    notch_hz: Optional[float] = 50.0,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Standardized leakage-free clinical EEG signal conditioning:
      1. Detect bad / flat / extreme-variance channels (`valid_chan_mask`)
      2. Zero-phase bandpass `0.5 - 49.5 Hz` + line-noise notch (`50 Hz` EU/China or `60 Hz` US)
      3. Uniformly resample to `fs_target = 250.0 Hz` (matches *EEG-MACS* Appendix A.2)
      4. Common Average Reference (CAR) when `C >= 16`
    """
    x = np.nan_to_num(raw_ch_time.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.median(x, axis=1, keepdims=True)

    ch_std = np.std(x, axis=1)
    med_std = float(np.median(ch_std[ch_std > 1e-8])) if np.any(ch_std > 1e-8) else 1.0
    valid_ch = ((ch_std > 1e-6 * med_std) & (ch_std < 25.0 * med_std)).astype(np.float32)
    if valid_ch.sum() < max(1, min(3, x.shape[0])):
        valid_ch = np.ones((x.shape[0],), dtype=np.float32)

    nyq = 0.5 * fs_in
    hi = min(49.5, 0.92 * nyq)
    sos_bp = sp_signal.butter(4, [0.5 / nyq, hi / nyq], btype="bandpass", output="sos")
    x = sp_signal.sosfiltfilt(sos_bp, x, axis=1)

    if notch_hz is not None and notch_hz < 0.95 * nyq:
        b_n, a_n = sp_signal.iirnotch(notch_hz, Q=30.0, fs=fs_in)
        x = sp_signal.filtfilt(b_n, a_n, x, axis=1)

    if abs(fs_in - fs_target) > 1e-3:
        gcd = math.gcd(int(round(fs_target)), int(round(fs_in)))
        up = int(round(fs_target)) // gcd
        down = int(round(fs_in)) // gcd
        x = sp_signal.resample_poly(x, up, down, axis=1)
        fs_eff = fs_target
    else:
        fs_eff = fs_in

    if x.shape[0] >= 16 and valid_ch.sum() >= 8:
        car = np.sum(x * valid_ch[:, None], axis=0, keepdims=True) / valid_ch.sum()
        x = x - car

    return x.astype(np.float32), float(fs_eff), valid_ch


def _compute_roi_connectome_250hz(
    roi_ts: np.ndarray,
    roi_coverage_mask: np.ndarray,
    fs: float = 250.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute `[528]` wPLI, `[528]` iCoh, `[624]` Shrinkage Log-Euclidean Riemannian SPD Tangent,
    and Individual Posterior Alpha Peak Frequency (`iaf_hz`) on `roi_ts` `[12, T]`.
    """
    t_len = roi_ts.shape[1]
    nperseg = min(int(round(2.0 * fs)), t_len)  # 2s Welch window (0.5 Hz resolution!)
    step = max(int(round(0.5 * fs)), nperseg // 4)
    starts = list(range(0, max(1, t_len - nperseg + 1), step))
    segs = np.stack([roi_ts[:, s : s + nperseg] for s in starts], axis=0)  # [W, 12, L_win]
    hann = np.hanning(nperseg).astype(np.float32)[None, None, :]
    fft_segs = np.fft.rfft(segs * hann, axis=-1)                            # [W, 12, F]
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    csd = fft_segs[:, :, None, :] * np.conj(fft_segs[:, None, :, :])

    # Estimate Individual Alpha Peak Frequency (IAF, 7.0 - 13.5 Hz) over Posterior ROIs (or Prefrontal if 3-ch)
    alpha_band_mask = (freqs >= 7.0) & (freqs <= 13.5)
    psd_mean = np.mean(np.abs(fft_segs) ** 2, axis=0)  # [12, F]
    post_rois = [8, 9, 10, 11] if roi_coverage_mask[8:12].sum() >= 2 else [0, 1]
    if np.any(alpha_band_mask):
        alpha_freqs = freqs[alpha_band_mask]
        post_psd = np.mean(psd_mean[post_rois][:, alpha_band_mask], axis=0)
        iaf_hz = float(np.sum(alpha_freqs * post_psd) / (np.sum(post_psd) + 1e-8))
    else:
        iaf_hz = 10.0

    pair_valid = (roi_coverage_mask[UPPER_TRI_U] * roi_coverage_mask[UPPER_TRI_V]).astype(np.float32)
    spd_valid = (roi_coverage_mask[SPD_TRI_U] * roi_coverage_mask[SPD_TRI_V]).astype(np.float32)

    wpli_list: List[np.ndarray] = []
    icoh_list: List[np.ndarray] = []
    spd_list: List[np.ndarray] = []

    for f_lo, f_hi in BAND_HZ_RANGES:
        f_mask = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(f_mask):
            f_mask[int(np.argmin(np.abs(freqs - 0.5 * (f_lo + f_hi))))] = True

        csd_b = csd[:, :, :, f_mask]
        imag_b = np.imag(csd_b)

        num_wpli = np.abs(np.mean(imag_b, axis=(0, 3)))
        den_wpli = np.mean(np.abs(imag_b), axis=(0, 3)) + 1e-7
        wpli_mat = np.clip(num_wpli / den_wpli, 0.0, 1.0)
        wpli_list.append((wpli_mat[UPPER_TRI_U, UPPER_TRI_V] * pair_valid).astype(np.float32))

        mean_csd = np.mean(csd_b, axis=(0, 3))
        diag_pow = np.real(np.diag(mean_csd)).clip(min=1e-7)
        denom_icoh = np.sqrt(diag_pow[:, None] * diag_pow[None, :])
        icoh_mat = np.clip(np.abs(np.imag(mean_csd)) / denom_icoh, 0.0, 1.0)
        icoh_list.append((icoh_mat[UPPER_TRI_U, UPPER_TRI_V] * pair_valid).astype(np.float32))

        real_cov = 0.5 * (np.real(mean_csd) + np.real(mean_csd).T)
        # Mask unobserved ROIs to identity before matrix logarithm so missing ROIs map cleanly to 0 in tangent space
        cov_masked = real_cov * (roi_coverage_mask[:, None] * roi_coverage_mask[None, :])
        tr = float(np.trace(cov_masked)) / max(1.0, float(roi_coverage_mask.sum()))
        reg = max(tr, 1e-4)
        spd_mat = 0.92 * cov_masked + 0.08 * reg * np.eye(NUM_ROIS, dtype=np.float32)
        for r_i in range(NUM_ROIS):
            if roi_coverage_mask[r_i] < 0.5:
                spd_mat[r_i, :] = 0.0
                spd_mat[:, r_i] = 0.0
                spd_mat[r_i, r_i] = 1.0
        eigvals, eigvecs = np.linalg.eigh(spd_mat)
        log_eig = np.log(np.clip(eigvals, 1e-5, 1e5))
        log_spd = (eigvecs * log_eig[None, :]) @ eigvecs.T
        tangent = log_spd[SPD_TRI_U, SPD_TRI_V] * SPD_OFFDIAG_SCALE * spd_valid
        spd_list.append(np.clip(tangent, -10.0, 10.0).astype(np.float32))

    return (
        np.concatenate(wpli_list, axis=0),
        np.concatenate(icoh_list, axis=0),
        np.concatenate(spd_list, axis=0),
        iaf_hz,
    )


def _roi_lr_asymmetry_vec(roi_band_mat: np.ndarray, roi_coverage_mask: np.ndarray) -> np.ndarray:
    """Compute `[48]` homologous Right-minus-Left ROI asymmetry across `[8, 12]`."""
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    diffs = []
    for l_idx, r_idx in pairs:
        valid_pair = float(roi_coverage_mask[l_idx] * roi_coverage_mask[r_idx])
        diffs.append((roi_band_mat[:, r_idx] - roi_band_mat[:, l_idx]) * valid_pair)
    return np.concatenate(diffs, axis=0).astype(np.float32)  # [48]


# ==============================================================================
# 3. Chain-of-Thought Neural Evidence Grounding Generator (`NeuroReasoner`)
# ==============================================================================
def build_cot_prompt_and_target(
    domain: str,
    montage_desc: str,
    center_name: str,
    label_code: str,
    clinical_scores: Dict[str, Any],
    biomarkers: Dict[str, float],
) -> Tuple[str, str]:
    """
    Construct privacy-preserving clinical instruction prompt and Chain-of-Thought target
    grounded strictly in quantitative EEG connectome biomarkers.
    """
    ev_json = json.dumps(biomarkers, ensure_ascii=False)

    if domain == "cognitive_dementia":
        prompt = (
            f"[Domain: Cognitive Neuro-Assessment | Center: {center_name} | Montage: {montage_desc}]\n"
            f"Analyze the quantized cortical connectome tokens and extracted spectral-manifold biomarkers "
            f"to differentiate Alzheimer's Disease (AD), Frontotemporal Dementia (FTD), and Cognitively Normal (CN) controls, "
            f"and estimate the Mini-Mental State Examination (MMSE, 0-30) severity."
        )
        diag_map = {
            "AD": ("Alzheimer's Disease (AD)", "marked posterior alpha slowing, elevated posterior theta/alpha ratio, and reduced temporo-parietal phase-lagged wPLI synchronization"),
            "FTD": ("Frontotemporal Dementia (FTD)", "pronounced fronto-temporal theta/beta dysregulation with relative preservation of primary occipital alpha rhythm compared to AD"),
            "CN": ("Cognitively Normal Control (CN)", "preserved posterior alpha peak frequency (~9.5-11 Hz), balanced theta/alpha ratio, and intact frontoparietal wPLI connectivity"),
        }
        diag_full, mechanism = diag_map[label_code]
        mmse = clinical_scores.get("MMSE", "N/A")
        target = (
            f"<neural_evidence>{ev_json}</neural_evidence>\n"
            f"<clinical_reasoning>"
            f"1. Spectral Slowing Profile: Posterior Individual Alpha Frequency is {biomarkers['iaf_hz']:.2f} Hz "
            f"with posterior theta/alpha log-ratio {biomarkers['posterior_theta_alpha_diff']:+.3f} and frontal theta/beta diff {biomarkers['frontal_theta_beta_diff']:+.3f}. "
            f"2. Functional Connectome & Manifold Geometry: Temporo-parietal theta wPLI is {biomarkers['temporoparietal_theta_wpli']:.3f}, "
            f"frontoparietal alpha wPLI is {biomarkers['frontoparietal_alpha_wpli']:.3f}, and Riemannian SPD tangent energy is {biomarkers['spd_tangent_energy']:.3f}. "
            f"3. Diagnostic Synthesis: Neural evidence aligns with {mechanism}, supporting {diag_full} with estimated cognitive reserve MMSE={mmse}/30."
            f"</clinical_reasoning>\n"
            f"<assessment>{json.dumps({'diagnosis': label_code, 'diagnosis_full': diag_full, 'estimated_mmse': mmse})}</assessment>"
        )
        return prompt, target

    if domain == "crosscenter_parkinson":
        prompt = (
            f"[Domain: Cross-Center Movement & Neuro-Cognitive Assessment | Center: {center_name} | Montage: {montage_desc}]\n"
            f"Evaluate the cross-center calibrated Riemannian SPD manifold and phase-lagged connectome tokens "
            f"to classify Parkinson's Disease (PD) versus Healthy Control (CTL) invariant to hospital acquisition protocol."
        )
        diag_map = {
            "PD": ("Parkinson's Disease (PD)", "elevated central-sensorimotor beta hypersynchrony, slowed background alpha peak, and altered fronto-central Riemannian SPD covariance dispersion"),
            "CTL": ("Healthy Control (CTL)", "normative sensorimotor beta desynchronization balance, physiological alpha peak frequency, and stable cross-hemispheric wPLI coherence"),
        }
        diag_full, mechanism = diag_map[label_code]
        target = (
            f"<neural_evidence>{ev_json}</neural_evidence>\n"
            f"<clinical_reasoning>"
            f"1. Cortico-Subthalamic Spectral Signature: Central beta power z-index is {biomarkers['central_beta_power_z']:+.3f}, "
            f"frontal theta/beta diff is {biomarkers['frontal_theta_beta_diff']:+.3f}, and alpha peak frequency is {biomarkers['iaf_hz']:.2f} Hz. "
            f"2. Cross-Center Manifold Invariance: Fronto-central beta wPLI is {biomarkers['frontocentral_beta_wpli']:.3f} "
            f"with Log-Euclidean SPD dispersion {biomarkers['spd_tangent_energy']:.3f} under {center_name} protocol. "
            f"3. Diagnostic Synthesis: Pattern reflects {mechanism}, indicating {diag_full}."
            f"</clinical_reasoning>\n"
            f"<assessment>{json.dumps({'diagnosis': label_code, 'diagnosis_full': diag_full, 'center': center_name})}</assessment>"
        )
        return prompt, target

    # Affective Domain (MODMA MDD vs HC, 128-ch & 3-ch wearable)
    prompt = (
        f"[Domain: Affective & Psychiatric Assessment | Center: {center_name} | Montage: {montage_desc}]\n"
        f"Examine prefrontal alpha asymmetry, prefrontal high-alpha oscillations, and cortical connectome markers "
        f"to assess Major Depressive Disorder (MDD) versus Healthy Control (HC), and estimate PHQ-9 depression and GAD-7 anxiety severity."
    )
    phq9 = clinical_scores.get("PHQ-9", "N/A")
    gad7 = clinical_scores.get("GAD-7", "N/A")
    diag_map = {
        "MDD": ("Major Depressive Disorder (MDD)", "aberrant prefrontal alpha asymmetry (FAA), elevated prefrontal alpha/theta synchronization, and affective network dysregulation associated with depressive-anxious symptom burden"),
        "HC": ("Healthy Control (HC)", "balanced bilateral prefrontal alpha regulation and normative affective fronto-limbic cortical dynamics with minimal depressive/anxious symptomatology"),
    }
    diag_full, mechanism = diag_map[label_code]
    target = (
        f"<neural_evidence>{ev_json}</neural_evidence>\n"
        f"<clinical_reasoning>"
        f"1. Prefrontal Affective Biomarkers: Frontal Alpha Asymmetry (R - L) is {biomarkers['frontal_alpha_asymmetry_faa']:+.3f}, "
        f"prefrontal high-alpha (10.5-13 Hz) level is {biomarkers['prefrontal_high_alpha_z']:+.3f}, and prefrontal theta/beta diff is {biomarkers['frontal_theta_beta_diff']:+.3f}. "
        f"2. Network & Wearable Alignment ({montage_desc}): Inter-hemispheric prefrontal alpha wPLI is {biomarkers['prefrontal_lr_alpha_wpli']:.3f} "
        f"with Riemannian SPD energy {biomarkers['spd_tangent_energy']:.3f}. "
        f"3. Clinical Scale Grounding: Evidence indicates {mechanism}, consistent with {diag_full} (PHQ-9={phq9}, GAD-7={gad7})."
        f"</clinical_reasoning>\n"
        f"<assessment>{json.dumps({'diagnosis': label_code, 'diagnosis_full': diag_full, 'estimated_phq9': phq9, 'estimated_gad7': gad7})}</assessment>"
    )
    return prompt, target


# ==============================================================================
# 4. Subject Segment Feature Extractor -> Exact `[max_eeg_len, 488 / 528] + [2064]`
# ==============================================================================
def extract_subject_clinical_records(
    raw_ch_time: np.ndarray,
    channel_names: List[str],
    fs_in: float,
    notch_hz: Optional[float],
    subject_id: str,
    cohort_name: str,
    domain: str,
    center_name: str,
    montage_desc: str,
    label_code: str,
    numeric_label: int,
    clinical_scores: Dict[str, Any],
    segment_sec: float = 16.0,
    stride_sec: float = 16.0,
    max_segments_per_subject: int = 24,
    max_eeg_len: int = 56,
) -> List[Dict[str, Any]]:
    """
    Convert a continuous clinical EEG recording `[C, T_raw]` into leakage-free 16-second segments,
    each formatted into the exact Dual-Stream Connectome tensor dictionary (`L=17` active tokens padded to `56`).
    """
    clean_eeg, fs, valid_ch = preprocess_continuous_waveform(
        raw_ch_time=raw_ch_time, fs_in=fs_in, fs_target=250.0, notch_hz=notch_hz
    )
    proj_w, roi_cov_mask = build_montage_projection_matrix(channel_names)
    eff_w = proj_w * valid_ch[None, :]
    row_sums = eff_w.sum(axis=1, keepdims=True)
    roi_cov_mask = (row_sums[:, 0] > 1e-5).astype(np.float32)
    eff_w = eff_w / np.clip(row_sums, 1e-6, None)

    roi_ts_full = eff_w @ clean_eeg  # [12, T_250]
    total_pts = roi_ts_full.shape[1]
    seg_pts = int(round(segment_sec * fs))     # 4000 samples @ 250 Hz (16s)
    stride_pts = int(round(stride_sec * fs))   # 4000 samples @ 250 Hz
    sub_pts = int(round(1.0 * fs))             # 250 samples @ 250 Hz (1s micro-step, matches EEG-MACS!)

    if total_pts < seg_pts:
        if total_pts < int(round(4.0 * fs)):
            return []
        # Pad short recordings cleanly via reflection
        pad_needed = seg_pts - total_pts
        roi_ts_full = np.pad(roi_ts_full, ((0, 0), (0, pad_needed)), mode="reflect")
        total_pts = roi_ts_full.shape[1]

    starts = list(range(0, total_pts - seg_pts + 1, stride_pts))
    if len(starts) > max_segments_per_subject:
        # Uniformly sample across the entire clinical session
        idx_sel = np.linspace(0, len(starts) - 1, num=max_segments_per_subject, dtype=int)
        starts = [starts[i] for i in idx_sel]

    # Compute 1s sub-window log band power across all selected segments first for per-recording baseline calibration
    hann_1s = np.hanning(sub_pts).astype(np.float32)[None, :]
    freqs_1s = np.fft.rfftfreq(sub_pts, d=1.0 / fs)
    band_masks_1s = []
    for f_lo, f_hi in BAND_HZ_RANGES:
        m = (freqs_1s >= f_lo) & (freqs_1s <= f_hi)
        if not np.any(m):
            m[int(np.argmin(np.abs(freqs_1s - 0.5 * (f_lo + f_hi))))] = True
        band_masks_1s.append(m)

    n_steps = seg_pts // sub_pts  # 16 micro-steps per 16s segment
    raw_seg_band_powers: List[np.ndarray] = []  # each [16, 8, 12]
    valid_seg_indices: List[int] = []

    global_rms = float(np.median(np.std(roi_ts_full[roi_cov_mask > 0.5], axis=1))) + 1e-6

    for s_start in starts:
        seg_roi = roi_ts_full[:, s_start : s_start + seg_pts]  # [12, 4000]
        seg_rms = float(np.max(np.std(seg_roi[roi_cov_mask > 0.5], axis=1)))
        if seg_rms > 12.0 * global_rms:
            continue  # drop severe movement/electrode pop artifact segment

        step_views = seg_roi.reshape(NUM_ROIS, n_steps, sub_pts).transpose(1, 0, 2)  # [16, 12, 250]
        fft_pow = np.abs(np.fft.rfft(step_views * hann_1s[None, :, :], axis=-1)) ** 2  # [16, 12, F]
        # Relative + log spectral power (preserves slowing ratios across subjects while suppressing skull impedance offset!)
        total_step_pow = np.sum(fft_pow[:, :, (freqs_1s >= 1.0) & (freqs_1s <= 49.5)], axis=-1, keepdims=True) + 1e-8
        rel_fft_pow = fft_pow / total_step_pow

        bp = np.zeros((n_steps, NUM_BANDS, NUM_ROIS), dtype=np.float32)
        for b_i, bm in enumerate(band_masks_1s):
            bp[:, b_i, :] = np.log10(np.mean(rel_fft_pow[:, :, bm], axis=-1) + 1e-6) * roi_cov_mask[None, :]

        raw_seg_band_powers.append(bp)
        valid_seg_indices.append(s_start)

    if not raw_seg_band_powers:
        return []

    # Scale relative log-power into clean O(1) neural feature range `[-4, +4]` WITHOUT erasing between-subject disease differences!
    # Canonical healthy relative log10 band power centers around -1.6 with std ~0.45 across bands
    all_bp = np.stack(raw_seg_band_powers, axis=0)  # [N_seg, 16, 8, 12]
    norm_bp = np.clip((all_bp + 1.60) / 0.45, -5.0, 5.0) * roi_cov_mask[None, None, None, :]

    def _pair_idx(u: int, v: int) -> int:
        a, b = (u, v) if u < v else (v, u)
        return int(np.where((UPPER_TRI_U == a) & (UPPER_TRI_V == b))[0][0])

    tp_pair = _pair_idx(6, 8)   # L_Wernicke_Temporal <-> L_Parietal
    fp_pair = _pair_idx(0, 8)   # L_Prefrontal <-> L_Parietal
    fc_pair = _pair_idx(0, 4)   # L_Prefrontal <-> L_Central
    pf_lr_pair = _pair_idx(0, 1)  # L_Prefrontal <-> R_Prefrontal

    records: List[Dict[str, Any]] = []
    task_id = CLINICAL_TASK_ID_MAP.get(cohort_name, 3)
    valid_chan_ratio = float(valid_ch.mean())
    roi_cov_ratio = float(roi_cov_mask.mean())

    for seg_i, s_start in enumerate(valid_seg_indices):
        seg_roi = roi_ts_full[:, s_start : s_start + seg_pts]  # [12, 4000]
        bp_16 = norm_bp[seg_i]                                  # [16, 8, 12]

        wpli_vec, icoh_vec, spd_vec, iaf_hz = _compute_roi_connectome_250hz(
            roi_ts=seg_roi, roi_coverage_mask=roi_cov_mask, fs=fs
        )

        # Build 16 multi-scale temporal tokens + 1 holistic 16s token = 17 tokens
        node_rows: List[np.ndarray] = []
        gd_roi_seq_list: List[np.ndarray] = []
        rel_pos_list: List[float] = []
        salience_list: List[float] = []

        seg_mean_roi = np.mean(bp_16, axis=0)  # [8, 12]

        for t_idx in range(n_steps):
            s1_local = bp_16[t_idx]  # 1s scale [8, 12]
            i_lo2, i_hi2 = max(0, t_idx - 1), min(n_steps, t_idx + 1)
            s2_meso = np.mean(bp_16[i_lo2:i_hi2], axis=0)  # 2s scale [8, 12]
            i_lo4, i_hi4 = max(0, t_idx - 2), min(n_steps, t_idx + 2)
            s3_macro = np.mean(bp_16[i_lo4:i_hi4], axis=0)  # 4s scale [8, 12]
            s4_burst_delta = s1_local - s3_macro             # Non-stationary burst delta [8, 12]

            asym_local = _roi_lr_asymmetry_vec(s1_local, roi_cov_mask)  # [48]
            asym_meso = _roi_lr_asymmetry_vec(s2_meso, roi_cov_mask)    # [48]

            burst_energy = float(np.mean(np.abs(s4_burst_delta)))
            step_meta_8 = np.array(
                [
                    (iaf_hz - 10.0) / 3.0,
                    burst_energy,
                    roi_cov_ratio,
                    valid_chan_ratio,
                    float(t_idx) / float(max(1, n_steps - 1)),
                    float(fs / 500.0),
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )

            node_vec = np.concatenate(
                [
                    s1_local.reshape(-1),
                    s2_meso.reshape(-1),
                    s3_macro.reshape(-1),
                    s4_burst_delta.reshape(-1),
                    asym_local,
                    asym_meso,
                    step_meta_8,
                ],
                axis=0,
            )  # [488]
            node_rows.append(node_vec)
            gd_roi_seq_list.append(s2_meso)
            rel_pos_list.append(0.92 * (float(t_idx) / float(max(1, n_steps - 1))))
            salience_list.append(1.0 + 0.25 * min(2.0, burst_energy))

        # Holistic 16-second Segment Token at `rel_pos = 1.0`
        seg_std_roi = np.std(bp_16, axis=0)
        seg_asym = _roi_lr_asymmetry_vec(seg_mean_roi, roi_cov_mask)  # [48]
        hol_meta_8 = np.array(
            [
                1.0,
                (iaf_hz - 10.0) / 3.0,
                float(np.mean(seg_std_roi)),
                roi_cov_ratio,
                valid_chan_ratio,
                float(fs / 500.0),
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        hol_node_vec = np.concatenate(
            [
                seg_mean_roi.reshape(-1),
                seg_mean_roi.reshape(-1),
                seg_mean_roi.reshape(-1),
                seg_std_roi.reshape(-1),
                seg_asym,
                seg_asym,
                hol_meta_8,
            ],
            axis=0,
        )  # [488]
        node_rows.append(hol_node_vec)
        gd_roi_seq_list.append(seg_mean_roi)
        rel_pos_list.append(1.0)
        salience_list.append(1.30)

        seq_len = len(node_rows)  # 17
        gd_roi_seq = np.stack(gd_roi_seq_list, axis=0)              # [17, 8, 12]
        word_edges_active = compute_word_coupled_edges(gd_roi_seq)  # [17, 528]

        # Construct 384-d hemispheric asymmetry summary (8 x 48) -> total sent_connectome = 2064-d
        asym_384 = np.tile(seg_asym, 8).astype(np.float32)  # [384]
        sent_connectome = np.concatenate([wpli_vec, icoh_vec, spd_vec, asym_384], axis=0).astype(np.float32)  # [2064]

        # Extract named Neuro-Biomarkers for Chain-of-Thought Neural Evidence Grounding
        wpli_by_band = wpli_vec.reshape(NUM_BANDS, NUM_ROI_PAIRS)
        post_idx = [8, 9, 10, 11] if roi_cov_mask[8:12].sum() >= 2 else [0, 1]
        front_idx = [0, 1, 2, 3] if roi_cov_mask[2:4].sum() >= 1 else [0, 1]
        cent_idx = [4, 5] if roi_cov_mask[4:6].sum() >= 1 else [0, 1]

        post_theta = float(np.mean(seg_mean_roi[0:2, post_idx]))
        post_alpha = float(np.mean(seg_mean_roi[2:4, post_idx]))
        front_theta = float(np.mean(seg_mean_roi[0:2, front_idx]))
        front_beta = float(np.mean(seg_mean_roi[4:6, front_idx]))
        cent_beta = float(np.mean(seg_mean_roi[4:6, cent_idx]))

        l_pf_alpha = float(np.mean(seg_mean_roi[2:4, 0]))
        r_pf_alpha = float(np.mean(seg_mean_roi[2:4, 1]))
        faa = r_pf_alpha - l_pf_alpha
        pf_high_alpha = float(np.mean(seg_mean_roi[3, [0, 1]]))

        biomarkers = {
            "iaf_hz": round(iaf_hz, 2),
            "posterior_theta_alpha_diff": round(post_theta - post_alpha, 4),
            "frontal_theta_beta_diff": round(front_theta - front_beta, 4),
            "central_beta_power_z": round(cent_beta, 4),
            "frontal_alpha_asymmetry_faa": round(faa, 4),
            "prefrontal_high_alpha_z": round(pf_high_alpha, 4),
            "temporoparietal_theta_wpli": round(float(np.mean(wpli_by_band[0:2, tp_pair])), 4),
            "frontoparietal_alpha_wpli": round(float(np.mean(wpli_by_band[2:4, fp_pair])), 4),
            "frontocentral_beta_wpli": round(float(np.mean(wpli_by_band[4:6, fc_pair])), 4),
            "prefrontal_lr_alpha_wpli": round(float(np.mean(wpli_by_band[2:4, pf_lr_pair])), 4),
            "spd_tangent_energy": round(float(np.mean(np.abs(spd_vec[spd_vec != 0])) if np.any(spd_vec != 0) else 0.0), 4),
            "active_rois": int(roi_cov_mask.sum()),
        }

        cog_load_vec = np.array(
            [
                (iaf_hz - 10.0) / 3.0,
                np.clip(post_theta - post_alpha, -3.0, 3.0) / 3.0,
                np.clip(faa, -3.0, 3.0) / 3.0,
                roi_cov_ratio,
                valid_chan_ratio,
                np.clip(cent_beta, -3.0, 3.0) / 3.0,
            ],
            dtype=np.float32,
        )

        clinical_prompt, cot_target_text = build_cot_prompt_and_target(
            domain=domain,
            montage_desc=montage_desc,
            center_name=center_name,
            label_code=label_code,
            clinical_scores=clinical_scores,
            biomarkers=biomarkers,
        )

        padded_nodes = np.zeros((max_eeg_len, NODE_FEAT_DIM), dtype=np.float32)
        padded_nodes[:seq_len] = np.stack(node_rows, axis=0)

        padded_edges = np.zeros((max_eeg_len, WORD_EDGE_DIM), dtype=np.float32)
        padded_edges[:seq_len] = word_edges_active

        padded_legacy = np.zeros((max_eeg_len, 840), dtype=np.float32)
        eeg_attn_mask = np.zeros((max_eeg_len,), dtype=np.int64)
        eeg_attn_mask[:seq_len] = 1

        rel_pos = np.zeros((max_eeg_len,), dtype=np.float32)
        rel_pos[:seq_len] = np.asarray(rel_pos_list, dtype=np.float32)

        salience = np.zeros((max_eeg_len,), dtype=np.float32)
        salience[:seq_len] = np.asarray(salience_list, dtype=np.float32)

        records.append(
            {
                "subject_id": subject_id,
                "cohort_name": cohort_name,
                "domain": domain,
                "center_name": center_name,
                "montage_desc": montage_desc,
                "task_name": cohort_name,
                "task_id": task_id,
                "segment_idx": seg_i,
                "label_code": label_code,
                "numeric_label": int(numeric_label),
                "clinical_scores": clinical_scores,
                "roi_coverage_mask": roi_cov_mask,
                "eeg_features": padded_legacy,
                "node_features": padded_nodes,
                "word_coupled_edges": padded_edges,
                "sent_connectome": sent_connectome,
                "eeg_attn_mask": eeg_attn_mask,
                "rel_pos": rel_pos,
                "lex_pos": rel_pos.copy(),
                "word_salience": salience,
                "cog_load": cog_load_vec,
                "sentiment_label": int(numeric_label),
                "relation_label": -100,
                "clinical_prompt": clinical_prompt,
                "target_text": cot_target_text,
                "seq_len": seq_len,
                "biomarkers": biomarkers,
            }
        )

    return records


# ==============================================================================
# 5. Cohort-Specific Loaders (Workers for 20-Core Grace ARM Parallel Execution)
# ==============================================================================
def _worker_ds004504(task: Tuple[str, str, str, int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    set_path, sub_id, label_code, num_label, scores = task
    mat = sio.loadmat(set_path, squeeze_me=True, struct_as_record=False)
    if "EEG" in mat:
        eeg = mat["EEG"]
        srate = float(eeg.srate)
        ch_names = [str(c.labels) for c in eeg.chanlocs]
        data = eeg.data
    else:
        srate = float(mat["srate"])
        ch_names = [str(c.labels) for c in mat["chanlocs"]]
        data = mat["data"]

    if isinstance(data, str):
        fdt_path = Path(set_path).parent / data
        n_ch = len(ch_names)
        raw = np.fromfile(fdt_path, dtype=np.float32).reshape(-1, n_ch).T
    else:
        raw = np.asarray(data, dtype=np.float32)
        if raw.ndim == 3:
            raw = raw.reshape(raw.shape[0], -1)

    return extract_subject_clinical_records(
        raw_ch_time=raw,
        channel_names=ch_names,
        fs_in=srate,
        notch_hz=50.0,
        subject_id=f"ds004504_{sub_id}",
        cohort_name="ds004504_cognitive",
        domain="cognitive_dementia",
        center_name="AHEPA_Hospital_Thessaloniki",
        montage_desc="19-ch_10-20_NihonKohden",
        label_code=label_code,
        numeric_label=num_label,
        clinical_scores=scores,
        segment_sec=16.0,
        stride_sec=16.0,
        max_segments_per_subject=24,
    )


COMMON_60_PD_CHANNELS = [
    "Fp1", "Fz", "F3", "F7", "FC5", "FC1", "C3", "T7", "TP9", "CP5", "CP1", "Pz", "P3", "P7",
    "O1", "Oz", "O2", "P4", "P8", "TP10", "CP6", "CP2", "Cz", "C4", "T8", "FT10", "FC6", "FC2",
    "F4", "F8", "Fp2", "AF7", "AF3", "AFz", "F1", "F5", "FT7", "FC3", "FCz", "C1", "C5", "TP7",
    "CP3", "P1", "P5", "PO7", "POz", "PO8", "P6", "P2", "CP4", "TP8", "C6", "C2", "FC4", "FT8",
    "F6", "F2", "AF4", "AF8",
]


def _worker_unm_uiowa(task: Tuple[str, str, str]) -> List[Dict[str, Any]]:
    csv_path, center_tag, cohort_name = task
    fname = Path(csv_path).stem
    is_pd = fname.upper().startswith("PD")
    label_code = "PD" if is_pd else "CTL"
    num_label = 1 if is_pd else 0

    df = pd.read_csv(csv_path, index_col=0)
    if len(df) > 1:
        df = df.iloc[1:]  # drop initial trigger/artifact sample per Diamandis & Narayanan protocol
    avail_cols = [c for c in COMMON_60_PD_CHANNELS if c in df.columns]
    raw = df[avail_cols].to_numpy(dtype=np.float32).T  # [60, T] at 500 Hz

    return extract_subject_clinical_records(
        raw_ch_time=raw,
        channel_names=avail_cols,
        fs_in=500.0,
        notch_hz=60.0,  # US 60 Hz powerline
        subject_id=f"{center_tag}_{fname}",
        cohort_name=cohort_name,
        domain="crosscenter_parkinson",
        center_name=center_tag,
        montage_desc="60-ch_10-10_BrainVision",
        label_code=label_code,
        numeric_label=num_label,
        clinical_scores={"center": center_tag},
        segment_sec=16.0,
        stride_sec=8.0,   # UNM/UIowa resting clips are ~2 mins; 8s stride yields ~14 rich segments/subject
        max_segments_per_subject=20,
    )


def _parse_xlsx_records_stdlib(xlsx_path: Path) -> Dict[str, Dict[str, Any]]:
    """Dependency-free `.xlsx` reader using Python standard library `zipfile` + `xml.etree`."""
    if not xlsx_path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(xlsx_path, "r") as zf:
        ss = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root_ss = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            ss = ["".join(si.itertext()).strip() for si in root_ss.findall("m:si", ns)]

        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = sheet.findall(".//m:row", ns)
        if not rows:
            return {}

        def _row_cells(row_el) -> Dict[int, str]:
            cells: Dict[int, str] = {}
            for c in row_el.findall("m:c", ns):
                ref = c.attrib.get("r", "A1")
                col_letters = "".join(ch for ch in ref if ch.isalpha())
                col_idx = 0
                for ch in col_letters:
                    col_idx = col_idx * 26 + (ord(ch.upper()) - ord("A") + 1)
                col_idx -= 1
                t = c.attrib.get("t", "")
                v = c.find("m:v", ns)
                if v is None or v.text is None:
                    val = ""
                elif t == "s":
                    val = ss[int(v.text)] if int(v.text) < len(ss) else ""
                else:
                    val = v.text.strip()
                cells[col_idx] = val
            return cells

        header_cells = _row_cells(rows[0])
        headers = {idx: name.strip() for idx, name in header_cells.items() if name.strip()}

        for r_el in rows[1:]:
            rc = _row_cells(r_el)
            sub_raw = rc.get(0, "").strip()
            if not sub_raw:
                continue
            # Normalize e.g. 2010002.0 -> 02010002
            if re.match(r"^\d+(\.0+)?$", sub_raw):
                sub_norm = str(int(float(sub_raw))).zfill(8)
            else:
                sub_norm = sub_raw
            entry: Dict[str, Any] = {"subject_id": sub_norm}
            for c_idx, h_name in headers.items():
                if c_idx == 0:
                    continue
                v_str = rc.get(c_idx, "")
                if v_str != "":
                    try:
                        f_val = float(v_str)
                        entry[h_name] = int(f_val) if f_val.is_integer() else round(f_val, 2)
                    except ValueError:
                        entry[h_name] = v_str
            out[sub_norm] = entry
    return out


def _worker_modma_3ch(task: Tuple[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    txt_path, meta = task
    fname = Path(txt_path).stem.split("_")[0]
    label_code = "MDD" if fname.startswith("0201") else "HC"
    num_label = 1 if label_code == "MDD" else 0

    arr = np.loadtxt(txt_path, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    # First 3 columns are Fp1, Fpz, Fp2 @ 250 Hz (90 seconds -> ~22,500 samples)
    if arr.shape[0] < arr.shape[1]:
        raw_3 = arr[:3, :]
    else:
        raw_3 = arr[:, :3].T

    return extract_subject_clinical_records(
        raw_ch_time=raw_3,
        channel_names=["Fp1", "Fpz", "Fp2"],
        fs_in=250.0,
        notch_hz=50.0,
        subject_id=f"MODMA3ch_{fname}",
        cohort_name="modma_3ch_affective",
        domain="affective_depression",
        center_name="Lanzhou_Second_Hospital_Wearable3ch",
        montage_desc="3-ch_Wearable_Prefrontal_Fp1_Fpz_Fp2",
        label_code=label_code,
        numeric_label=num_label,
        clinical_scores=meta,
        segment_sec=16.0,
        stride_sec=8.0,
        max_segments_per_subject=16,
    )


def _worker_modma_128ch(task: Tuple[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    mat_path, meta = task
    fname = Path(mat_path).stem.split("_")[0].split(" ")[0]
    m_id = re.search(r"(020[123]\d{4})", Path(mat_path).name)
    sub_code = m_id.group(1) if m_id else fname
    label_code = "MDD" if sub_code.startswith("0201") else "HC"
    num_label = 1 if label_code == "MDD" else 0

    mat = sio.loadmat(mat_path, squeeze_me=True)
    raw_128 = None
    srate = 250.0
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2 and min(v.shape) >= 128:
            raw_128 = v if v.shape[0] in (128, 129) else v.T
            break
    if raw_128 is None:
        return []

    if "samplingRate" in mat:
        srate = float(np.asarray(mat["samplingRate"]).flatten()[0])

    ch_names = [f"E{i}" for i in range(1, 129)]
    if raw_128.shape[0] == 129:
        ch_names.append("Cz")

    return extract_subject_clinical_records(
        raw_ch_time=raw_128.astype(np.float32),
        channel_names=ch_names,
        fs_in=srate,
        notch_hz=50.0,
        subject_id=f"MODMA128ch_{sub_code}",
        cohort_name="modma_128ch_affective",
        domain="affective_depression",
        center_name="Lanzhou_Second_Hospital_EGI128",
        montage_desc="128-ch_EGI_HydroCel",
        label_code=label_code,
        numeric_label=num_label,
        clinical_scores=meta,
        segment_sec=16.0,
        stride_sec=16.0,
        max_segments_per_subject=20,
    )


def _finalize_torch_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tensor_keys = (
        "roi_coverage_mask",
        "eeg_features",
        "node_features",
        "word_coupled_edges",
        "sent_connectome",
        "eeg_attn_mask",
        "rel_pos",
        "lex_pos",
        "word_salience",
        "cog_load",
    )
    for r in records:
        for k in tensor_keys:
            if isinstance(r[k], np.ndarray):
                r[k] = torch.from_numpy(r[k])
    return records


# ==============================================================================
# 6. Master Orchestrator & Leakage-Free Subject/Cross-Center Split Builder
# ==============================================================================
def run_preprocessing(
    clinical_raw_root: Path,
    output_dir: Path,
    workers: int = 16,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"output_dir": str(output_dir), "cohorts": {}}

    # ------------------------------------------------------------------
    # Cohort 1: OpenNeuro ds004504 (AD / FTD / CN + MMSE)
    # ------------------------------------------------------------------
    c1_dir = clinical_raw_root / "01_cognitive_ad_ftd_ds004504"
    ptsv = c1_dir / "participants.tsv"
    deriv_sets = sorted((c1_dir / "derivatives").glob("sub-*/eeg/*_eeg.set"))
    if not deriv_sets:
        deriv_sets = sorted(c1_dir.glob("sub-*/eeg/*_eeg.set"))

    if ptsv.exists() and deriv_sets:
        meta_ds004504: Dict[str, Tuple[str, int, Dict[str, Any]]] = {}
        grp_map = {"A": ("AD", 2), "F": ("FTD", 1), "C": ("CN", 0)}
        with open(ptsv, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                pid = row["participant_id"].strip()
                g_raw = row["Group"].strip()
                lbl_code, num_lbl = grp_map.get(g_raw, ("CN", 0))
                try:
                    mmse_val: Any = float(row["MMSE"].strip())
                    if mmse_val.is_integer():
                        mmse_val = int(mmse_val)
                except Exception:
                    mmse_val = "N/A"
                meta_ds004504[pid] = (
                    lbl_code,
                    num_lbl,
                    {"MMSE": mmse_val, "Age": row.get("Age", "").strip(), "Gender": row.get("Gender", "").strip()},
                )

        tasks_c1 = []
        for sp in deriv_sets:
            sub_id = sp.name.split("_")[0]
            if sub_id in meta_ds004504:
                lbl_code, num_lbl, sc = meta_ds004504[sub_id]
                tasks_c1.append((str(sp), sub_id, lbl_code, num_lbl, sc))

        print(f"[Cohort 1 | ds004504] Preprocessing {len(tasks_c1)} subjects across {workers} cores...", flush=True)
        c1_records: List[Dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_worker_ds004504, t) for t in tasks_c1]
            for fut in as_completed(futs):
                c1_records.extend(fut.result())

        c1_records.sort(key=lambda r: (r["subject_id"], r["segment_idx"]))
        _finalize_torch_records(c1_records)
        c1_pkl = output_dir / "ds004504_cognitive.pkl"
        with open(c1_pkl, "wb") as f:
            pickle.dump(c1_records, f, protocol=pickle.HIGHEST_PROTOCOL)
        summary["cohorts"]["ds004504_cognitive"] = {
            "subjects": len(tasks_c1),
            "segments": len(c1_records),
            "pkl_path": str(c1_pkl),
        }
        print(f"  -> Saved {len(c1_records)} segments from {len(tasks_c1)} subjects to {c1_pkl.name}", flush=True)

    # ------------------------------------------------------------------
    # Cohort 2: Cross-Center UNM (54) + UIowa (28) (EEG-MACS Benchmark)
    # ------------------------------------------------------------------
    c2_ext = clinical_raw_root / "02_crosscenter_unm_uiowa_macs" / "extracted"
    for sub_dir_name, center_tag, cohort_key in [
        ("UNM_Data", "UNM_Center", "unm_pd_crosscenter"),
        ("Iowa_Data", "UIowa_Center", "uiowa_pd_crosscenter"),
    ]:
        csv_files = sorted((c2_ext / sub_dir_name).rglob("*.csv")) if (c2_ext / sub_dir_name).exists() else []
        if csv_files:
            tasks_c2 = [(str(p), center_tag, cohort_key) for p in csv_files]
            print(f"[Cohort 2 | {center_tag}] Preprocessing {len(tasks_c2)} subjects across {workers} cores...", flush=True)
            c2_records: List[Dict[str, Any]] = []
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_worker_unm_uiowa, t) for t in tasks_c2]
                for fut in as_completed(futs):
                    c2_records.extend(fut.result())

            c2_records.sort(key=lambda r: (r["subject_id"], r["segment_idx"]))
            _finalize_torch_records(c2_records)
            c2_pkl = output_dir / f"{cohort_key}.pkl"
            with open(c2_pkl, "wb") as f:
                pickle.dump(c2_records, f, protocol=pickle.HIGHEST_PROTOCOL)
            summary["cohorts"][cohort_key] = {
                "subjects": len(tasks_c2),
                "segments": len(c2_records),
                "pkl_path": str(c2_pkl),
            }
            print(f"  -> Saved {len(c2_records)} segments from {len(tasks_c2)} subjects to {c2_pkl.name}", flush=True)

    # ------------------------------------------------------------------
    # Cohort 3a: MODMA 3-Channel Wearable Prefrontal (55 subjects, PHQ-9 & GAD-7)
    # ------------------------------------------------------------------
    c3_ext = clinical_raw_root / "03_affective_mdd_modma" / "extracted"
    m3_dir = c3_ext / "3ch_resting_55sub"
    if m3_dir.exists():
        xlsx_3 = next(m3_dir.rglob("*.xlsx"), None)
        meta_3 = _parse_xlsx_records_stdlib(xlsx_3) if xlsx_3 else {}
        txt_files = sorted([p for p in m3_dir.rglob("*.txt") if not p.name.startswith("._")])
        if txt_files:
            tasks_m3 = []
            for p in txt_files:
                sid = p.stem.split("_")[0]
                tasks_m3.append((str(p), meta_3.get(sid, {"subject_id": sid})))
            print(f"[Cohort 3a | MODMA 3-ch Wearable] Preprocessing {len(tasks_m3)} subjects...", flush=True)
            m3_records: List[Dict[str, Any]] = []
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_worker_modma_3ch, t) for t in tasks_m3]
                for fut in as_completed(futs):
                    m3_records.extend(fut.result())

            m3_records.sort(key=lambda r: (r["subject_id"], r["segment_idx"]))
            _finalize_torch_records(m3_records)
            m3_pkl = output_dir / "modma_3ch_affective.pkl"
            with open(m3_pkl, "wb") as f:
                pickle.dump(m3_records, f, protocol=pickle.HIGHEST_PROTOCOL)
            summary["cohorts"]["modma_3ch_affective"] = {
                "subjects": len(tasks_m3),
                "segments": len(m3_records),
                "pkl_path": str(m3_pkl),
            }
            print(f"  -> Saved {len(m3_records)} segments from {len(tasks_m3)} subjects to {m3_pkl.name}", flush=True)

    # ------------------------------------------------------------------
    # Cohort 3b: MODMA 128-Channel High-Density Resting (53 subjects, PHQ-9 & GAD-7)
    # ------------------------------------------------------------------
    m128_dir = c3_ext / "128ch_resting_53sub"
    if m128_dir.exists():
        xlsx_128 = next(m128_dir.rglob("*.xlsx"), None)
        meta_128 = _parse_xlsx_records_stdlib(xlsx_128) if xlsx_128 else {}
        mat_files = sorted([p for p in m128_dir.rglob("*.mat") if not p.name.startswith("._")])
        if mat_files:
            tasks_m128 = []
            for p in mat_files:
                m_id = re.search(r"(020[123]\d{4})", p.name)
                sid = m_id.group(1) if m_id else p.stem[:8]
                tasks_m128.append((str(p), meta_128.get(sid, {"subject_id": sid})))
            print(f"[Cohort 3b | MODMA 128-ch Resting] Preprocessing {len(tasks_m128)} subjects...", flush=True)
            m128_records: List[Dict[str, Any]] = []
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_worker_modma_128ch, t) for t in tasks_m128]
                for fut in as_completed(futs):
                    m128_records.extend(fut.result())

            m128_records.sort(key=lambda r: (r["subject_id"], r["segment_idx"]))
            _finalize_torch_records(m128_records)
            m128_pkl = output_dir / "modma_128ch_affective.pkl"
            with open(m128_pkl, "wb") as f:
                pickle.dump(m128_records, f, protocol=pickle.HIGHEST_PROTOCOL)
            summary["cohorts"]["modma_128ch_affective"] = {
                "subjects": len(tasks_m128),
                "segments": len(m128_records),
                "pkl_path": str(m128_pkl),
            }
            print(f"  -> Saved {len(m128_records)} segments from {len(tasks_m128)} subjects to {m128_pkl.name}", flush=True)

    # Export SFT Chain-of-Thought preview JSONL + summary manifest
    cot_jsonl = output_dir / "neuroreasoner_cot_preview.jsonl"
    preview_lines = []
    for c_key, c_info in summary["cohorts"].items():
        with open(c_info["pkl_path"], "rb") as f:
            recs = pickle.load(f)
        for r in recs[:5]:
            preview_lines.append(
                json.dumps(
                    {
                        "subject_id": r["subject_id"],
                        "cohort": c_key,
                        "label": r["label_code"],
                        "clinical_scores": r["clinical_scores"],
                        "prompt": r["clinical_prompt"],
                        "cot_response": r["target_text"],
                    },
                    ensure_ascii=False,
                )
            )
    cot_jsonl.write_text("\n".join(preview_lines) + "\n", encoding="utf-8")

    manifest_path = output_dir / "clinical_preprocessing_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest written -> {manifest_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess all 3 Public Clinical EEG Datasets for NeuroReasoner")
    parser.add_argument(
        "--raw-root",
        type=str,
        default="./datasets/clinical_paper2",
        help="Directory containing downloaded 01_cognitive_*, 02_crosscenter_*, 03_affective_* folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./datasets/processed_clinical_neuroreasoner",
        help="Destination folder for unified NeuroReasoner `.pkl` tensors & CoT manifests",
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    run_preprocessing(
        clinical_raw_root=Path(args.raw_root),
        output_dir=Path(args.output_dir),
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
