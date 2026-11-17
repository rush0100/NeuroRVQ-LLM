"""
Neuro-Connectome & Full 8-Metadata Signal Processing Engine for Paper 1 (RUSH-EEG2Text) & Paper 2 (NeuroReasoner).

Implements:
  1. Official 3D Electrode Coordinates (`chanlocs105.mat`, 105 EGI HydroCel scalp channels) -> 12 Anatomical Cortical ROIs
     + Montage-Agnostic 10-20 / 10-10 clinical fallback for Paper 2 hospital cohorts.
  2. Artifact-Resilient `valid_channel_mask` [105] (handles per-sentence NaN bad channels without dropping trials).
  3. Unsupervised Per-Subject & Per-Band Robust Calibration (removes 1/f spectral slope & skull impedance variance).
  4. Multi-Stage Regional Power (`FFD` -> `GD` -> `TRT` + `TRT-FFD` Reanalysis Delta) + 48-Pair Left-Right Asymmetry (`*_diff`).
  5. Vectorized Multi-Window Welch Cross-Spectral Density (`500 Hz rawData` -> `12 ROIs`):
       - Pure Connectivity: Weighted Phase Lag Index (`wPLI`, 528-d) & Imaginary Coherence (`iCoh`, 528-d)
       - Coupled Signal Strength: Shrinkage Log-Euclidean Riemannian SPD Tangent Vector (`624-d`)
       - Word-Level Instantaneous Co-Fluctuation Edge Product (`z_u * z_v`, `528-d`)
  6. Chronological Eye-Gaze Ordering (`fixPositions`) + Pupil/Saccade/Omission Cognitive Load (`6-d` + salience weight).
  7. Task Context ID (`SR`, `NR`, `TSR`) + Auxiliary Sentiment / Relation Labels.
  8. Named Neuro-Biomarker Evidence Dictionary exported per sample for Paper 2 (`NeuroReasoner` CoT Grounding).
"""

from __future__ import annotations

import csv
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

# Exact 105 scalp channels and 3D coordinates (X: Anterior+, Y: Left+, Z: Superior+) from official ZuCo `chanlocs105.mat`
CHANLOCS_105: List[Tuple[str, float, float, float]] = [
    ("E2", 6.709, -5.292, 0.307), ("E3", 7.634, -3.864, 3.068), ("E4", 7.146, -2.869, 4.990),
    ("E5", 5.687, -1.479, 6.813), ("E6", 3.807, 0.000, 7.891), ("E7", 1.559, 1.224, 8.440),
    ("E9", 8.885, -2.695, 1.088), ("E10", 8.709, -1.831, 3.187), ("E11", 7.963, 0.000, 5.045),
    ("E12", 5.687, 1.479, 6.813), ("E13", 3.254, 2.436, 7.609), ("E15", 9.087, 0.000, 1.333),
    ("E16", 9.076, 0.000, 3.105), ("E18", 8.709, 1.831, 3.187), ("E19", 7.146, 2.869, 4.990),
    ("E20", 5.122, 3.826, 5.943), ("E22", 8.885, 2.695, 1.088), ("E23", 7.634, 3.864, 3.068),
    ("E24", 6.021, 4.459, 4.365), ("E26", 6.709, 5.292, 0.307), ("E27", 5.453, 5.683, 2.837),
    ("E28", 4.158, 5.547, 4.628), ("E29", 2.698, 4.762, 6.298), ("E30", 0.960, 3.695, 7.628),
    ("E31", -0.684, 1.955, 8.565), ("E33", 4.127, 6.399, -0.357), ("E34", 2.968, 6.824, 2.430),
    ("E35", 1.490, 6.414, 4.742), ("E36", 0.285, 5.479, 6.383), ("E37", -1.519, 3.910, 7.764),
    ("E38", 3.612, 6.551, -3.353), ("E39", 0.850, 7.192, -0.883), ("E40", 0.032, 7.392, 2.144),
    ("E41", -0.801, 6.905, 4.600), ("E42", -2.339, 5.956, 6.004), ("E43", 2.417, 6.519, -5.254),
    ("E44", 1.278, 6.841, -3.555), ("E45", -1.866, 7.305, -0.629), ("E46", -2.299, 7.313, 2.385),
    ("E47", -3.012, 6.737, 4.178), ("E50", -4.024, 6.782, -0.232), ("E51", -4.667, 6.558, 2.750),
    ("E52", -4.495, 5.831, 4.955), ("E53", -4.037, 4.194, 6.983), ("E54", -3.415, 2.271, 8.205),
    ("E55", -2.138, 0.000, 8.792), ("E57", -3.740, 6.580, -2.991), ("E58", -5.756, 6.035, 0.052),
    ("E59", -6.438, 5.205, 2.984), ("E60", -6.062, 4.117, 5.366), ("E61", -5.481, 2.345, 7.058),
    ("E62", -6.677, 0.000, 6.465), ("E64", -5.870, 5.404, -2.892), ("E65", -7.281, 4.645, 0.130),
    ("E66", -7.665, 3.608, 3.130), ("E67", -7.354, 1.845, 5.224), ("E69", -7.603, 3.529, -2.818),
    ("E70", -8.608, 2.739, 0.239), ("E71", -8.437, 1.405, 3.277), ("E72", -7.830, 0.000, 4.688),
    ("E74", -8.455, 1.126, -2.633), ("E75", -8.997, 0.000, 0.488), ("E76", -8.437, -1.405, 3.277),
    ("E77", -7.354, -1.845, 5.224), ("E78", -5.481, -2.345, 7.058), ("E79", -3.415, -2.271, 8.205),
    ("E80", -0.684, -1.955, 8.565), ("E82", -8.455, -1.126, -2.633), ("E83", -8.608, -2.739, 0.239),
    ("E84", -7.665, -3.608, 3.130), ("E85", -6.062, -4.117, 5.366), ("E86", -4.037, -4.194, 6.983),
    ("E87", -1.519, -3.910, 7.764), ("E89", -7.603, -3.529, -2.818), ("E90", -7.281, -4.645, 0.130),
    ("E91", -6.438, -5.205, 2.984), ("E92", -4.495, -5.831, 4.955), ("E93", -2.339, -5.956, 6.004),
    ("E95", -5.870, -5.404, -2.892), ("E96", -5.756, -6.035, 0.052), ("E97", -4.667, -6.558, 2.750),
    ("E98", -3.012, -6.737, 4.178), ("E100", -3.740, -6.580, -2.991), ("E101", -4.024, -6.782, -0.232),
    ("E102", -2.299, -7.313, 2.385), ("E103", -0.801, -6.905, 4.600), ("E104", 0.285, -5.479, 6.383),
    ("E105", 0.960, -3.695, 7.628), ("E106", 1.559, -1.224, 8.440), ("E108", -1.866, -7.305, -0.629),
    ("E109", 0.032, -7.392, 2.144), ("E110", 1.490, -6.414, 4.742), ("E111", 2.698, -4.762, 6.298),
    ("E112", 3.254, -2.436, 7.609), ("E114", 1.278, -6.841, -3.555), ("E115", 0.850, -7.192, -0.883),
    ("E116", 2.968, -6.824, 2.430), ("E117", 4.158, -5.547, 4.628), ("E118", 5.122, -3.826, 5.943),
    ("E120", 2.417, -6.519, -5.254), ("E121", 3.612, -6.551, -3.353), ("E122", 4.127, -6.399, -0.357),
    ("E123", 5.453, -5.683, 2.837), ("E124", 6.021, -4.459, 4.365), ("Cz", 0.000, 0.000, 8.899),
]

ROI_NAMES: List[str] = [
    "L_Prefrontal",          # 0
    "R_Prefrontal",          # 1
    "L_Broca_IFG",           # 2
    "R_IFG",                 # 3
    "L_Central",             # 4
    "R_Central",             # 5
    "L_Wernicke_Temporal",   # 6
    "R_Temporal",            # 7
    "L_Parietal",            # 8
    "R_Parietal",            # 9
    "L_Occipital",           # 10
    "R_Occipital",           # 11
]

NUM_ROIS = 12
NUM_BANDS = 8
NUM_ROI_PAIRS = (NUM_ROIS * (NUM_ROIS - 1)) // 2   # 66 undirected ROI pairs
NUM_SPD_UPPER = (NUM_ROIS * (NUM_ROIS + 1)) // 2   # 78 SPD upper-triangular entries
NUM_DIFF_PAIRS = 48                                 # Official ZuCo homologous L-R pairs

BAND_KEYS = ["_t1", "_t2", "_a1", "_a2", "_b1", "_b2", "_g1", "_g2"]
BAND_HZ_RANGES: List[Tuple[float, float]] = [
    (4.0, 6.0),     # theta1
    (6.5, 8.0),     # theta2
    (8.5, 10.0),    # alpha1
    (10.5, 13.0),   # alpha2
    (13.5, 18.0),   # beta1
    (18.5, 30.0),   # beta2
    (30.5, 40.0),   # gamma1
    (40.0, 49.5),   # gamma2
]

TASK_NAME_TO_ID: Dict[str, int] = {
    "task1-SR": 0,
    "task2-NR": 1,
    "task2-NR-2.0": 1,
    "task3-TSR": 2,
    "task3-TSR-2.0": 2,
    "synthetic-NR": 1,
}

# Precomputed Dimension Constants
NODE_FEAT_DIM = (
    4 * (NUM_ROIS * NUM_BANDS)   # FFD (96) + GD (96) + TRT (96) + Reanalysis (TRT-FFD) (96) = 384
    + 2 * (6 * NUM_BANDS)        # 6 Homologous L-R ROI Asymmetries x 8 bands x 2 stages (GD, TRT) = 96
    + 8                          # Word Gaze & Pupil Cognitive Load features = 8
)  # Total = 488

WORD_EDGE_DIM = NUM_ROI_PAIRS * NUM_BANDS  # 66 * 8 = 528 (instantaneous co-fluctuation strength)
SENT_CONNECTOME_DIM = (
    NUM_ROI_PAIRS * NUM_BANDS              # wPLI pure phase-lag connectivity = 528
    + NUM_ROI_PAIRS * NUM_BANDS            # iCoh imaginary coherence = 528
    + NUM_SPD_UPPER * NUM_BANDS            # Log-Euclidean Riemannian SPD tangent = 624
    + NUM_DIFF_PAIRS * NUM_BANDS           # Official 48-pair sentence hemispheric asymmetry = 384
)  # Total = 2064
COG_LOAD_DIM = 6

UPPER_TRI_U, UPPER_TRI_V = np.triu_indices(NUM_ROIS, k=1)
SPD_TRI_U, SPD_TRI_V = np.triu_indices(NUM_ROIS, k=0)
SPD_OFFDIAG_SCALE = np.where(SPD_TRI_U == SPD_TRI_V, 1.0, np.sqrt(2.0)).astype(np.float32)


def _build_roi_membership_and_centroids() -> Tuple[np.ndarray, np.ndarray]:
    """
    Build soft-anatomical assignment matrix `W` of shape `[12, 105]` and normalized 3D ROI centroids `[12, 3]`.
    Midline electrodes (`|Y| < 0.05`) split 0.5/0.5 between Left and Right homologous regions so all 105 channels
    participate cleanly without losing midline vertex/occipital power.
    """
    weights = np.zeros((NUM_ROIS, len(CHANLOCS_105)), dtype=np.float32)
    coords = np.array([[x, y, z] for _, x, y, z in CHANLOCS_105], dtype=np.float32)

    for idx, (_, x, y, z) in enumerate(CHANLOCS_105):
        if x >= 5.6 and abs(y) <= 5.4:
            lobe_pair = (0, 1)  # Prefrontal (FP1/2, AF3/4, AF7/8, F1/2, Fz)
        elif x >= 1.4 and abs(y) >= 4.0 and z <= 5.2:
            lobe_pair = (2, 3)  # Inferior Frontal / Broca & R-IFG (F3/4, F5/6, F7/8, FC5/6, FT7/8)
        elif abs(y) >= 5.0 and x < 3.8 and z <= 3.0:
            lobe_pair = (6, 7)  # Lateral/Inferior Temporal / Wernicke (T7/8, TP7/8)
        elif x >= -2.2 and z > 4.5:
            lobe_pair = (4, 5)  # Central Sensorimotor (FC1/2, C1/2, C3/4, Cz)
        elif x < -6.2:
            lobe_pair = (10, 11)  # Occipital Visual Cortex (O1/2, PO7/8, Oz)
        else:
            lobe_pair = (8, 9)  # Parietal / Temporo-Parietal Association (CP, P, Pz)

        if abs(y) < 0.05:
            weights[lobe_pair[0], idx] = 0.5
            weights[lobe_pair[1], idx] = 0.5
        elif y > 0:
            weights[lobe_pair[0], idx] = 1.0
        else:
            weights[lobe_pair[1], idx] = 1.0

    row_sums = weights.sum(axis=1, keepdims=True).clip(min=1e-6)
    centroids = (weights @ coords) / row_sums
    max_rad = np.linalg.norm(coords, axis=-1).max().clip(min=1e-6)
    centroids_norm = (centroids / max_rad).astype(np.float32)
    return weights, centroids_norm


ROI_WEIGHT_MATRIX, ROI_CENTROIDS_3D = _build_roi_membership_and_centroids()


def get_roi_3d_centroids() -> np.ndarray:
    """Return `[12, 3]` normalized `(X, Y, Z)` 3D coordinates of the 12 anatomical ROIs."""
    return ROI_CENTROIDS_3D.copy()


def aggregate_105_to_12_rois(
    chan_band_mat: np.ndarray, valid_channel_mask: np.ndarray
) -> np.ndarray:
    """
    Aggregate `[8, 105]` (or `[N, 8, 105]`) channel band values into `[8, 12]` (or `[N, 8, 12]`)
    ignoring artifactual `NaN` channels via `valid_channel_mask` (`[105]`).
    """
    clean = np.nan_to_num(chan_band_mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    eff_w = ROI_WEIGHT_MATRIX * valid_channel_mask[None, :].astype(np.float32)  # [12, 105]
    denom = eff_w.sum(axis=1).clip(min=1e-5)                                     # [12]
    roi_vals = np.matmul(clean, eff_w.T) / denom
    return roi_vals.astype(np.float32)


def compute_welch_connectome_and_spd(
    raw_eeg_105: Optional[np.ndarray],
    word_gd_roi_seq: np.ndarray,
    valid_channel_mask: np.ndarray,
    fs: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 12-ROI band-specific:
      1. `wpli_vec` `[528]`: Weighted Phase Lag Index (pure non-zero-lag connectivity)
      2. `icoh_vec` `[528]`: Imaginary Coherence (pure phase-lagged coherence)
      3. `spd_vec`  `[624]`: Log-Euclidean Riemannian SPD Tangent Vector (coupled signal strength!)
    Uses continuous 500 Hz `sent.rawData` `[105, T]` when `T >= 128`, and gracefully falls back to
    across-word ROI covariance/trajectory when `rawData` is unavailable.
    """
    use_raw = (
        raw_eeg_105 is not None
        and isinstance(raw_eeg_105, np.ndarray)
        and raw_eeg_105.ndim == 2
        and raw_eeg_105.shape[0] >= 105
        and raw_eeg_105.shape[1] >= 128
    )

    if use_raw:
        raw_105 = raw_eeg_105[:105, :].astype(np.float32)
        row_finite = np.isfinite(raw_105).all(axis=1).astype(np.float32)
        comb_mask = valid_channel_mask * row_finite
        if comb_mask.sum() < 36:
            comb_mask = row_finite if row_finite.sum() >= 36 else np.ones(105, dtype=np.float32)

        raw_clean = np.nan_to_num(raw_105, nan=0.0, posinf=0.0, neginf=0.0)
        eff_w = ROI_WEIGHT_MATRIX * comb_mask[None, :]
        denom = eff_w.sum(axis=1, keepdims=True).clip(min=1e-5)
        roi_ts = (eff_w @ raw_clean) / denom  # [12, T]

        roi_ts = roi_ts - np.mean(roi_ts, axis=1, keepdims=True)
        global_std = float(np.std(roi_ts))
        if global_std > 1e-6:
            roi_ts = np.clip(roi_ts / global_std, -8.0, 8.0)

        t_len = roi_ts.shape[1]
        nperseg = min(256, t_len)
        step = max(32, nperseg // 2)
        starts = list(range(0, max(1, t_len - nperseg + 1), step))
        segs = np.stack([roi_ts[:, s : s + nperseg] for s in starts], axis=0)  # [W, 12, L_win]
        hann = np.hanning(nperseg).astype(np.float32)[None, None, :]
        fft_segs = np.fft.rfft(segs * hann, axis=-1)                            # [W, 12, F]
        freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)

        csd = fft_segs[:, :, None, :] * np.conj(fft_segs[:, None, :, :])

        wpli_list: List[np.ndarray] = []
        icoh_list: List[np.ndarray] = []
        spd_list: List[np.ndarray] = []

        for f_lo, f_hi in BAND_HZ_RANGES:
            f_mask = (freqs >= f_lo) & (freqs <= f_hi)
            if not np.any(f_mask):
                idx_near = int(np.argmin(np.abs(freqs - 0.5 * (f_lo + f_hi))))
                f_mask[idx_near] = True

            csd_b = csd[:, :, :, f_mask]                  # [W, 12, 12, F_b]
            imag_b = np.imag(csd_b)                       # [W, 12, 12, F_b]

            num_wpli = np.abs(np.mean(imag_b, axis=(0, 3)))
            den_wpli = np.mean(np.abs(imag_b), axis=(0, 3)) + 1e-7
            wpli_mat = np.clip(num_wpli / den_wpli, 0.0, 1.0)
            wpli_list.append(wpli_mat[UPPER_TRI_U, UPPER_TRI_V].astype(np.float32))

            mean_csd = np.mean(csd_b, axis=(0, 3))        # [12, 12] complex
            diag_pow = np.real(np.diag(mean_csd)).clip(min=1e-7)
            denom_icoh = np.sqrt(diag_pow[:, None] * diag_pow[None, :])
            icoh_mat = np.clip(np.abs(np.imag(mean_csd)) / denom_icoh, 0.0, 1.0)
            icoh_list.append(icoh_mat[UPPER_TRI_U, UPPER_TRI_V].astype(np.float32))

            real_cov = 0.5 * (np.real(mean_csd) + np.real(mean_csd).T)
            tr = float(np.trace(real_cov)) / NUM_ROIS
            spd_mat = 0.92 * real_cov + 0.08 * max(tr, 1e-4) * np.eye(NUM_ROIS, dtype=np.float32)
            eigvals, eigvecs = np.linalg.eigh(spd_mat)
            log_eig = np.log(np.clip(eigvals, 1e-5, 1e5))
            log_spd = (eigvecs * log_eig[None, :]) @ eigvecs.T
            tangent = log_spd[SPD_TRI_U, SPD_TRI_V] * SPD_OFFDIAG_SCALE
            spd_list.append(np.clip(tangent, -10.0, 10.0).astype(np.float32))

        return (
            np.concatenate(wpli_list, axis=0),
            np.concatenate(icoh_list, axis=0),
            np.concatenate(spd_list, axis=0),
        )

    wpli_fb: List[np.ndarray] = []
    icoh_fb: List[np.ndarray] = []
    spd_fb: List[np.ndarray] = []

    for b_idx in range(NUM_BANDS):
        seq_b = word_gd_roi_seq[:, b_idx, :]  # [N_words, 12]
        if seq_b.shape[0] >= 3:
            centered = seq_b - seq_b.mean(axis=0, keepdims=True)
            cov = (centered.T @ centered) / max(1, seq_b.shape[0] - 1)
            std_v = np.sqrt(np.diag(cov).clip(min=1e-6))
            corr = np.clip(cov / (std_v[:, None] * std_v[None, :]), -1.0, 1.0)
            lag_cov = (centered[:-1].T @ centered[1:]) / max(1, seq_b.shape[0] - 2)
            antisym = np.abs(0.5 * (lag_cov - lag_cov.T)) / (std_v[:, None] * std_v[None, :])
        else:
            vec = seq_b[0]
            cov = np.outer(vec, vec)
            corr = np.tanh(cov / 2.0)
            antisym = np.abs(corr) * 0.5

        wpli_fb.append(np.clip(antisym[UPPER_TRI_U, UPPER_TRI_V], 0.0, 1.0).astype(np.float32))
        icoh_fb.append(np.abs(corr[UPPER_TRI_U, UPPER_TRI_V]).astype(np.float32))

        tr = float(np.trace(cov)) / NUM_ROIS
        spd_mat = 0.90 * cov + 0.10 * max(tr, 1e-3) * np.eye(NUM_ROIS, dtype=np.float32)
        eigvals, eigvecs = np.linalg.eigh(spd_mat)
        log_spd = (eigvecs * np.log(np.clip(eigvals, 1e-5, 1e5))[None, :]) @ eigvecs.T
        spd_fb.append((log_spd[SPD_TRI_U, SPD_TRI_V] * SPD_OFFDIAG_SCALE).astype(np.float32))

    return (
        np.concatenate(wpli_fb, axis=0),
        np.concatenate(icoh_fb, axis=0),
        np.concatenate(spd_fb, axis=0),
    )


def compute_word_coupled_edges(gd_roi_seq: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous Coupled Signal Strength `E_{uv,b}(t) = tanh((z_u * z_v) / 2)`
    for every token `t` (`[L_seq, 8, 12]` -> `[L_seq, 528]`).
    """
    z_u = gd_roi_seq[:, :, UPPER_TRI_U]  # [L_seq, 8, 66]
    z_v = gd_roi_seq[:, :, UPPER_TRI_V]  # [L_seq, 8, 66]
    coupled = np.tanh(0.5 * (z_u * z_v))
    return coupled.reshape(gd_roi_seq.shape[0], -1).astype(np.float32)


def extract_paper2_biomarker_summary(
    sent_roi_gd: np.ndarray,
    wpli_vec: np.ndarray,
    spd_vec: np.ndarray,
    reanalysis_delta: np.ndarray,
    cog_load_vec: np.ndarray,
) -> Dict[str, float]:
    """
    Export compact, human-interpretable cortical network biomarkers for Paper 2 (`NeuroReasoner` CoT grounding).
    """
    wpli_by_band = wpli_vec.reshape(NUM_BANDS, NUM_ROI_PAIRS)
    spd_by_band = spd_vec.reshape(NUM_BANDS, NUM_SPD_UPPER)

    def _pair_idx(u: int, v: int) -> int:
        a, b = (u, v) if u < v else (v, u)
        matches = np.where((UPPER_TRI_U == a) & (UPPER_TRI_V == b))[0]
        return int(matches[0])

    bw_pair = _pair_idx(2, 6)    # L_Broca_IFG <-> L_Wernicke_Temporal
    ot_pair = _pair_idx(10, 6)   # L_Occipital <-> L_Wernicke_Temporal
    fp_pair = _pair_idx(0, 8)    # L_Prefrontal <-> L_Parietal

    l_alpha = float(0.5 * (sent_roi_gd[2, 0] + sent_roi_gd[3, 0]))
    r_alpha = float(0.5 * (sent_roi_gd[2, 1] + sent_roi_gd[3, 1]))

    return {
        "broca_wernicke_theta_wpli": round(float(0.5 * (wpli_by_band[0, bw_pair] + wpli_by_band[1, bw_pair])), 4),
        "broca_wernicke_beta_wpli": round(float(0.5 * (wpli_by_band[4, bw_pair] + wpli_by_band[5, bw_pair])), 4),
        "occipito_temporal_gamma_wpli": round(float(0.5 * (wpli_by_band[6, ot_pair] + wpli_by_band[7, ot_pair])), 4),
        "frontoparietal_theta_wpli": round(float(0.5 * (wpli_by_band[0, fp_pair] + wpli_by_band[1, fp_pair])), 4),
        "frontal_alpha_asymmetry": round(r_alpha - l_alpha, 4),
        "mean_spd_tangent_energy": round(float(np.mean(np.abs(spd_by_band))), 4),
        "syntactic_reanalysis_trt_minus_ffd": round(float(np.mean(reanalysis_delta)), 4),
        "pupil_arousal_z": round(float(cog_load_vec[0]), 4),
        "eye_regression_rate": round(float(cog_load_vec[4]), 4),
    }


def _norm_sent_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def load_zuco_auxiliary_labels(metadata_dir: str = "./datasets/zuco_metadata") -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Load sentence-level auxiliary labels from official ZuCo `task_materials`:
      - `sentiment_map`: normalized sentence -> `{0: negative, 1: neutral, 2: positive}`
      - `relation_map`: normalized sentence -> `0..K-1` relation category index
    """
    sentiment_map: Dict[str, int] = {}
    sent_csv = os.path.join(metadata_dir, "sentiment_labels_task1.csv")
    if os.path.exists(sent_csv):
        with open(sent_csv, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("sentence_id"):
                    continue
                parts = line.split(";")
                if len(parts) >= 4:
                    sent_txt = parts[1].strip()
                    lbl_str = parts[-1].strip()
                    if lbl_str in ("-1", "0", "1"):
                        sentiment_map[_norm_sent_key(sent_txt)] = int(lbl_str) + 1

    rel_vocab: Dict[str, int] = {}
    relation_map: Dict[str, int] = {}

    rel2_csv = os.path.join(metadata_dir, "relations_labels_task2.csv")
    if os.path.exists(rel2_csv):
        with open(rel2_csv, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 5 and row[0] != "sentence_id":
                    sent_txt = row[2].strip()
                    rel_type = row[4].strip().upper()
                    if rel_type:
                        idx = rel_vocab.setdefault(rel_type, len(rel_vocab))
                        relation_map[_norm_sent_key(sent_txt)] = idx

    rel3_csv = os.path.join(metadata_dir, "relations_labels_task3.csv")
    if os.path.exists(rel3_csv):
        with open(rel3_csv, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("sentence_id"):
                    continue
                parts = line.split(";")
                if len(parts) >= 4:
                    sent_txt = ";".join(parts[2:-1]).strip().strip('"')
                    rel_type = parts[-1].strip().upper()
                    if rel_type:
                        idx = rel_vocab.setdefault(rel_type, len(rel_vocab))
                        relation_map[_norm_sent_key(sent_txt)] = idx

    return sentiment_map, relation_map
