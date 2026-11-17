"""
Unified Dual-Stream + Full 8-Metadata Preprocessor for ZuCo 1.0 (scipy MAT), ZuCo 2.0 (HDF5 MAT), & Synthetic Smoke Tests.
Parallelized across CPU cores (`ProcessPoolExecutor`) for rapid extraction on NVIDIA DGX Spark (20-core Grace ARM CPU).

Extracts all 8 ZuCo Metadata Dimensions per Trial:
  1. 3D Scalp Coordinates (`chanlocs105.mat`) -> 12 Anatomical Cortical ROIs
  2. Hemispheric Asymmetry (`mean_*_diff`, `GD_*_diff`, `TRT_*_diff` across 48 homologous L-R pairs)
  3. Continuous 500 Hz `sent.rawData` -> Welch `wPLI` (528), `iCoh` (528), & Riemannian `SPD Tangent` (624)
  4. Multi-Stage Reading Windows (`FFD`, `GD`, `TRT`, and `TRT - FFD` syntactic reanalysis delta)
  5. Per-Sentence `valid_channel_mask` [105] (salvages trials with isolated NaN bad channels instead of dropping!)
  6. True Chronological Eye-Gaze Scanpath Order (`fixPositions` + `nFixations` + normalized relative progress `[0, 1]`)
  7. Pupil & Oculomotor Cognitive Load (`pupilsize`, `omissionRate`, `saccade velocity/amplitude`, fixation durations)
  8. Task Context ID (`SR=0, NR=1, TSR=2`) + Auxiliary Sentiment (`0,1,2`) / Relation Category Labels
"""

from __future__ import annotations

import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm

from src.data.neuro_connectome import (
    BAND_KEYS,
    COG_LOAD_DIM,
    NODE_FEAT_DIM,
    NUM_BANDS,
    NUM_DIFF_PAIRS,
    NUM_ROIS,
    SENT_CONNECTOME_DIM,
    TASK_NAME_TO_ID,
    WORD_EDGE_DIM,
    _norm_sent_key,
    aggregate_105_to_12_rois,
    compute_welch_connectome_and_spd,
    compute_word_coupled_edges,
    extract_paper2_biomarker_summary,
    load_zuco_auxiliary_labels,
)


def clean_target_text(text: str) -> str:
    text = text.replace("emp11111ty", "empty").replace("film.1", "film.")
    return " ".join(text.strip().split())


def _scalar_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    arr = np.asarray(val).flatten()
    if arr.size == 0:
        return default
    try:
        v = float(arr[0])
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _first_fixation_pos(val: Any, fallback_idx: int) -> Tuple[float, int]:
    """Return `(first_fixation_order, regression_revisits)` from `word.fixPositions`."""
    if val is None:
        return float(fallback_idx + 1), 0
    arr = np.asarray(val).flatten()
    finite = [float(x) for x in arr if np.isfinite(x) and float(x) > 0]
    if not finite:
        return float(fallback_idx + 1), 0
    return float(min(finite)), max(0, len(finite) - 1)


def _safe_band_vec(arr_like: Any, expected_len: int) -> Optional[np.ndarray]:
    if arr_like is None:
        return None
    arr = np.asarray(arr_like, dtype=np.float32).flatten()
    if arr.size < expected_len:
        return None
    return arr[:expected_len]


def _robust_subject_stats(stack_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-subject median and IQR scale across `[N_obs, ...]` ignoring NaNs."""
    stacked = np.stack(stack_list, axis=0)
    med = np.nanmedian(stacked, axis=0).astype(np.float32)
    q75 = np.nanpercentile(stacked, 75, axis=0).astype(np.float32)
    q25 = np.nanpercentile(stacked, 25, axis=0).astype(np.float32)
    scale = ((q75 - q25) / 1.349).astype(np.float32)
    med = np.nan_to_num(med, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-5), scale, 1.0).astype(np.float32)
    return med, scale


def _norm_with_subject_stats(arr: np.ndarray, med: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (arr - med) / scale
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(z, -6.0, 6.0).astype(np.float32)


def _roi_lr_asymmetry(roi_8x12: np.ndarray) -> np.ndarray:
    """Compute 6 homologous Left-minus-Right ROI asymmetries across 8 bands (`[8, 6]` -> `[48]`)."""
    left = roi_8x12[:, 0::2]   # [8, 6] (L_Prefrontal, L_Broca, L_Central, L_Wernicke, L_Parietal, L_Occipital)
    right = roi_8x12[:, 1::2]  # [8, 6]
    return np.clip(left - right, -6.0, 6.0).reshape(-1).astype(np.float32)


def assemble_dual_stream_subject_records(
    raw_trials: List[Dict[str, Any]],
    subject_id: str,
    task_name: str,
    max_eeg_len: int,
    sentiment_map: Dict[str, int],
    relation_map: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    Two-Pass Per-Subject Calibration & Dual-Stream Feature Assembly:
      Pass 1: Estimate unsupervised per-subject `(channel, band)` & `pupil` baselines across all trials of this subject.
      Pass 2: Calibrate each trial, sort words by true gaze scanpath (`fixPositions`), extract 12-ROI Multi-Stage Node
              features, Welch `wPLI/iCoh/SPD` connectome, word co-fluctuation edges, cognitive load, & auxiliary labels.
    """
    if not raw_trials:
        return []

    # --- PASS 1: Unsupervised Subject Calibration (zero text labels used!) ---
    sent_band_obs: List[np.ndarray] = [t["sent_105"] for t in raw_trials]
    word_gd_obs: List[np.ndarray] = []
    word_ffd_obs: List[np.ndarray] = []
    word_trt_obs: List[np.ndarray] = []
    diff_obs: List[np.ndarray] = [t["sent_diff_48"] for t in raw_trials if t["sent_diff_48"] is not None]
    pupil_obs: List[float] = []

    for t in raw_trials:
        for w in t["words"]:
            word_gd_obs.append(w["gd_105"])
            word_ffd_obs.append(w["ffd_105"])
            word_trt_obs.append(w["trt_105"])
            if w["pupil"] > 0:
                pupil_obs.append(w["pupil"])
        if t["mean_pupil"] > 0:
            pupil_obs.append(t["mean_pupil"])

    sent_med, sent_scale = _robust_subject_stats(sent_band_obs)
    gd_med, gd_scale = _robust_subject_stats(word_gd_obs) if word_gd_obs else (sent_med, sent_scale)
    ffd_med, ffd_scale = _robust_subject_stats(word_ffd_obs) if word_ffd_obs else (gd_med, gd_scale)
    trt_med, trt_scale = _robust_subject_stats(word_trt_obs) if word_trt_obs else (gd_med, gd_scale)

    if diff_obs:
        diff_med, diff_scale = _robust_subject_stats(diff_obs)
    else:
        diff_med = np.zeros((NUM_BANDS, NUM_DIFF_PAIRS), dtype=np.float32)
        diff_scale = np.ones((NUM_BANDS, NUM_DIFF_PAIRS), dtype=np.float32)

    if len(pupil_obs) >= 4:
        p_arr = np.asarray(pupil_obs, dtype=np.float32)
        pupil_med = float(np.median(p_arr))
        pupil_iqr = max(1.0, float(np.percentile(p_arr, 75) - np.percentile(p_arr, 25)) / 1.349)
    else:
        pupil_med, pupil_iqr = 0.0, 1.0

    task_id = TASK_NAME_TO_ID.get(task_name, 1)
    records: List[Dict[str, Any]] = []

    # --- PASS 2: Trial-Level Dual-Stream + 8-Metadata Construction ---
    for t in raw_trials:
        target_text = clean_target_text(t["target_text"])
        if not target_text or len(t["words"]) == 0:
            continue

        valid_mask = t["valid_channel_mask"]  # [105] float32 (1.0 = clean, 0.0 = artifact NaN)
        if float(valid_mask.sum()) < 40.0:
            continue

        # Sort words in TRUE chronological human reading order (`first_fix_pos`, tie-breaking by lexical index)
        words_sorted = sorted(t["words"], key=lambda w: (w["first_fix_pos"], w["lex_idx"]))
        if len(words_sorted) > max_eeg_len - 1:
            words_sorted = words_sorted[: max_eeg_len - 1]

        n_words = len(words_sorted)
        total_lex_words = max(1, int(t["total_words"]))

        # Calibrate sentence-level 105-ch power & aggregate into 12 Anatomical ROIs
        sent_z_105 = _norm_with_subject_stats(t["sent_105"], sent_med, sent_scale)  # [8, 105]
        sent_roi = aggregate_105_to_12_rois(sent_z_105, valid_mask)                 # [8, 12]

        if t["sent_diff_48"] is not None:
            sent_diff_norm = _norm_with_subject_stats(t["sent_diff_48"], diff_med, diff_scale).reshape(-1)  # [384]
        else:
            sent_diff_norm = np.zeros((NUM_BANDS * NUM_DIFF_PAIRS,), dtype=np.float32)

        node_rows: List[np.ndarray] = []
        gd_roi_seq_list: List[np.ndarray] = []
        legacy_840_rows: List[np.ndarray] = []
        rel_pos_list: List[float] = []
        lex_pos_list: List[float] = []
        salience_list: List[float] = []
        reanalysis_deltas: List[np.ndarray] = []
        regressions_count = 0

        for scan_idx, w in enumerate(words_sorted):
            ffd_z = _norm_with_subject_stats(w["ffd_105"], ffd_med, ffd_scale)
            gd_z = _norm_with_subject_stats(w["gd_105"], gd_med, gd_scale)
            trt_z = _norm_with_subject_stats(w["trt_105"], trt_med, trt_scale)

            ffd_roi = aggregate_105_to_12_rois(ffd_z, valid_mask)  # [8, 12]
            gd_roi = aggregate_105_to_12_rois(gd_z, valid_mask)    # [8, 12]
            trt_roi = aggregate_105_to_12_rois(trt_z, valid_mask)  # [8, 12]
            reanalysis = np.clip(trt_roi - ffd_roi, -6.0, 6.0)     # [8, 12] syntactic reanalysis power!

            gd_asym = _roi_lr_asymmetry(gd_roi)                    # [48]
            trt_asym = _roi_lr_asymmetry(trt_roi)                  # [48]

            if w["revisits"] > 0:
                regressions_count += 1

            pupil_z = float(np.clip((w["pupil"] - pupil_med) / pupil_iqr, -4.0, 4.0)) if w["pupil"] > 0 else 0.0
            ffd_ms = np.log1p(max(0.0, w["ffd_ms"])) / 6.5
            gd_ms = np.log1p(max(0.0, w["gd_ms"])) / 6.5
            trt_ms = np.log1p(max(0.0, w["trt_ms"])) / 6.5
            revisit_ratio = min(3.0, float(w["revisits"])) / 3.0
            nfix_norm = min(5.0, float(w["n_fix"])) / 5.0
            scan_prog = float(scan_idx) / float(max(1, n_words - 1)) if n_words > 1 else 0.5
            lex_prog = float(w["lex_idx"]) / float(max(1, total_lex_words - 1)) if total_lex_words > 1 else 0.5

            word_meta_8 = np.array(
                [ffd_ms, gd_ms, trt_ms, revisit_ratio, nfix_norm, pupil_z / 4.0, scan_prog, lex_prog - scan_prog],
                dtype=np.float32,
            )

            node_vec = np.concatenate(
                [
                    ffd_roi.reshape(-1),       # 96
                    gd_roi.reshape(-1),        # 96
                    trt_roi.reshape(-1),       # 96
                    reanalysis.reshape(-1),    # 96
                    gd_asym,                   # 48
                    trt_asym,                  # 48
                    word_meta_8,               # 8
                ],
                axis=0,
            )  # [488]
            node_rows.append(node_vec)
            gd_roi_seq_list.append(gd_roi)
            legacy_840_rows.append((gd_z * valid_mask[None, :]).reshape(-1))
            reanalysis_deltas.append(reanalysis)

            # Relative temporal scanpath progress in [0.0, 0.92], reserving 1.0 for holistic sentence token!
            rel_pos_list.append(0.92 * scan_prog)
            lex_pos_list.append(0.92 * lex_prog)
            salience_list.append(1.0 + 0.35 * min(3.0, float(w["n_fix"]) - 1.0) + 0.15 * max(0.0, pupil_z))

        # Append Compulsory Holistic Sentence Slot as the final active token (`rel_pos = 1.0`)
        mean_reanalysis = np.mean(np.stack(reanalysis_deltas, axis=0), axis=0) if reanalysis_deltas else np.zeros_like(sent_roi)
        sent_asym = _roi_lr_asymmetry(sent_roi)
        sent_pupil_z = float(np.clip((t["mean_pupil"] - pupil_med) / pupil_iqr, -4.0, 4.0)) if t["mean_pupil"] > 0 else 0.0
        omission_rate = float(np.clip(t["omission_rate"], 0.0, 1.0))
        reg_rate = float(regressions_count) / float(max(1, n_words))
        valid_chan_ratio = float(valid_mask.mean())

        sent_meta_8 = np.array(
            [
                1.0,
                omission_rate,
                reg_rate,
                sent_pupil_z / 4.0,
                valid_chan_ratio,
                np.log1p(float(t["sacc_vel"])) / 8.0,
                np.log1p(float(t["sacc_amp"])) / 5.0,
                0.0,
            ],
            dtype=np.float32,
        )
        sent_node_vec = np.concatenate(
            [
                sent_roi.reshape(-1),
                sent_roi.reshape(-1),
                sent_roi.reshape(-1),
                mean_reanalysis.reshape(-1),
                sent_asym,
                sent_asym,
                sent_meta_8,
            ],
            axis=0,
        )  # [488]
        node_rows.append(sent_node_vec)
        gd_roi_seq_list.append(sent_roi)
        legacy_840_rows.append((sent_z_105 * valid_mask[None, :]).reshape(-1))
        rel_pos_list.append(1.0)
        lex_pos_list.append(1.0)
        salience_list.append(1.25)

        seq_len = len(node_rows)
        gd_roi_seq = np.stack(gd_roi_seq_list, axis=0)              # [seq_len, 8, 12]
        word_edges_active = compute_word_coupled_edges(gd_roi_seq)  # [seq_len, 528]

        # Compute Welch wPLI (528), iCoh (528), & Riemannian SPD Tangent (624) from continuous 500 Hz rawData!
        wpli_vec, icoh_vec, spd_vec = compute_welch_connectome_and_spd(
            raw_eeg_105=t["raw_eeg_105"],
            word_gd_roi_seq=gd_roi_seq[:-1] if seq_len > 1 else gd_roi_seq,
            valid_channel_mask=valid_mask,
            fs=500.0,
        )
        sent_connectome = np.concatenate([wpli_vec, icoh_vec, spd_vec, sent_diff_norm], axis=0).astype(np.float32)  # [2064]

        cog_load_vec = np.array(
            [
                sent_pupil_z / 4.0,
                omission_rate,
                reg_rate,
                np.log1p(float(t["sacc_vel"])) / 8.0,
                np.log1p(float(t["sacc_amp"])) / 5.0,
                valid_chan_ratio,
            ],
            dtype=np.float32,
        )

        # Pad sequences to `max_eeg_len`
        padded_nodes = np.zeros((max_eeg_len, NODE_FEAT_DIM), dtype=np.float32)
        padded_nodes[:seq_len] = np.stack(node_rows, axis=0)

        padded_edges = np.zeros((max_eeg_len, WORD_EDGE_DIM), dtype=np.float32)
        padded_edges[:seq_len] = word_edges_active

        padded_legacy = np.zeros((max_eeg_len, 840), dtype=np.float32)
        padded_legacy[:seq_len] = np.stack(legacy_840_rows, axis=0)

        eeg_attn_mask = np.zeros((max_eeg_len,), dtype=np.int64)
        eeg_attn_mask[:seq_len] = 1

        rel_pos = np.zeros((max_eeg_len,), dtype=np.float32)
        rel_pos[:seq_len] = np.asarray(rel_pos_list, dtype=np.float32)

        lex_pos = np.zeros((max_eeg_len,), dtype=np.float32)
        lex_pos[:seq_len] = np.asarray(lex_pos_list, dtype=np.float32)

        salience = np.zeros((max_eeg_len,), dtype=np.float32)
        salience[:seq_len] = np.asarray(salience_list, dtype=np.float32)

        norm_key = _norm_sent_key(target_text)
        sentiment_label = sentiment_map.get(norm_key, -100)
        relation_label = relation_map.get(norm_key, -100)

        biomarkers = extract_paper2_biomarker_summary(
            sent_roi_gd=sent_roi,
            wpli_vec=wpli_vec,
            spd_vec=spd_vec,
            reanalysis_delta=mean_reanalysis,
            cog_load_vec=cog_load_vec,
        )

        records.append(
            {
                "subject_id": subject_id,
                "task_name": task_name,
                "task_id": task_id,
                "sentence_idx": int(t["sentence_idx"]),
                "target_text": target_text,
                "selected_words": [w["content"] for w in words_sorted],
                "node_features": torch.from_numpy(padded_nodes),          # [max_len, 488]
                "word_coupled_edges": torch.from_numpy(padded_edges),     # [max_len, 528]
                "sent_connectome": torch.from_numpy(sent_connectome),     # [2064]
                "eeg_features": torch.from_numpy(padded_legacy),          # [max_len, 840] legacy compat
                "eeg_attn_mask": torch.from_numpy(eeg_attn_mask),         # [max_len]
                "rel_pos": torch.from_numpy(rel_pos),                     # [max_len] in [0, 1] (zero length leak!)
                "lex_pos": torch.from_numpy(lex_pos),                     # [max_len] in [0, 1]
                "word_salience": torch.from_numpy(salience),              # [max_len]
                "valid_channel_mask": torch.from_numpy(valid_mask),       # [105]
                "cog_load": torch.from_numpy(cog_load_vec),               # [6]
                "sentiment_label": int(sentiment_label),                  # {0,1,2} or -100
                "relation_label": int(relation_label),                    # 0..K-1 or -100
                "biomarker_evidence": biomarkers,
                "seq_len": seq_len,
            }
        )

    return records


def process_zuco_v1_mat(
    mat_path: str,
    task_name: str,
    max_eeg_len: int = 58,
    metadata_dir: str = "./datasets/zuco_metadata",
) -> List[Dict[str, Any]]:
    """Parse a ZuCo 1.0 MATLAB v5 file extracting all 8 metadata sources + raw 500 Hz connectomes."""
    subject_id = os.path.basename(mat_path).split("_")[0].replace("results", "").strip()
    mat_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["sentenceData"]
    sentiment_map, relation_map = load_zuco_auxiliary_labels(metadata_dir)

    raw_trials: List[Dict[str, Any]] = []
    for sent_idx, sent in enumerate(mat_data):
        word_data = getattr(sent, "word", float("nan"))
        if isinstance(word_data, float):
            continue

        # Sentence-level 8 bands [8, 105] + valid_channel_mask [105]
        sent_bands: List[np.ndarray] = []
        valid_105 = np.ones((105,), dtype=np.float32)
        ok_sent = True
        for b in BAND_KEYS:
            arr = _safe_band_vec(getattr(sent, f"mean{b}", None), 105)
            if arr is None:
                ok_sent = False
                break
            valid_105 *= np.isfinite(arr).astype(np.float32)
            sent_bands.append(arr)
        if not ok_sent or valid_105.sum() < 40:
            continue

        # Sentence-level 48 homologous pair differences [8, 48]
        diff_bands: List[np.ndarray] = []
        for b in BAND_KEYS:
            d_arr = _safe_band_vec(getattr(sent, f"mean{b}_diff", None), NUM_DIFF_PAIRS)
            if d_arr is not None:
                diff_bands.append(np.nan_to_num(d_arr, nan=0.0))
        sent_diff_48 = np.stack(diff_bands, axis=0) if len(diff_bands) == NUM_BANDS else None

        # Continuous 500 Hz rawData [105, T]
        raw_eeg = getattr(sent, "rawData", None)
        raw_eeg_arr = np.asarray(raw_eeg, dtype=np.float32) if raw_eeg is not None and not isinstance(raw_eeg, float) else None
        if raw_eeg_arr is not None and raw_eeg_arr.ndim == 2 and raw_eeg_arr.shape[0] != 105 and raw_eeg_arr.shape[1] == 105:
            raw_eeg_arr = raw_eeg_arr.T

        # Sentence-level oculomotor & pupil cognitive load
        omission_rate = _scalar_float(getattr(sent, "omissionRate", 0.0), 0.0)
        sacc_vel = _scalar_float(getattr(sent, "saccMeanVel", 0.0), 0.0)
        sacc_amp = _scalar_float(getattr(sent, "saccMeanAmp", 0.0), 0.0)
        mean_pupil = 0.0
        af = getattr(sent, "allFixations", None)
        if af is not None and not isinstance(af, float):
            ps = getattr(af, "pupilsize", None)
            if ps is not None:
                ps_arr = np.asarray(ps, dtype=np.float32).flatten()
                ps_fin = ps_arr[np.isfinite(ps_arr) & (ps_arr > 0)]
                if ps_fin.size > 0:
                    mean_pupil = float(np.mean(ps_fin))
            if sacc_amp == 0.0 and hasattr(af, "x") and hasattr(af, "y"):
                ax = np.asarray(getattr(af, "x"), dtype=np.float32).flatten()
                ay = np.asarray(getattr(af, "y"), dtype=np.float32).flatten()
                if ax.size >= 2 and ay.size >= 2:
                    sacc_amp = float(np.nanmean(np.hypot(np.diff(ax), np.diff(ay))))

        iterable_words = [word_data] if not hasattr(word_data, "__iter__") else list(word_data)
        parsed_words: List[Dict[str, Any]] = []
        for lex_idx, w in enumerate(iterable_words):
            n_fix = _scalar_float(getattr(w, "nFixations", 0), 0.0)
            if n_fix <= 0:
                continue

            gd_list, ffd_list, trt_list = [], [], []
            ok_w = True
            for b in BAND_KEYS:
                g_v = _safe_band_vec(getattr(w, f"GD{b}", None), 105)
                if g_v is None:
                    ok_w = False
                    break
                f_v = _safe_band_vec(getattr(w, f"FFD{b}", None), 105)
                t_v = _safe_band_vec(getattr(w, f"TRT{b}", None), 105)
                gd_list.append(g_v)
                ffd_list.append(f_v if f_v is not None else g_v)
                trt_list.append(t_v if t_v is not None else g_v)
            if not ok_w:
                continue

            first_pos, revisits = _first_fixation_pos(getattr(w, "fixPositions", None), lex_idx)
            w_pupil = _scalar_float(getattr(w, "meanPupilSize", getattr(w, "GD_pupilsize", 0.0)), 0.0)
            parsed_words.append(
                {
                    "lex_idx": lex_idx,
                    "content": str(getattr(w, "content", "")),
                    "n_fix": n_fix,
                    "first_fix_pos": first_pos,
                    "revisits": revisits,
                    "pupil": w_pupil,
                    "ffd_ms": _scalar_float(getattr(w, "FFD", 0.0), 0.0),
                    "gd_ms": _scalar_float(getattr(w, "GD", 0.0), 0.0),
                    "trt_ms": _scalar_float(getattr(w, "TRT", 0.0), 0.0),
                    "ffd_105": np.stack(ffd_list, axis=0),
                    "gd_105": np.stack(gd_list, axis=0),
                    "trt_105": np.stack(trt_list, axis=0),
                }
            )

        if parsed_words:
            raw_trials.append(
                {
                    "sentence_idx": sent_idx,
                    "target_text": str(getattr(sent, "content", "")),
                    "total_words": len(iterable_words),
                    "sent_105": np.stack(sent_bands, axis=0),
                    "sent_diff_48": sent_diff_48,
                    "valid_channel_mask": valid_105,
                    "raw_eeg_105": raw_eeg_arr,
                    "omission_rate": omission_rate,
                    "sacc_vel": sacc_vel,
                    "sacc_amp": sacc_amp,
                    "mean_pupil": mean_pupil,
                    "words": parsed_words,
                }
            )

    return assemble_dual_stream_subject_records(
        raw_trials=raw_trials,
        subject_id=subject_id,
        task_name=task_name,
        max_eeg_len=max_eeg_len,
        sentiment_map=sentiment_map,
        relation_map=relation_map,
    )


def _h5_read_str(f: h5py.File, ref: Any) -> str:
    arr = np.asarray(f[ref])
    return "".join(chr(int(c)) for c in arr.flatten() if int(c) != 0)


def _h5_deref_array(f: h5py.File, ref: Any) -> Optional[np.ndarray]:
    try:
        obj = f[ref]
        if isinstance(obj, h5py.Dataset):
            return np.asarray(obj, dtype=np.float32)
    except Exception:
        return None
    return None


def process_zuco_v2_mat(
    mat_path: str,
    task_name: str = "task2-NR-2.0",
    max_eeg_len: int = 58,
    metadata_dir: str = "./datasets/zuco_metadata",
) -> List[Dict[str, Any]]:
    """Parse a ZuCo 2.0 MATLAB v7.3 HDF5 file extracting all 8 metadata sources + raw 500 Hz connectomes."""
    subject_id = os.path.basename(mat_path).split("_")[0].replace("results", "").strip()
    if subject_id == "YMH":
        return []  # Corrupted subject excluded per official ZuCo 2.0 & CET-MAE protocol
    sentiment_map, relation_map = load_zuco_auxiliary_labels(metadata_dir)

    raw_trials: List[Dict[str, Any]] = []
    with h5py.File(mat_path, "r") as f:
        sent_data = f["sentenceData"]
        content_refs = sent_data["content"]
        word_refs = sent_data["word"]
        raw_refs = sent_data["rawData"] if "rawData" in sent_data else None
        omiss_refs = sent_data["omissionRate"] if "omissionRate" in sent_data else None
        sacc_vel_refs = sent_data["saccMeanVel"] if "saccMeanVel" in sent_data else None
        sacc_amp_refs = sent_data["saccMeanAmp"] if "saccMeanAmp" in sent_data else None
        mean_band_refs = {b: sent_data[f"mean{b}"] for b in BAND_KEYS}
        diff_band_refs = {b: sent_data[f"mean{b}_diff"] for b in BAND_KEYS if f"mean{b}_diff" in sent_data}

        num_sents = len(content_refs)
        for sent_idx in range(num_sents):
            try:
                target_text = _h5_read_str(f, content_refs[sent_idx][0])
                sent_bands: List[np.ndarray] = []
                valid_105 = np.ones((105,), dtype=np.float32)
                ok_sent = True
                for b in BAND_KEYS:
                    arr = _h5_deref_array(f, mean_band_refs[b][sent_idx][0])
                    v105 = _safe_band_vec(arr, 105)
                    if v105 is None:
                        ok_sent = False
                        break
                    valid_105 *= np.isfinite(v105).astype(np.float32)
                    sent_bands.append(v105)
                if not ok_sent or valid_105.sum() < 40:
                    continue

                diff_bands: List[np.ndarray] = []
                if len(diff_band_refs) == NUM_BANDS:
                    for b in BAND_KEYS:
                        d_arr = _safe_band_vec(_h5_deref_array(f, diff_band_refs[b][sent_idx][0]), NUM_DIFF_PAIRS)
                        if d_arr is not None:
                            diff_bands.append(np.nan_to_num(d_arr, nan=0.0))
                sent_diff_48 = np.stack(diff_bands, axis=0) if len(diff_bands) == NUM_BANDS else None

                raw_eeg_arr = _h5_deref_array(f, raw_refs[sent_idx][0]) if raw_refs is not None else None
                if raw_eeg_arr is not None and raw_eeg_arr.ndim == 2:
                    if raw_eeg_arr.shape[0] != 105 and raw_eeg_arr.shape[1] == 105:
                        raw_eeg_arr = raw_eeg_arr.T

                omission_rate = _scalar_float(_h5_deref_array(f, omiss_refs[sent_idx][0]), 0.0) if omiss_refs is not None else 0.0
                sacc_vel = _scalar_float(_h5_deref_array(f, sacc_vel_refs[sent_idx][0]), 0.0) if sacc_vel_refs is not None else 0.0
                sacc_amp = _scalar_float(_h5_deref_array(f, sacc_amp_refs[sent_idx][0]), 0.0) if sacc_amp_refs is not None else 0.0

                w_group = f[word_refs[sent_idx][0]]
                if not isinstance(w_group, h5py.Group) or "nFixations" not in w_group:
                    continue

                w_contents = w_group["content"]
                n_fix_ds = w_group["nFixations"]
                fix_pos_ds = w_group["fixPositions"] if "fixPositions" in w_group else None
                pupil_ds = w_group["meanPupilSize"] if "meanPupilSize" in w_group else None
                ffd_dur_ds = w_group["FFD"] if "FFD" in w_group else None
                gd_dur_ds = w_group["GD"] if "GD" in w_group else None
                trt_dur_ds = w_group["TRT"] if "TRT" in w_group else None

                gd_refs = {b: w_group[f"GD{b}"] for b in BAND_KEYS}
                ffd_refs = {b: w_group[f"FFD{b}"] for b in BAND_KEYS if f"FFD{b}" in w_group}
                trt_refs = {b: w_group[f"TRT{b}"] for b in BAND_KEYS if f"TRT{b}" in w_group}

                parsed_words: List[Dict[str, Any]] = []
                pupil_vals: List[float] = []
                total_w = len(w_contents)
                for lex_idx in range(total_w):
                    n_fix = _scalar_float(_h5_deref_array(f, n_fix_ds[lex_idx][0]), 0.0)
                    if n_fix <= 0:
                        continue

                    gd_list, ffd_list, trt_list = [], [], []
                    ok_w = True
                    for b in BAND_KEYS:
                        g_v = _safe_band_vec(_h5_deref_array(f, gd_refs[b][lex_idx][0]), 105)
                        if g_v is None:
                            ok_w = False
                            break
                        f_v = _safe_band_vec(_h5_deref_array(f, ffd_refs[b][lex_idx][0]), 105) if b in ffd_refs else None
                        t_v = _safe_band_vec(_h5_deref_array(f, trt_refs[b][lex_idx][0]), 105) if b in trt_refs else None
                        gd_list.append(g_v)
                        ffd_list.append(f_v if f_v is not None else g_v)
                        trt_list.append(t_v if t_v is not None else g_v)
                    if not ok_w:
                        continue

                    fp_arr = _h5_deref_array(f, fix_pos_ds[lex_idx][0]) if fix_pos_ds is not None else None
                    first_pos, revisits = _first_fixation_pos(fp_arr, lex_idx)
                    w_pupil = _scalar_float(_h5_deref_array(f, pupil_ds[lex_idx][0]), 0.0) if pupil_ds is not None else 0.0
                    if w_pupil > 0:
                        pupil_vals.append(w_pupil)

                    parsed_words.append(
                        {
                            "lex_idx": lex_idx,
                            "content": _h5_read_str(f, w_contents[lex_idx][0]),
                            "n_fix": n_fix,
                            "first_fix_pos": first_pos,
                            "revisits": revisits,
                            "pupil": w_pupil,
                            "ffd_ms": _scalar_float(_h5_deref_array(f, ffd_dur_ds[lex_idx][0]), 0.0) if ffd_dur_ds is not None else 0.0,
                            "gd_ms": _scalar_float(_h5_deref_array(f, gd_dur_ds[lex_idx][0]), 0.0) if gd_dur_ds is not None else 0.0,
                            "trt_ms": _scalar_float(_h5_deref_array(f, trt_dur_ds[lex_idx][0]), 0.0) if trt_dur_ds is not None else 0.0,
                            "ffd_105": np.stack(ffd_list, axis=0),
                            "gd_105": np.stack(gd_list, axis=0),
                            "trt_105": np.stack(trt_list, axis=0),
                        }
                    )

                if parsed_words:
                    raw_trials.append(
                        {
                            "sentence_idx": sent_idx,
                            "target_text": target_text,
                            "total_words": total_w,
                            "sent_105": np.stack(sent_bands, axis=0),
                            "sent_diff_48": sent_diff_48,
                            "valid_channel_mask": valid_105,
                            "raw_eeg_105": raw_eeg_arr,
                            "omission_rate": omission_rate,
                            "sacc_vel": sacc_vel,
                            "sacc_amp": sacc_amp,
                            "mean_pupil": float(np.mean(pupil_vals)) if pupil_vals else 0.0,
                            "words": parsed_words,
                        }
                    )
            except Exception:
                continue

    return assemble_dual_stream_subject_records(
        raw_trials=raw_trials,
        subject_id=subject_id,
        task_name=task_name,
        max_eeg_len=max_eeg_len,
        sentiment_map=sentiment_map,
        relation_map=relation_map,
    )


def _worker_v1(args: Tuple[str, str, int, str, str]) -> Tuple[str, str]:
    mf, task_name, max_len, meta_dir, tmp_dir = args
    recs = process_zuco_v1_mat(mf, task_name=task_name, max_eeg_len=max_len, metadata_dir=meta_dir)
    out_tmp = os.path.join(tmp_dir, os.path.basename(mf) + ".pkl")
    with open(out_tmp, "wb") as f:
        pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
    return mf, out_tmp


def _worker_v2(args: Tuple[str, str, int, str, str]) -> Tuple[str, str]:
    mf, task_name, max_len, meta_dir, tmp_dir = args
    recs = process_zuco_v2_mat(mf, task_name=task_name, max_eeg_len=max_len, metadata_dir=meta_dir)
    out_tmp = os.path.join(tmp_dir, os.path.basename(mf) + ".pkl")
    with open(out_tmp, "wb") as f:
        pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
    return mf, out_tmp


def generate_synthetic_zuco(
    output_dir: str,
    num_subjects: int = 5,
    sentences_per_subject: int = 24,
    max_eeg_len: int = 32,
    seed: int = 42,
) -> str:
    """Create a deterministic Dual-Stream + 8-Metadata synthetic dataset for rapid smoke tests."""
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    corpus = [
        ("Brain computer interfaces translate neural oscillations into natural language.", 2),
        ("Residual vector quantization suppresses session specific electrode noise.", 1),
        ("Large language models decode open vocabulary sentences without teacher forcing.", 2),
        ("Harbin Institute of Technology Shenzhen advances neural engineering research.", 2),
        ("Unified memory architecture accelerates multimodal brain foundation models.", 2),
        ("Contrastive masked autoencoders align cortical rhythms with semantic tokens.", 1),
        ("Cross subject generalization requires invariant discrete neural representations.", 1),
        ("Slow, silly and unintentionally hilarious.", 0),
    ]

    prototypes = {
        sent: rng.normal(0.0, 1.0, size=(NUM_BANDS, 105)).astype(np.float32)
        for sent, _ in corpus
    }
    phase_offsets = {
        sent: rng.uniform(0.0, np.pi, size=(12,)).astype(np.float32)
        for sent, _ in corpus
    }

    all_records: List[Dict[str, Any]] = []
    t_axis = np.linspace(0.0, 2.0, 1000, dtype=np.float32)

    for sub_i in range(1, num_subjects + 1):
        sub_id = f"SUB{sub_i:02d}"
        sub_gain = rng.uniform(0.7, 1.4, size=(NUM_BANDS, 105)).astype(np.float32)
        sub_offset = rng.normal(0.0, 0.5, size=(NUM_BANDS, 105)).astype(np.float32)
        raw_trials: List[Dict[str, Any]] = []

        for sent_i in range(sentences_per_subject):
            sent, sent_lbl = corpus[sent_i % len(corpus)]
            words = sent.replace(".", "").replace(",", "").split()
            proto = prototypes[sent]
            ph = phase_offsets[sent]

            # Synthesize 500 Hz continuous 105-ch EEG with sentence-specific phase-lagged oscillations
            osc_12 = np.sin(2.0 * np.pi * 10.0 * t_axis[None, :] + ph[:, None]) + 0.5 * np.sin(2.0 * np.pi * 6.0 * t_axis[None, :] + 1.7 * ph[:, None])
            raw_105 = np.repeat(osc_12, 9, axis=0)[:105] + rng.normal(0.0, 0.35, size=(105, 1000)).astype(np.float32)

            parsed_words = []
            for lex_idx, w_str in enumerate(words):
                base_w = (proto + rng.normal(0.0, 0.3, size=(NUM_BANDS, 105)).astype(np.float32)) * sub_gain + sub_offset
                parsed_words.append(
                    {
                        "lex_idx": lex_idx,
                        "content": w_str,
                        "n_fix": 1.0 + float(lex_idx % 2),
                        "first_fix_pos": float(lex_idx + 1),
                        "revisits": int(lex_idx % 3 == 0),
                        "pupil": float(650.0 + 25.0 * sent_lbl + rng.normal(0, 5)),
                        "ffd_ms": 110.0,
                        "gd_ms": 145.0,
                        "trt_ms": 210.0,
                        "ffd_105": base_w * 0.9,
                        "gd_105": base_w,
                        "trt_105": base_w * 1.15,
                    }
                )

            raw_trials.append(
                {
                    "sentence_idx": sent_i,
                    "target_text": sent,
                    "total_words": len(words),
                    "sent_105": proto * sub_gain + sub_offset,
                    "sent_diff_48": rng.normal(0.0, 0.4, size=(NUM_BANDS, NUM_DIFF_PAIRS)).astype(np.float32),
                    "valid_channel_mask": np.ones((105,), dtype=np.float32),
                    "raw_eeg_105": raw_105,
                    "omission_rate": 0.15,
                    "sacc_vel": 120.0,
                    "sacc_amp": 3.4,
                    "mean_pupil": 665.0,
                    "words": parsed_words,
                }
            )

        sent_map = {_norm_sent_key(s): lbl for s, lbl in corpus}
        recs = assemble_dual_stream_subject_records(
            raw_trials=raw_trials,
            subject_id=sub_id,
            task_name="synthetic-NR",
            max_eeg_len=max_eeg_len,
            sentiment_map=sent_map,
            relation_map={},
        )
        all_records.extend(recs)

    out_file = os.path.join(output_dir, "synthetic-NR.pkl")
    with open(out_file, "wb") as handle:
        pickle.dump(all_records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Synthetic Connectome ZuCo] Saved {len(all_records)} samples -> {out_file}")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel Dual-Stream + 8-Metadata ZuCo Preprocessor")
    parser.add_argument("--raw-root", type=str, default="./datasets/zuco_raw")
    parser.add_argument("--metadata-dir", type=str, default="./datasets/zuco_metadata")
    parser.add_argument("--output-dir", type=str, default="./datasets/processed_zuco_connectome")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional subset of tasks e.g. task1-SR")
    parser.add_argument("--max-subjects", type=int, default=0, help="Optional cap on subjects per task (e.g. 11)")
    parser.add_argument("--max-eeg-len", type=int, default=58)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_zuco(args.output_dir, max_eeg_len=args.max_eeg_len)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    v1_tasks = ["task1-SR", "task2-NR", "task3-TSR"]
    v2_tasks = ["task2-NR-2.0", "task3-TSR-2.0"]
    if args.tasks:
        allowed = set(args.tasks)
        v1_tasks = [t for t in v1_tasks if t in allowed]
        v2_tasks = [t for t in v2_tasks if t in allowed]

    for task in v1_tasks:
        mat_dir = os.path.join(args.raw_root, task, "Matlab_files")
        mat_files = sorted(mf for mf in glob(os.path.join(mat_dir, "*.mat")) if not os.path.exists(mf + ".aria2"))
        if args.max_subjects > 0:
            mat_files = mat_files[: args.max_subjects]
        if not mat_files:
            continue

        print(f"[Connectome ZuCo v1] Parallel processing `{task}` ({len(mat_files)} MAT files across {args.workers} workers)...")
        tmp_dir = os.path.join(args.output_dir, f"_tmp_{task}")
        os.makedirs(tmp_dir, exist_ok=True)
        jobs = [(mf, task, args.max_eeg_len, args.metadata_dir, tmp_dir) for mf in mat_files]
        tmp_paths: Dict[str, str] = {}
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            futs = {pool.submit(_worker_v1, j): j[0] for j in jobs}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=task):
                mf, p_tmp = fut.result()
                tmp_paths[mf] = p_tmp

        task_samples: List[Dict[str, Any]] = []
        for mf in mat_files:
            with open(tmp_paths[mf], "rb") as tf:
                task_samples.extend(pickle.load(tf))
            os.remove(tmp_paths[mf])
        os.rmdir(tmp_dir)

        out_path = os.path.join(args.output_dir, f"{task}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(task_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  -> Saved {len(task_samples)} Dual-Stream Connectome samples to {out_path}")

    for task in v2_tasks:
        mat_dir = os.path.join(args.raw_root, task, "Matlab_files")
        mat_files = sorted(mf for mf in glob(os.path.join(mat_dir, "*.mat")) if not os.path.exists(mf + ".aria2"))
        if args.max_subjects > 0:
            mat_files = mat_files[: args.max_subjects]
        if not mat_files:
            continue

        print(f"[Connectome ZuCo v2] Parallel processing `{task}` ({len(mat_files)} HDF5 MAT files across {args.workers} workers)...")
        tmp_dir = os.path.join(args.output_dir, f"_tmp_{task}")
        os.makedirs(tmp_dir, exist_ok=True)
        jobs = [(mf, task, args.max_eeg_len, args.metadata_dir, tmp_dir) for mf in mat_files]
        tmp_paths = {}
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            futs = {pool.submit(_worker_v2, j): j[0] for j in jobs}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=task):
                mf, p_tmp = fut.result()
                tmp_paths[mf] = p_tmp

        task_samples = []
        for mf in mat_files:
            with open(tmp_paths[mf], "rb") as tf:
                task_samples.extend(pickle.load(tf))
            os.remove(tmp_paths[mf])
        os.rmdir(tmp_dir)

        out_path = os.path.join(args.output_dir, f"{task}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(task_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  -> Saved {len(task_samples)} Dual-Stream Connectome samples to {out_path}")


if __name__ == "__main__":
    main()
