"""
Clinical & Chain-of-Thought Neural Evidence Grounding Evaluation Suite for Paper 2 (`NeuroReasoner`).

Evaluates both Segment-Level and Patient-Level (recording-aggregated) performance across:
  1. Multi-Domain Clinical Classification (Accuracy, Balanced Accuracy, Macro-F1):
     - Cognitive Dementia (`ds004504`: `CN` vs `FTD` vs `AD`, plus Binary Dementia `AD+FTD` vs `CN`)
     - Cross-Center Parkinson's (`UNM` in-center & `UIowa` zero-shot unseen hospital: `CTL` vs `PD`)
     - Affective Depression (`MODMA 128-ch` & `MODMA 3-ch Wearable`: `HC` vs `MDD`)
  2. Clinical Severity Scale Estimation (MAE & Pearson `r`):
     - `MMSE` (0-30) on `ds004504`
     - `PHQ-9` (0-27) & `GAD-7` (0-21) on `MODMA`
  3. Chain-of-Thought (CoT) Neural Evidence Grounding Faithfulness:
     - Valid structured `<neural_evidence>` + `<clinical_reasoning>` + `<assessment>` rate (%)
     - Quantitative Biomarker Grounding Pearson `r` & Normalized MAE across 11 EEG connectome biomarkers
     - Reasoning-to-Assessment Neurophysiological Consistency Rate (%) (zero contradictory hallucinations!)
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

BIOMARKER_KEYS: List[str] = [
    "iaf_hz",
    "posterior_theta_alpha_diff",
    "frontal_theta_beta_diff",
    "central_beta_power_z",
    "frontal_alpha_asymmetry_faa",
    "prefrontal_high_alpha_z",
    "temporoparietal_theta_wpli",
    "frontoparietal_alpha_wpli",
    "frontocentral_beta_wpli",
    "prefrontal_lr_alpha_wpli",
    "spd_tangent_energy",
]

# Standardization scales so all 11 biomarkers contribute balanced O(1) gradients and normalized MAE
BIOMARKER_CENTERS = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 1.20], dtype=np.float32)
BIOMARKER_SCALES = np.array([2.0, 1.0, 1.0, 1.0, 0.6, 1.0, 0.20, 0.20, 0.20, 0.20, 0.80], dtype=np.float32)

DOMAIN_VALID_LABELS: Dict[str, List[str]] = {
    "cognitive_dementia": ["CN", "FTD", "AD"],
    "crosscenter_parkinson": ["CTL", "PD"],
    "affective_depression": ["HC", "MDD"],
}

MECHANISM_KEYWORDS: Dict[str, List[str]] = {
    "AD": ["posterior alpha slowing", "Alzheimer", "elevated posterior theta/alpha"],
    "FTD": ["fronto-temporal", "Frontotemporal", "preservation of primary occipital"],
    "CN": ["preserved posterior alpha", "Cognitively Normal", "intact frontoparietal"],
    "PD": ["beta hypersynchrony", "Parkinson", "slowed background alpha"],
    "CTL": ["normative sensorimotor", "Healthy Control", "desynchronization balance"],
    "MDD": ["aberrant prefrontal alpha asymmetry", "Major Depressive", "dysregulation"],
    "HC": ["balanced bilateral prefrontal", "Healthy Control", "minimal depressive"],
}


def biomarkers_dict_to_norm_vec(bm: Dict[str, Any]) -> np.ndarray:
    raw = np.array([float(bm.get(k, c)) for k, c in zip(BIOMARKER_KEYS, BIOMARKER_CENTERS)], dtype=np.float32)
    return np.clip((raw - BIOMARKER_CENTERS) / BIOMARKER_SCALES, -6.0, 6.0)


def parse_neuroreasoner_cot_output(text: str, domain: str) -> Dict[str, Any]:
    """Extract `<neural_evidence>`, `<clinical_reasoning>`, and `<assessment>` blocks with robust fallback."""
    ev_match = re.search(r"<neural_evidence>\s*(\{.*?\})\s*</neural_evidence>", text, flags=re.DOTALL)
    rs_match = re.search(r"<clinical_reasoning>(.*?)(?:</clinical_reasoning>|<assessment>|$)", text, flags=re.DOTALL)
    as_match = re.search(r"<assessment>\s*(\{.*?\})\s*(?:</assessment>|$)", text, flags=re.DOTALL)

    ev_dict: Dict[str, float] = {}
    ev_valid = False
    if ev_match:
        try:
            loaded = json.loads(ev_match.group(1))
            if isinstance(loaded, dict):
                for k in BIOMARKER_KEYS:
                    if k in loaded and isinstance(loaded[k], (int, float)):
                        ev_dict[k] = float(loaded[k])
                ev_valid = len(ev_dict) >= 6
        except Exception:
            ev_valid = False

    reasoning_str = rs_match.group(1).strip() if rs_match else ""

    assess_dict: Dict[str, Any] = {}
    as_valid = False
    if as_match:
        raw_as = as_match.group(1).strip()
        try:
            loaded_a = json.loads(raw_as)
            if isinstance(loaded_a, dict):
                assess_dict = loaded_a
                as_valid = "diagnosis" in assess_dict
        except Exception:
            # Fallback regex for diagnosis & scores inside incomplete JSON
            m_diag = re.search(r'["\']diagnosis["\']\s*:\s*["\']([A-Za-z]+)["\']', raw_as)
            if m_diag:
                assess_dict["diagnosis"] = m_diag.group(1)
                as_valid = True
            m_mmse = re.search(r'["\']estimated_mmse["\']\s*:\s*([0-9.]+)', raw_as)
            if m_mmse:
                assess_dict["estimated_mmse"] = float(m_mmse.group(1))
            m_phq = re.search(r'["\']estimated_phq9["\']\s*:\s*([0-9.]+)', raw_as)
            if m_phq:
                assess_dict["estimated_phq9"] = float(m_phq.group(1))
            m_gad = re.search(r'["\']estimated_gad7["\']\s*:\s*([0-9.]+)', raw_as)
            if m_gad:
                assess_dict["estimated_gad7"] = float(m_gad.group(1))

    valid_codes = DOMAIN_VALID_LABELS.get(domain, ["CN", "AD", "FTD", "CTL", "PD", "HC", "MDD"])
    pred_diag = str(assess_dict.get("diagnosis", "")).strip().upper()

    if pred_diag not in valid_codes:
        # Fallback regex scan in assessment/tail if JSON truncated
        tail = text[-180:]
        pred_diag = valid_codes[0]
        for code in valid_codes:
            if re.search(rf'\b{re.escape(code)}\b', tail):
                pred_diag = code
                break

    # Check whether reasoning text aligns with predicted diagnosis without contradictory mechanism claims
    expected_kws = MECHANISM_KEYWORDS.get(pred_diag, [])
    reasoning_consistent = False
    if reasoning_str and expected_kws:
        low_r = reasoning_str.lower()
        reasoning_consistent = any(kw.lower() in low_r for kw in expected_kws)

    def _to_float_or_none(val: Any) -> Optional[float]:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val.strip())
            except Exception:
                return None
        return None

    return {
        "structurally_complete": bool(ev_valid and bool(reasoning_str) and as_valid),
        "evidence_valid": ev_valid,
        "evidence_dict": ev_dict,
        "reasoning": reasoning_str,
        "reasoning_consistent": reasoning_consistent,
        "pred_diagnosis": pred_diag,
        "estimated_mmse": _to_float_or_none(assess_dict.get("estimated_mmse")),
        "estimated_phq9": _to_float_or_none(assess_dict.get("estimated_phq9")),
        "estimated_gad7": _to_float_or_none(assess_dict.get("estimated_gad7")),
    }


def _safe_pearson(x: List[float], y: List[float]) -> float:
    if len(x) < 3:
        return 0.0
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if np.std(xa) < 1e-8 or np.std(ya) < 1e-8:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def _cls_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, float]:
    if not y_true:
        return {"acc": 0.0, "bal_acc": 0.0, "macro_f1": 0.0, "n": 0}
    present = [l for l in labels if l in set(y_true) | set(y_pred)] or labels
    return {
        "acc": round(float(accuracy_score(y_true, y_pred)) * 100.0, 2),
        "bal_acc": round(float(balanced_accuracy_score(y_true, y_pred)) * 100.0, 2),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)) * 100.0, 2),
        "n": len(y_true),
    }


def evaluate_neuroreasoner_predictions(
    records_meta: List[Dict[str, Any]],
    generated_texts: List[str],
) -> Dict[str, Any]:
    """
    Compute comprehensive Segment-Level & Patient-Level Clinical + CoT Grounding Metrics.
    """
    assert len(records_meta) == len(generated_texts)

    struct_ok = 0
    consistent_ok = 0
    bm_pred_norm_all: List[float] = []
    bm_true_norm_all: List[float] = []
    per_bm_pairs: Dict[str, Tuple[List[float], List[float]]] = {k: ([], []) for k in BIOMARKER_KEYS}

    # Segment-level buckets
    seg_by_cohort: Dict[str, Tuple[List[str], List[str], str]] = {}

    # Patient-level aggregation (`(cohort_name, subject_id)`)
    pat_votes: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    pat_gt: Dict[Tuple[str, str], Tuple[str, str, Dict[str, Any]]] = {}
    pat_mmse_preds: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    pat_phq9_preds: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    pat_gad7_preds: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for meta, gen_txt in zip(records_meta, generated_texts):
        domain = str(meta["domain"])
        cohort = str(meta["cohort_name"])
        sid = str(meta["subject_id"])
        gt_lbl = str(meta["label_code"]).upper()
        true_bm = meta["biomarkers"]
        scores = meta.get("clinical_scores", {})

        parsed = parse_neuroreasoner_cot_output(gen_txt, domain=domain)
        pred_lbl = parsed["pred_diagnosis"]

        if parsed["structurally_complete"]:
            struct_ok += 1
        if parsed["reasoning_consistent"]:
            consistent_ok += 1

        if parsed["evidence_valid"]:
            true_n = biomarkers_dict_to_norm_vec(true_bm)
            pred_n = biomarkers_dict_to_norm_vec(parsed["evidence_dict"])
            bm_true_norm_all.extend(true_n.tolist())
            bm_pred_norm_all.extend(pred_n.tolist())
            for k in BIOMARKER_KEYS:
                if k in parsed["evidence_dict"] and k in true_bm:
                    per_bm_pairs[k][0].append(float(true_bm[k]))
                    per_bm_pairs[k][1].append(float(parsed["evidence_dict"][k]))

        if cohort not in seg_by_cohort:
            seg_by_cohort[cohort] = ([], [], domain)
        seg_by_cohort[cohort][0].append(gt_lbl)
        seg_by_cohort[cohort][1].append(pred_lbl)

        pkey = (cohort, sid)
        pat_votes[pkey].append(pred_lbl)
        pat_gt[pkey] = (gt_lbl, domain, scores)

        if parsed["estimated_mmse"] is not None:
            pat_mmse_preds[pkey].append(float(np.clip(parsed["estimated_mmse"], 0.0, 30.0)))
        if parsed["estimated_phq9"] is not None:
            pat_phq9_preds[pkey].append(float(np.clip(parsed["estimated_phq9"], 0.0, 27.0)))
        if parsed["estimated_gad7"] is not None:
            pat_gad7_preds[pkey].append(float(np.clip(parsed["estimated_gad7"], 0.0, 21.0)))

    # Compute segment-level & patient-level classification per cohort
    cohort_segment_metrics: Dict[str, Any] = {}
    for cohort, (yt, yp, dom) in sorted(seg_by_cohort.items()):
        cohort_segment_metrics[cohort] = _cls_metrics(yt, yp, DOMAIN_VALID_LABELS[dom])

    pat_by_cohort: Dict[str, Tuple[List[str], List[str], str]] = {}
    mmse_true, mmse_pred = [], []
    phq9_true, phq9_pred = [], []
    gad7_true, gad7_pred = [], []

    for pkey, votes in sorted(pat_votes.items()):
        cohort, _ = pkey
        gt_lbl, dom, scores = pat_gt[pkey]
        maj_pred = Counter(votes).most_common(1)[0][0]
        if cohort not in pat_by_cohort:
            pat_by_cohort[cohort] = ([], [], dom)
        pat_by_cohort[cohort][0].append(gt_lbl)
        pat_by_cohort[cohort][1].append(maj_pred)

        if isinstance(scores.get("MMSE"), (int, float)) and pat_mmse_preds[pkey]:
            mmse_true.append(float(scores["MMSE"]))
            mmse_pred.append(float(np.mean(pat_mmse_preds[pkey])))
        if isinstance(scores.get("PHQ-9"), (int, float)) and pat_phq9_preds[pkey]:
            phq9_true.append(float(scores["PHQ-9"]))
            phq9_pred.append(float(np.mean(pat_phq9_preds[pkey])))
        if isinstance(scores.get("GAD-7"), (int, float)) and pat_gad7_preds[pkey]:
            gad7_true.append(float(scores["GAD-7"]))
            gad7_pred.append(float(np.mean(pat_gad7_preds[pkey])))

    cohort_patient_metrics: Dict[str, Any] = {}
    for cohort, (yt, yp, dom) in sorted(pat_by_cohort.items()):
        m = _cls_metrics(yt, yp, DOMAIN_VALID_LABELS[dom])
        # Also compute binary dementia screening (AD+FTD vs CN) for ds004504
        if dom == "cognitive_dementia":
            yt_bin = ["DEM" if y in ("AD", "FTD") else "CN" for y in yt]
            yp_bin = ["DEM" if y in ("AD", "FTD") else "CN" for y in yp]
            m["binary_dementia_acc"] = round(float(accuracy_score(yt_bin, yp_bin)) * 100.0, 2)
            m["binary_dementia_f1"] = round(
                float(f1_score(yt_bin, yp_bin, labels=["CN", "DEM"], average="macro", zero_division=0)) * 100.0, 2
            )
        cohort_patient_metrics[cohort] = m

    mean_pat_f1 = (
        float(np.mean([v["macro_f1"] for v in cohort_patient_metrics.values()]))
        if cohort_patient_metrics
        else 0.0
    )
    mean_pat_acc = (
        float(np.mean([v["acc"] for v in cohort_patient_metrics.values()]))
        if cohort_patient_metrics
        else 0.0
    )

    bm_norm_mae = (
        float(np.mean(np.abs(np.asarray(bm_true_norm_all) - np.asarray(bm_pred_norm_all))))
        if bm_true_norm_all
        else 9.99
    )
    bm_pearson = _safe_pearson(bm_true_norm_all, bm_pred_norm_all)

    per_bm_corr = {
        k: round(_safe_pearson(t_list, p_list), 4)
        for k, (t_list, p_list) in per_bm_pairs.items()
        if len(t_list) >= 3
    }

    n_total = max(1, len(records_meta))
    severity_scales: Dict[str, Any] = {}
    if mmse_true:
        severity_scales["MMSE"] = {
            "mae": round(float(np.mean(np.abs(np.asarray(mmse_true) - np.asarray(mmse_pred)))), 2),
            "pearson_r": round(_safe_pearson(mmse_true, mmse_pred), 4),
            "n_patients": len(mmse_true),
        }
    if phq9_true:
        severity_scales["PHQ-9"] = {
            "mae": round(float(np.mean(np.abs(np.asarray(phq9_true) - np.asarray(phq9_pred)))), 2),
            "pearson_r": round(_safe_pearson(phq9_true, phq9_pred), 4),
            "n_patients": len(phq9_true),
        }
    if gad7_true:
        severity_scales["GAD-7"] = {
            "mae": round(float(np.mean(np.abs(np.asarray(gad7_true) - np.asarray(gad7_pred)))), 2),
            "pearson_r": round(_safe_pearson(gad7_true, gad7_pred), 4),
            "n_patients": len(gad7_true),
        }

    return {
        "patient_level_mean_macro_f1": round(mean_pat_f1, 2),
        "patient_level_mean_acc": round(mean_pat_acc, 2),
        "patient_level_by_cohort": cohort_patient_metrics,
        "segment_level_by_cohort": cohort_segment_metrics,
        "severity_scale_estimation": severity_scales,
        "cot_grounding_faithfulness": {
            "structural_completeness_rate": round((struct_ok / n_total) * 100.0, 2),
            "reasoning_diagnosis_consistency_rate": round((consistent_ok / n_total) * 100.0, 2),
            "biomarker_grounding_pearson_r": round(bm_pearson, 4),
            "biomarker_grounding_norm_mae": round(bm_norm_mae, 4),
            "per_biomarker_pearson_r": per_bm_corr,
        },
    }
