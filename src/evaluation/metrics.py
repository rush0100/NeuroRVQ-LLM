"""
Evaluation Suite matching CET-MAE (ACL 2024) and BrainECHO (ACL 2025):
  - Corpus BLEU-1, BLEU-2, BLEU-3, BLEU-4
  - ROUGE-1 Precision, Recall, F1
  - Word Error Rate (WER) compatible with jiwer 3.x & 4.x
  - Gaussian Noise Input Sanity Gap (Delta BLEU-4 = BLEU-4_EEG - BLEU-4_Noise)
"""

from __future__ import annotations

from typing import Dict, List

import jiwer
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer


def compute_bci_text_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute standardized BLEU-1..4 (%), ROUGE-1 P/R/F (%), and WER (%) on open-vocabulary outputs."""
    assert len(predictions) == len(references), "Mismatched prediction and reference counts."
    smooth_fn = SmoothingFunction().method1

    list_of_refs = [[ref.strip().lower().split()] for ref in references]
    hyps = [pred.strip().lower().split() if pred.strip() else ["<empty>"] for pred in predictions]

    bleu1 = corpus_bleu(list_of_refs, hyps, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smooth_fn) * 100.0
    bleu2 = corpus_bleu(list_of_refs, hyps, weights=(0.5, 0.5, 0.0, 0.0), smoothing_function=smooth_fn) * 100.0
    bleu3 = corpus_bleu(list_of_refs, hyps, weights=(1 / 3, 1 / 3, 1 / 3, 0.0), smoothing_function=smooth_fn) * 100.0
    bleu4 = corpus_bleu(list_of_refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn) * 100.0

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1_p, r1_r, r1_f, rl_f = 0.0, 0.0, 0.0, 0.0
    clean_refs: List[str] = []
    clean_preds: List[str] = []

    for pred, ref in zip(predictions, references):
        c_ref = ref.strip().lower() or "empty"
        c_pred = pred.strip().lower()
        clean_refs.append(c_ref)
        clean_preds.append(c_pred if c_pred else "<empty>")

        scores = scorer.score(c_ref, c_pred)
        r1_p += scores["rouge1"].precision
        r1_r += scores["rouge1"].recall
        r1_f += scores["rouge1"].fmeasure
        rl_f += scores["rougeL"].fmeasure

    if hasattr(jiwer, "process_words"):
        out = jiwer.process_words(clean_refs, clean_preds)
        wer_val = float(out.wer) * 100.0
    else:
        wer_val = float(jiwer.wer(clean_refs, clean_preds)) * 100.0

    n = max(1, len(references))
    return {
        "BLEU-1": round(bleu1, 2),
        "BLEU-2": round(bleu2, 2),
        "BLEU-3": round(bleu3, 2),
        "BLEU-4": round(bleu4, 2),
        "ROUGE-1_P": round((r1_p / n) * 100.0, 2),
        "ROUGE-1_R": round((r1_r / n) * 100.0, 2),
        "ROUGE-1_F1": round((r1_f / n) * 100.0, 2),
        "ROUGE-L_F1": round((rl_f / n) * 100.0, 2),
        "WER": round(wer_val, 2),
    }
