"""
Paper 2 Trainer & Privacy-Preserving Cross-Center Evaluator (`NeuroReasoner`).

Supports three rigorous experimental modes from a single unified script:
  1. `federated_lora` (Default Proposed Method):
     - Partitions training data strictly by `cohort_name` into isolated hospital silos
       (`AHEPA_Thessaloniki_19ch`, `UNM_Albuquerque_60ch`, `Lanzhou_128ch`, `Lanzhou_Wearable_3ch`).
     - Raw EEG waveforms/tensors NEVER cross hospital boundaries.
     - Each communication round runs local gradient updates per hospital silo on its private cohort,
       applies optional Gaussian Local Differential Privacy (`dp_noise_std`), and aggregates ONLY lightweight
       trainable adapter/encoder parameters via sample-weighted `FedAvg`!
  2. `centralized`:
     - Pooled multi-center baseline (upper-bound privacy-utility reference).
  3. `naive_no_grounding` (`use_evidence_grounding: false`):
     - Ablates explicit biomarker/diagnostic grounding (`bio_weight=0, diag_weight=0`) to quantify
       Chain-of-Thought Neural Evidence Grounding gains over naive EEG-LLM adapters (`WaveMind`-style).

Evaluates at completion on:
  - Patient-disjoint Held-Out Multi-Domain Test Cohorts (`ds004504`, `UNM`, `MODMA 128-ch`, `MODMA 3-ch`)
  - Strictly Unseen External Hospital Cohort (`UIowa_Center`, 28 subjects — Zero-Shot Cross-Center Transfer!)
  - Pure Gaussian Noise `N(0, I)` Sanity Audit across both In-Center Test and Unseen `UIowa`!
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.clinical_dataset import (
    NeuroReasonerClinicalDataset,
    collate_clinical_neuroreasoner,
    load_neuroreasoner_splits,
)
from src.evaluation.clinical_metrics import (
    biomarkers_dict_to_norm_vec,
    evaluate_neuroreasoner_predictions,
)
from src.models.eeg_llm_bridge import load_llm_and_tokenizer
from src.models.neuroreasoner_model import NeuroReasonerModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_cfg)


def build_biomarker_tensor(batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    vecs = [biomarkers_dict_to_norm_vec(bm) for bm in batch["biomarkers"]]
    return torch.from_numpy(np.stack(vecs, axis=0)).to(device)


@torch.no_grad()
def run_clinical_cot_evaluation(
    model: NeuroReasonerModel,
    dataset: NeuroReasonerClinicalDataset,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
    max_prompt_tokens: int = 128,
    max_new_tokens: int = 230,
    replace_with_gaussian_noise: bool = False,
    desc: str = "Clinical CoT Eval",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_clinical_neuroreasoner,
    )
    meta_records: List[Dict[str, Any]] = []
    gen_outputs: List[str] = []
    preview_samples: List[Dict[str, Any]] = []
    seen_cohorts_preview: set[str] = set()

    for batch in tqdm(loader, desc=desc, leave=False):
        decoded = model.generate_clinical_cot(
            batch=batch,
            device=device,
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            replace_with_gaussian_noise=replace_with_gaussian_noise,
        )
        gen_outputs.extend(decoded)
        for i in range(len(decoded)):
            m = {
                "subject_id": batch["subject_id"][i],
                "cohort_name": batch["cohort_name"][i],
                "domain": batch["domain"][i],
                "center_name": batch["center_name"][i],
                "label_code": batch["label_code"][i],
                "clinical_scores": batch["clinical_scores"][i],
                "biomarkers": batch["biomarkers"][i],
            }
            meta_records.append(m)
            c_name = m["cohort_name"]
            if len(preview_samples) < 8 and (c_name not in seen_cohorts_preview or len(preview_samples) < 5):
                seen_cohorts_preview.add(c_name)
                preview_samples.append(
                    {
                        "subject_id": m["subject_id"],
                        "cohort": c_name,
                        "ground_truth_label": m["label_code"],
                        "ground_truth_scores": m["clinical_scores"],
                        "generated_cot_report": decoded[i],
                    }
                )

    metrics = evaluate_neuroreasoner_predictions(meta_records, gen_outputs)
    return metrics, preview_samples


def build_optimizer_for_model(model: NeuroReasonerModel, tr_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    lora_params = [p for _, p in model.llm.named_parameters() if p.requires_grad]
    enc_params = (
        list(model.eeg_encoder.parameters())
        + list(model.rvq.parameters())
        + list(model.roi_coverage_adapter.parameters())
    )
    bridge_params = (
        list(model.pooler.parameters())
        + list(model.projector.parameters())
        + list(model.evidence_bridge.parameters())
    )
    return torch.optim.AdamW(
        [
            {"params": bridge_params, "lr": float(tr_cfg["lr_projector"])},
            {"params": lora_params, "lr": float(tr_cfg["lr_lora"])},
            {"params": enc_params, "lr": float(tr_cfg["lr_encoder"])},
        ],
        weight_decay=float(tr_cfg.get("weight_decay", 0.02)),
    )


def train_local_dataloader(
    model: NeuroReasonerModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum: int,
    max_prompt_tokens: int,
    max_target_tokens: int,
    rvq_weight: float,
    bio_weight: float,
    diag_weight: float,
) -> Dict[str, float]:
    model.train()
    tot_loss, lm_sum, bio_sum, diag_sum, ppl_sum = 0.0, 0.0, 0.0, 0.0, 0.0
    optimizer.zero_grad()

    n_steps = max(1, len(loader))
    for step, batch in enumerate(loader, start=1):
        bm_norm = build_biomarker_tensor(batch, device=device)
        out = model(
            batch=batch,
            target_biomarkers_norm=bm_norm,
            device=device,
            max_prompt_tokens=max_prompt_tokens,
            max_target_tokens=max_target_tokens,
            rvq_weight=rvq_weight,
            bio_weight=bio_weight,
            diag_weight=diag_weight,
        )
        loss = out["loss"] / grad_accum
        loss.backward()

        tot_loss += float(out["loss"].item())
        lm_sum += float(out["lm_loss"].item())
        bio_sum += float(out["bio_loss"].item())
        diag_sum += float(out["diag_loss"].item())
        ppl_sum += float(out["perplexity"])

        if step % grad_accum == 0 or step == n_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    return {
        "loss": round(tot_loss / n_steps, 4),
        "lm_loss": round(lm_sum / n_steps, 4),
        "bio_loss": round(bio_sum / n_steps, 4),
        "diag_loss": round(diag_sum / n_steps, 4),
        "perplexity": round(ppl_sum / n_steps, 2),
    }


def fedavg_states(states_and_weights: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    """Sample-weighted Federated Averaging (`FedAvg`) across hospital silos."""
    total_samples = float(sum(w for _, w in states_and_weights))
    keys = list(states_and_weights[0][0].keys())
    agg: Dict[str, torch.Tensor] = {}

    for k in keys:
        first_t = states_and_weights[0][0][k]
        if not first_t.is_floating_point():
            # Boolean/integer buffer (e.g., `initialized` flag)
            agg[k] = first_t.clone()
            continue
        acc = torch.zeros_like(first_t, dtype=torch.float32)
        for sd, w in states_and_weights:
            acc = acc + sd[k].float() * (float(w) / max(1.0, total_samples))
        agg[k] = acc
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train & Evaluate Paper 2: NeuroReasoner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration")
    parser.add_argument("--paradigm", type=str, default=None, choices=["federated_lora", "centralized"])
    parser.add_argument("--disable-grounding", action="store_true", help="Ablation: disable explicit biomarker/diagnostic bridge losses")
    parser.add_argument("--exp-suffix", type=str, default="", help="Optional experiment name suffix")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Resume continual training from existing privacy state checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training rounds/epochs")
    parser.add_argument("--lr-lora", type=float, default=None, help="Override LoRA learning rate")
    parser.add_argument("--max-target-tokens", type=int, default=None, help="Override max target tokens")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max new tokens for generation")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.paradigm:
        cfg["training"]["paradigm"] = args.paradigm
    if args.disable_grounding:
        cfg["llm"]["cot_evidence_grounding"] = False
    if args.exp_suffix:
        cfg["experiment_name"] = f"{cfg['experiment_name']}_{args.exp_suffix}"
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.lr_lora is not None:
        cfg["training"]["lr_lora"] = args.lr_lora
    if args.max_target_tokens is not None:
        cfg["training"]["max_target_tokens"] = args.max_target_tokens
    if args.max_new_tokens is not None:
        cfg["llm"]["max_new_tokens"] = args.max_new_tokens

    exp_name = cfg["experiment_name"]
    out_dir = os.path.join("./checkpoints", exp_name)
    log_dir = os.path.join("./logs", exp_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])

    splits = load_neuroreasoner_splits(
        processed_root=cfg["data"]["processed_root"],
        protocol=cfg["data"]["protocol"],
        fold_idx=int(cfg["data"].get("fold_idx", 0)),
        seed=int(cfg["seed"]),
    )
    train_ds = splits["train"]
    val_ds = splits["val"]
    test_ds = splits["test"]
    ext_ds = splits.get("external_crosscenter_test")

    llm_cfg = cfg["llm"]
    llm, tokenizer = load_llm_and_tokenizer(
        model_name_or_path=llm_cfg["model_name_or_path"],
        quantization=llm_cfg.get("quantization", "none"),
        precision=cfg["precision"],
        use_lora=bool(llm_cfg.get("use_lora", True)),
        lora_r=int(llm_cfg.get("lora_r", 16)),
        lora_alpha=int(llm_cfg.get("lora_alpha", 32)),
        lora_dropout=float(llm_cfg.get("lora_dropout", 0.05)),
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if llm_cfg.get("quantization", "none") != "nf4":
        llm.to(device)

    enc_cfg = cfg["eeg_encoder"]
    rvq_cfg = cfg["rvq"]
    model = NeuroReasonerModel(
        llm=llm,
        tokenizer=tokenizer,
        embed_dim=int(enc_cfg["embed_dim"]),
        encoder_layers=int(enc_cfg["num_layers"]),
        encoder_heads=int(enc_cfg["num_heads"]),
        ffn_dim=int(enc_cfg["ffn_dim"]),
        dropout=float(enc_cfg["dropout"]),
        stream_mode=str(enc_cfg["stream_mode"]),
        num_latent_queries=int(enc_cfg["num_latent_queries"]),
        rvq_num_quantizers=int(rvq_cfg["num_quantizers"]),
        rvq_codebook_size=int(rvq_cfg["codebook_size"]),
        rvq_commitment_weight=float(rvq_cfg["commitment_weight"]),
        rvq_quantizer_dropout=float(rvq_cfg["quantizer_dropout"]),
        rvq_ema_decay=float(rvq_cfg["ema_decay"]),
        max_eeg_len=max(64, int(cfg["data"]["max_eeg_len"]) + 8),
        init_paper1_ckpt=enc_cfg.get("init_from_paper1_stage1"),
    )
    # Move non-LLM submodules to device
    model.eeg_encoder.to(device)
    model.rvq.to(device)
    model.roi_coverage_adapter.to(device)
    model.pooler.to(device)
    model.projector.to(device)
    model.evidence_bridge.to(device)

    tr_cfg = cfg["training"]
    paradigm = str(tr_cfg.get("paradigm", "federated_lora"))
    use_grounding = bool(llm_cfg.get("cot_evidence_grounding", True))
    bio_w = float(tr_cfg.get("bio_loss_weight", 0.8)) if use_grounding else 0.0
    diag_w = float(tr_cfg.get("diag_loss_weight", 0.8)) if use_grounding else 0.0
    rvq_w = float(tr_cfg.get("rvq_loss_weight", 0.5))
    dp_noise = float(tr_cfg.get("dp_noise_std", 0.0))

    batch_size = int(tr_cfg["batch_size"])
    eval_batch_size = int(tr_cfg.get("eval_batch_size", 32))
    grad_accum = int(tr_cfg.get("grad_accum_steps", 2))
    epochs = int(tr_cfg["epochs"])
    patience = int(tr_cfg.get("early_stop_patience", 5))
    num_workers = int(cfg["data"].get("num_workers", 4))
    max_p_toks = int(llm_cfg.get("max_prompt_tokens", 128))
    max_new_toks = int(llm_cfg.get("max_new_tokens", 430))
    max_tgt_toks = int(tr_cfg.get("max_target_tokens", max_new_toks))

    best_ckpt_path = os.path.join(out_dir, "neuroreasoner_privacy_state_best.pt")

    resume_path = args.resume_from_checkpoint or tr_cfg.get("resume_from_checkpoint")
    if resume_path and os.path.exists(resume_path):
        res_sd = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_privacy_preserving_state(res_sd)
        print(f"[NeuroReasoner] Resumed trainable model state from `{resume_path}`!")

    print(
        f"[NeuroReasoner] Exp=`{exp_name}` | Paradigm=`{paradigm}` | Grounding=`{use_grounding}` (DP={dp_noise}) | "
        f"Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)} "
        f"ExtUIowa={len(ext_ds) if ext_ds else 0}"
    )

    t0 = time.time()
    history: List[Dict[str, Any]] = []

    if not args.eval_only:
        best_score = -1e9
        no_improve = 0

        # Partition indices by hospital cohort for `federated_lora`
        silo_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, rec in enumerate(train_ds.samples):
            silo_indices[str(rec["cohort_name"])].append(idx)

        # Build deterministic fast validation view (`<= 4` segments per patient across all validation patients)
        # for per-round checkpoint selection, while running 100% of segments at final evaluation!
        val_max_seg = int(tr_cfg.get("val_max_segments_per_patient", 4))
        val_per_pat: Dict[str, List[int]] = defaultdict(list)
        for idx, rec in enumerate(val_ds.samples):
            pk = f"{rec['cohort_name']}::{rec['subject_id']}"
            if len(val_per_pat[pk]) < val_max_seg:
                val_per_pat[pk].append(idx)
        fast_val_indices = sorted([i for idxs in val_per_pat.values() for i in idxs])
        fast_val_ds = NeuroReasonerClinicalDataset([val_ds.samples[i] for i in fast_val_indices])
        print(f"[NeuroReasoner] Per-round validation uses all {len(val_per_pat)} unseen val patients ({len(fast_val_ds)}/{len(val_ds)} segments) with KV-cache enabled!")

        central_opt = build_optimizer_for_model(model, tr_cfg) if paradigm == "centralized" else None
        central_sched = (
            torch.optim.lr_scheduler.CosineAnnealingLR(central_opt, T_max=max(1, epochs))
            if central_opt is not None
            else None
        )

        for round_idx in range(1, epochs + 1):
            # Cosine LR decay factor for federated rounds
            lr_mult = max(0.10, 0.5 * (1.0 + math.cos(math.pi * (round_idx - 1) / max(1, epochs))))

            if paradigm == "federated_lora":
                global_state = model.get_privacy_preserving_state(dp_noise_std=0.0)
                silo_updates: List[Tuple[Dict[str, torch.Tensor], int]] = []
                round_losses: List[float] = []
                round_bio: List[float] = []
                round_diag: List[float] = []

                for s_i, (silo_name, idxs) in enumerate(sorted(silo_indices.items()), start=1):
                    t_silo = time.time()
                    model.load_privacy_preserving_state(global_state)
                    silo_cfg = copy.deepcopy(tr_cfg)
                    silo_cfg["lr_projector"] = float(tr_cfg["lr_projector"]) * lr_mult
                    silo_cfg["lr_lora"] = float(tr_cfg["lr_lora"]) * lr_mult
                    silo_cfg["lr_encoder"] = float(tr_cfg["lr_encoder"]) * lr_mult

                    silo_opt = build_optimizer_for_model(model, silo_cfg)
                    silo_loader = DataLoader(
                        Subset(train_ds, idxs),
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        collate_fn=collate_clinical_neuroreasoner,
                    )
                    stats = train_local_dataloader(
                        model=model,
                        loader=silo_loader,
                        optimizer=silo_opt,
                        device=device,
                        grad_accum=grad_accum,
                        max_prompt_tokens=max_p_toks,
                        max_target_tokens=max_tgt_toks,
                        rvq_weight=rvq_w,
                        bio_weight=bio_w,
                        diag_weight=diag_w,
                    )
                    silo_state = model.get_privacy_preserving_state(dp_noise_std=dp_noise)
                    silo_updates.append((silo_state, len(idxs)))
                    round_losses.append(stats["loss"])
                    round_bio.append(stats["bio_loss"])
                    round_diag.append(stats["diag_loss"])
                    print(
                        f"  -> [Rd {round_idx:02d} | Silo {s_i}/{len(silo_indices)} `{silo_name}` ({len(idxs)} segs)] "
                        f"Loss={stats['loss']:.4f} Bio={stats['bio_loss']:.4f} Diag={stats['diag_loss']:.4f} ({time.time()-t_silo:.1f}s)",
                        flush=True,
                    )

                agg_state = fedavg_states(silo_updates)
                model.load_privacy_preserving_state(agg_state)
                tr_summary = {
                    "loss": round(float(np.mean(round_losses)), 4),
                    "bio_loss": round(float(np.mean(round_bio)), 4),
                    "diag_loss": round(float(np.mean(round_diag)), 4),
                }
            else:
                assert central_opt is not None and central_sched is not None
                train_loader = DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    collate_fn=collate_clinical_neuroreasoner,
                )
                stats = train_local_dataloader(
                    model=model,
                    loader=train_loader,
                    optimizer=central_opt,
                    device=device,
                    grad_accum=grad_accum,
                    max_prompt_tokens=max_p_toks,
                    max_target_tokens=max_tgt_toks,
                    rvq_weight=rvq_w,
                    bio_weight=bio_w,
                    diag_weight=diag_w,
                )
                central_sched.step()
                tr_summary = {
                    "loss": stats["loss"],
                    "bio_loss": stats["bio_loss"],
                    "diag_loss": stats["diag_loss"],
                }

            val_metrics, _ = run_clinical_cot_evaluation(
                model=model,
                dataset=fast_val_ds,
                device=device,
                batch_size=eval_batch_size,
                num_workers=num_workers,
                max_prompt_tokens=max_p_toks,
                max_new_tokens=max_new_toks,
                replace_with_gaussian_noise=False,
                desc=f"Val Round {round_idx:02d}",
            )
            pat_f1 = float(val_metrics["patient_level_mean_macro_f1"])
            pat_acc = float(val_metrics["patient_level_mean_acc"])
            cot_f = val_metrics["cot_grounding_faithfulness"]
            bm_r = float(cot_f["biomarker_grounding_pearson_r"])
            cons_rate = float(cot_f["reasoning_diagnosis_consistency_rate"])

            # Composite clinical + grounding selection score
            sel_score = pat_f1 + 15.0 * bm_r + 0.10 * cons_rate

            print(
                f"[Round {round_idx:02d}/{epochs}] Tr(Loss={tr_summary['loss']:.4f} Bio={tr_summary['bio_loss']:.4f} "
                f"Diag={tr_summary['diag_loss']:.4f}) | "
                f"Val PatF1={pat_f1:.2f}% PatAcc={pat_acc:.2f}% | "
                f"CoT(Bio_r={bm_r:.3f} Consist={cons_rate:.1f}% Struct={cot_f['structural_completeness_rate']:.1f}%)"
            )

            history.append(
                {
                    "round": round_idx,
                    "train": tr_summary,
                    "val": val_metrics,
                    "selection_score": round(sel_score, 4),
                }
            )
            with open(os.path.join(log_dir, "neuroreasoner_history.json"), "w", encoding="utf-8") as hf:
                json.dump(history, hf, indent=2)

            if sel_score >= best_score:
                best_score = sel_score
                no_improve = 0
                torch.save(model.get_privacy_preserving_state(dp_noise_std=0.0), best_ckpt_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[NeuroReasoner Early Stop] Plateaued for {patience} rounds.")
                    break

    if os.path.exists(best_ckpt_path):
        best_sd = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        model.load_privacy_preserving_state(best_sd)

    print("\n=== Final Evaluation 1/3: Held-Out Patient-Disjoint Multi-Center Test Cohorts (Real EEG) ===")
    in_test_real, in_examples = run_clinical_cot_evaluation(
        model=model,
        dataset=test_ds,
        device=device,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        max_prompt_tokens=max_p_toks,
        max_new_tokens=max_new_toks,
        replace_with_gaussian_noise=False,
        desc="Test (In-Center Real EEG)",
    )
    print(json.dumps(in_test_real, indent=2))

    print("\n=== Final Evaluation 2/3: Pure Gaussian Noise N(0,I) Sanity Audit ===")
    in_test_noise, _ = run_clinical_cot_evaluation(
        model=model,
        dataset=test_ds,
        device=device,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        max_prompt_tokens=max_p_toks,
        max_new_tokens=max_new_toks,
        replace_with_gaussian_noise=True,
        desc="Test (Gaussian Noise Sanity)",
    )

    ext_real_metrics: Dict[str, Any] = {}
    ext_noise_metrics: Dict[str, Any] = {}
    ext_examples: List[Dict[str, Any]] = []
    if ext_ds is not None and len(ext_ds) > 0:
        print("\n=== Final Evaluation 3/3: Zero-Shot Unseen External Hospital (`UIowa_Center`, 28 Subjects) ===")
        ext_real_metrics, ext_examples = run_clinical_cot_evaluation(
            model=model,
            dataset=ext_ds,
            device=device,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            max_prompt_tokens=max_p_toks,
            max_new_tokens=max_new_toks,
            replace_with_gaussian_noise=False,
            desc="Zero-Shot UIowa Real EEG",
        )
        ext_noise_metrics, _ = run_clinical_cot_evaluation(
            model=model,
            dataset=ext_ds,
            device=device,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            max_prompt_tokens=max_p_toks,
            max_new_tokens=max_new_toks,
            replace_with_gaussian_noise=True,
            desc="Zero-Shot UIowa Noise",
        )
        print(json.dumps(ext_real_metrics, indent=2))

    report = {
        "experiment": exp_name,
        "paradigm": paradigm,
        "cot_evidence_grounding": use_grounding,
        "dp_noise_std": dp_noise,
        "elapsed_min": round((time.time() - t0) / 60.0, 2),
        "in_center_patient_disjoint_test_real_eeg": in_test_real,
        "in_center_patient_disjoint_test_gaussian_noise": in_test_noise,
        "in_center_sanity_gap_macro_f1": round(
            in_test_real["patient_level_mean_macro_f1"] - in_test_noise["patient_level_mean_macro_f1"], 2
        ),
        "in_center_sanity_gap_biomarker_r": round(
            in_test_real["cot_grounding_faithfulness"]["biomarker_grounding_pearson_r"]
            - in_test_noise["cot_grounding_faithfulness"]["biomarker_grounding_pearson_r"],
            4,
        ),
        "external_zero_shot_uiowa_real_eeg": ext_real_metrics,
        "external_zero_shot_uiowa_gaussian_noise": ext_noise_metrics,
        "sample_cot_reports_in_center": in_examples,
        "sample_cot_reports_external_uiowa": ext_examples,
        "config": cfg,
    }
    report_path = os.path.join(log_dir, "neuroreasoner_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2, ensure_ascii=False)
    print(f"\n[Saved NeuroReasoner Final Report] -> {report_path}")


if __name__ == "__main__":
    main()
