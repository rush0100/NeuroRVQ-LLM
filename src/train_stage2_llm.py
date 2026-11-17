"""
Stage 2 Training & Teacher-Forcing-Free Evaluation Script:
Aligns Frozen/Pretrained CET-RVQ Discrete Brain Tokens -> Decoder-Only LLM (LoRA / QLoRA on DGX Spark)
and runs both Real-EEG Decoding and Gaussian-Noise Sanity Evaluation (BrainECHO ACL 2025 Protocol).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import random
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import collate_eeg_text, load_and_split_zuco
from src.evaluation.metrics import compute_bci_text_metrics
from src.models.cet_rvq_pretrainer import CETRVQPretrainer
from src.models.eeg_llm_bridge import RUSHEEG2TextForCausalDecoding, load_llm_and_tokenizer


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_cfg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _encode_contextual_sentences(
    model: RUSHEEG2TextForCausalDecoding,
    texts: list[str],
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Extract mean-pooled last-hidden-state contextual embeddings [N, D] using the underlying backbone."""
    base_m = model.llm.get_base_model() if hasattr(model.llm, "get_base_model") else model.llm
    vecs: list[torch.Tensor] = []
    for s in range(0, len(texts), batch_size):
        chunk = [t.strip() if t.strip() else "empty" for t in texts[s : s + batch_size]]
        tok = model.tokenizer(chunk, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        if getattr(model, "is_seq2seq", False):
            enc_out = base_m.get_encoder()(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
            h = enc_out.last_hidden_state.float()
        else:
            core = getattr(base_m, "model", base_m)
            out = core(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], return_dict=True)
            h = out.last_hidden_state.float()
        m = tok["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        vecs.append(pooled.cpu())
    return torch.cat(vecs, dim=0)


def evaluate_teacher_forcing_free(
    model: RUSHEEG2TextForCausalDecoding,
    dataloader: DataLoader,
    device: torch.device,
    max_new_tokens: int = 48,
    num_beams: int = 4,
    repetition_penalty: float = 1.25,
    no_repeat_ngram_size: int = 3,
    replace_with_gaussian_noise: bool = False,
    compute_per_subject: bool = False,
) -> tuple[dict[str, float], list[dict[str, str]], dict[str, Any]]:
    model.eval()
    preds: list[str] = []
    refs: list[str] = []
    subs_all: list[str] = []
    sent_labels_all: list[int] = []
    by_sub_examples: dict[str, list[dict[str, str]]] = defaultdict(list)

    desc = "Eval (Gaussian Noise Sanity)" if replace_with_gaussian_noise else "Eval (Real EEG TF-Free)"
    for batch in tqdm(dataloader, desc=desc, leave=False):
        eeg = batch["eeg_features"].to(device)
        mask = batch["eeg_attn_mask"].to(device)
        decoded = model.generate_teacher_forcing_free(
            eeg_features=eeg,
            eeg_attn_mask=mask,
            node_features=batch["node_features"].to(device),
            word_coupled_edges=batch["word_coupled_edges"].to(device),
            sent_connectome=batch["sent_connectome"].to(device),
            rel_pos=batch["rel_pos"].to(device),
            lex_pos=batch["lex_pos"].to(device),
            word_salience=batch["word_salience"].to(device),
            cog_load=batch["cog_load"].to(device),
            task_id=batch["task_id"].to(device),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            replace_with_gaussian_noise=replace_with_gaussian_noise,
        )
        preds.extend(decoded)
        refs.extend(batch["target_text"])
        subs_all.extend(batch["subject_id"])
        sent_labels_all.extend([int(x) for x in batch["sentiment_label"].tolist()])

        for sub, ref, hyp in zip(batch["subject_id"], batch["target_text"], decoded):
            by_sub_examples[sub].append({"subject": sub, "reference": ref, "decoded": hyp})

    metrics = compute_bci_text_metrics(preds, refs)

    # Contextual Semantic Cosine Similarity (mean-centered on reference distribution to eliminate anisotropy!)
    ref_vecs = _encode_contextual_sentences(model, refs, device=device)
    pred_vecs = _encode_contextual_sentences(model, preds, device=device)
    ref_mean = ref_vecs.mean(dim=0, keepdim=True)
    ref_norm = F.normalize(ref_vecs - ref_mean, dim=-1)
    pred_norm = F.normalize(pred_vecs - ref_mean, dim=-1)
    cos_per_item = (ref_norm * pred_norm).sum(dim=-1)
    metrics["Semantic_CosSim"] = round(float(cos_per_item.mean().item()) * 100.0, 2)

    # Zero-shot Affective/Sentiment Polarity Preservation via reference sentiment centroids (on labeled SR trials)
    valid_sent_idx = [i for i, y in enumerate(sent_labels_all) if y in (0, 1, 2)]
    if len(valid_sent_idx) >= 9:
        centroids = []
        present_classes = []
        for c_id in (0, 1, 2):
            c_rows = [i for i in valid_sent_idx if sent_labels_all[i] == c_id]
            if c_rows:
                centroids.append(F.normalize(ref_norm[c_rows].mean(dim=0, keepdim=True), dim=-1))
                present_classes.append(c_id)
        if len(present_classes) >= 2:
            cent_mat = torch.cat(centroids, dim=0)  # [C, D]
            sims = pred_norm[valid_sent_idx] @ cent_mat.t()
            pred_cls = [present_classes[int(k)] for k in torch.argmax(sims, dim=-1).tolist()]
            gt_cls = [sent_labels_all[i] for i in valid_sent_idx]
            metrics["Sentiment_Agree"] = round(float(np.mean([a == b for a, b in zip(pred_cls, gt_cls)])) * 100.0, 2)

    # Collect balanced qualitative examples across ALL unseen subjects (not just the first subject in dataloader!)
    samples_preview: list[dict[str, str]] = []
    for sub in sorted(by_sub_examples.keys()):
        items = by_sub_examples[sub]
        # Rank by unigram+semantic overlap to surface both verbatim hits & representative semantic paraphrases per subject
        scored = []
        for ex in items:
            r_set = set(ex["reference"].lower().split())
            p_set = set(ex["decoded"].lower().split())
            ov = len(r_set & p_set) / max(1, len(r_set))
            scored.append((ov, ex))
        scored.sort(key=lambda t: t[0], reverse=True)
        top_picks = [x[1] for x in scored[:3]]
        if len(scored) >= 6:
            top_picks.append(scored[len(scored) // 4][1])
        samples_preview.extend(top_picks)

    extra: dict[str, Any] = {}
    if compute_per_subject:
        per_sub: dict[str, Any] = {}
        for sub in sorted(set(subs_all)):
            idxs = [i for i, s in enumerate(subs_all) if s == sub]
            sub_p = [preds[i] for i in idxs]
            sub_r = [refs[i] for i in idxs]
            sm = compute_bci_text_metrics(sub_p, sub_r)
            sm["Semantic_CosSim"] = round(float(cos_per_item[idxs].mean().item()) * 100.0, 2)
            sm["n_trials"] = len(idxs)
            per_sub[sub] = sm
        extra["per_subject"] = per_sub

    return metrics, samples_preview, extra


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Train & Evaluate RUSH-EEG2Text Decoder")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run TF-Free + Noise Sanity evaluation")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["experiment"]["seed"])
    device = resolve_device(cfg["hardware"]["device"])
    os.makedirs(cfg["experiment"]["output_dir"], exist_ok=True)
    os.makedirs(cfg["experiment"]["log_dir"], exist_ok=True)

    eeg_only_mode = str(cfg["data"].get("eeg_only_mode", "none"))
    train_ds, val_ds, test_ds = load_and_split_zuco(
        dataset_root=cfg["data"]["dataset_root"],
        tasks=cfg["data"]["tasks"],
        split_strategy=cfg["data"]["split_strategy"],
        held_out_test_subjects=cfg["data"].get("held_out_test_subjects"),
        held_out_val_subjects=cfg["data"].get("held_out_val_subjects"),
        eeg_only_mode=eeg_only_mode,
    )

    s1_ckpt_path = str(
        cfg["stage2_llm"].get(
            "stage1_ckpt_path",
            os.path.join(cfg["experiment"]["output_dir"], "stage1_cet_rvq_best.pt"),
        )
    )
    if not os.path.exists(s1_ckpt_path):
        raise FileNotFoundError(
            f"Missing Stage 1 checkpoint at {s1_ckpt_path}. Run `python -m src.train_stage1 --config {args.config}` first."
        )

    s1_ckpt = torch.load(s1_ckpt_path, map_location="cpu", weights_only=False)
    s1_cfg = s1_ckpt.get("config", cfg)["stage1_cet_rvq"]
    s2_cfg = cfg["stage2_llm"]
    is_legacy_vq = any("codebook.weight" in k for k in s1_ckpt["model_state_dict"].keys())
    has_Buffer_scale = "logit_scale" in s1_ckpt["model_state_dict"]

    cet_rvq = CETRVQPretrainer(
        eeg_feature_dim=cfg["data"]["eeg_feature_dim"],
        num_channels=cfg["data"]["num_channels"],
        num_bands=len(cfg["data"]["bands"]),
        embed_dim=s1_cfg["embed_dim"],
        llm_hidden_dim=int(s1_ckpt["llm_hidden_dim"]),
        encoder_layers=s1_cfg["encoder_layers"],
        encoder_heads=s1_cfg["encoder_heads"],
        ffn_dim=s1_cfg["ffn_dim"],
        dropout=s1_cfg["dropout"],
        use_resampler=s1_cfg["use_length_invariant_resampler"],
        num_latent_queries=s1_cfg["num_latent_queries"],
        rvq_num_quantizers=s1_cfg["rvq_num_quantizers"],
        rvq_codebook_size=s1_cfg["rvq_codebook_size"],
        rvq_commitment_weight=s1_cfg["rvq_commitment_weight"],
        rvq_quantizer_dropout=float(s1_cfg.get("rvq_quantizer_dropout", 0.0)),
        rvq_ema_decay=float(s1_cfg.get("rvq_ema_decay", 0.98)),
        max_eeg_len=max(128, cfg["data"]["max_eeg_len"] + 8),
        stream_mode=str(s1_cfg.get("stream_mode", "dual")),
        use_task_conditioning=bool(s1_cfg.get("use_task_conditioning", True)),
        use_band_topography_norm=bool(s1_cfg.get("use_band_topography_norm", False)),
        use_attentive_pooler=bool(s1_cfg.get("use_attentive_pooler", False)),
        proj_dropout=float(s1_cfg.get("proj_dropout", 0.1)),
        init_temperature=float(s1_cfg.get("init_temperature", 0.07)),
        learnable_temperature=bool(s1_cfg.get("learnable_temperature", has_Buffer_scale)),
        max_logit_scale=float(s1_cfg.get("max_logit_scale", 20.0)),
        legacy_vanilla_vq=is_legacy_vq,
    )
    cet_rvq.load_state_dict(s1_ckpt["model_state_dict"], strict=False)
    cet_rvq.to(device)

    llm, tokenizer = load_llm_and_tokenizer(
        model_name_or_path=s2_cfg["llm_name_or_path"],
        quantization=s2_cfg.get("quantization", "none"),
        precision=cfg["hardware"]["precision"],
        use_lora=bool(s2_cfg.get("use_lora", True)),
        lora_r=int(s2_cfg.get("lora_r", 16)),
        lora_alpha=int(s2_cfg.get("lora_alpha", 32)),
        lora_dropout=float(s2_cfg.get("lora_dropout", 0.05)),
        lora_target_modules=s2_cfg.get("lora_target_modules"),
    )
    if s2_cfg["quantization"] != "nf4":
        llm.to(device)

    freeze_cet_rvq = bool(s2_cfg.get("freeze_cet_rvq", False))
    model = RUSHEEG2TextForCausalDecoding(
        cet_rvq=cet_rvq,
        llm=llm,
        tokenizer=tokenizer,
        prompt_template=s2_cfg["prompt_template"],
        freeze_cet_rvq=freeze_cet_rvq,
    )
    model.projector.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(s2_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["hardware"]["num_workers"]),
        collate_fn=collate_eeg_text,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(s2_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["hardware"]["num_workers"]),
        collate_fn=collate_eeg_text,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(s2_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["hardware"]["num_workers"]),
        collate_fn=collate_eeg_text,
    )

    cet_ckpt = os.path.join(cfg["experiment"]["output_dir"], "stage2_cet_rvq_best.pt")
    bridge_ckpt = os.path.join(cfg["experiment"]["output_dir"], "stage2_projector_best.pt")
    lora_dir = os.path.join(cfg["experiment"]["output_dir"], "stage2_lora_adapter")

    if not args.eval_only:
        lora_params = [p for n, p in model.llm.named_parameters() if p.requires_grad]
        cet_params = [p for n, p in model.cet_rvq.named_parameters() if p.requires_grad]
        param_groups = [
            {"params": model.projector.parameters(), "lr": float(s2_cfg["lr_projector"])},
            {"params": lora_params, "lr": float(s2_cfg["lr_lora"])},
        ]
        if cet_params:
            param_groups.append({"params": cet_params, "lr": float(s2_cfg.get("lr_eeg_encoder", 3.0e-5))})
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=float(s2_cfg["weight_decay"]),
        )

        epochs = int(s2_cfg["epochs"])
        grad_accum = int(s2_cfg.get("grad_accum_steps", 1))
        max_drop = float(s2_cfg.get("scheduled_word_dropout_max", 0.5))
        patience = int(s2_cfg.get("early_stop_patience", 5))
        eval_val_beams = int(s2_cfg.get("val_eval_num_beams", 2))
        best_score = -1.0
        epochs_no_improve = 0
        s2_history: list[dict[str, object]] = []

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

        for epoch in range(1, epochs + 1):
            model.train()
            # Linearly ramp target word dropout so the LLM relies on RVQ brain tokens instead of preceding text
            word_drop = max_drop * (epoch / max(1, epochs))
            running_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader, start=1):
                eeg = batch["eeg_features"].to(device)
                mask = batch["eeg_attn_mask"].to(device)
                out = model(
                    eeg_features=eeg,
                    eeg_attn_mask=mask,
                    target_texts=batch["target_text"],
                    node_features=batch["node_features"].to(device),
                    word_coupled_edges=batch["word_coupled_edges"].to(device),
                    sent_connectome=batch["sent_connectome"].to(device),
                    rel_pos=batch["rel_pos"].to(device),
                    lex_pos=batch["lex_pos"].to(device),
                    word_salience=batch["word_salience"].to(device),
                    cog_load=batch["cog_load"].to(device),
                    task_id=batch["task_id"].to(device),
                    max_text_len=int(cfg["data"]["max_text_len"]),
                    word_dropout_prob=word_drop,
                )
                loss = out["loss"] / grad_accum
                loss.backward()
                running_loss += float(out["loss"].item())

                if step % grad_accum == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            scheduler.step()

            val_metrics, _, _ = evaluate_teacher_forcing_free(
                model,
                val_loader,
                device=device,
                max_new_tokens=int(cfg["data"]["max_text_len"]),
                num_beams=eval_val_beams,
                repetition_penalty=float(s2_cfg["eval_repetition_penalty"]),
                no_repeat_ngram_size=int(s2_cfg["eval_no_repeat_ngram_size"]),
                replace_with_gaussian_noise=False,
                compute_per_subject=False,
            )
            avg_loss = running_loss / max(1, len(train_loader))
            cur_score = (
                float(val_metrics["BLEU-1"])
                + 2.0 * float(val_metrics["BLEU-4"])
                + 0.5 * float(val_metrics["ROUGE-1_F1"])
                + 0.3 * float(val_metrics.get("Semantic_CosSim", 0.0))
            )
            print(
                f"[Stage 2 | Epoch {epoch:02d}/{epochs}] Loss={avg_loss:.4f} (Drop={word_drop:.2f}) | "
                f"Val TF-Free BLEU-1={val_metrics['BLEU-1']:.2f}% BLEU-4={val_metrics['BLEU-4']:.2f}% "
                f"ROUGE-1={val_metrics['ROUGE-1_F1']:.2f}% SemSim={val_metrics.get('Semantic_CosSim', 0.0):.2f}% WER={val_metrics['WER']:.2f}%"
            )
            s2_history.append({"epoch": epoch, "train_loss": round(avg_loss, 4), "val_metrics": val_metrics})
            with open(os.path.join(cfg["experiment"]["log_dir"], "stage2_history.json"), "w", encoding="utf-8") as hf:
                json.dump(s2_history, hf, indent=2)

            if cur_score >= best_score:
                best_score = cur_score
                epochs_no_improve = 0
                torch.save(model.projector.state_dict(), bridge_ckpt)
                torch.save(model.cet_rvq.state_dict(), cet_ckpt)
                if bool(s2_cfg["use_lora"]):
                    model.llm.save_pretrained(lora_dir)
                else:
                    torch.save(model.llm.state_dict(), os.path.join(cfg["experiment"]["output_dir"], "stage2_full_llm_best.pt"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"[Stage 2 Early Stop] Validation BLEU/ROUGE plateaued for {patience} epochs.")
                    break

    if os.path.exists(cet_ckpt):
        model.cet_rvq.load_state_dict(torch.load(cet_ckpt, map_location=device, weights_only=True))
    if os.path.exists(bridge_ckpt):
        model.projector.load_state_dict(torch.load(bridge_ckpt, map_location=device, weights_only=True))
    full_llm_ckpt = os.path.join(cfg["experiment"]["output_dir"], "stage2_full_llm_best.pt")
    if bool(s2_cfg["use_lora"]) and os.path.exists(os.path.join(lora_dir, "adapter_config.json")):
        if hasattr(model.llm, "load_adapter"):
            model.llm.load_adapter(lora_dir, adapter_name="default")
            model.llm.set_adapter("default")
        else:
            from peft import PeftModel
            base_llm = model.llm.get_base_model() if hasattr(model.llm, "get_base_model") else model.llm
            model.llm = PeftModel.from_pretrained(base_llm, lora_dir)
        if s2_cfg["quantization"] != "nf4":
            model.llm.to(device)
    elif (not bool(s2_cfg["use_lora"])) and os.path.exists(full_llm_ckpt):
        model.llm.load_state_dict(torch.load(full_llm_ckpt, map_location=device, weights_only=True))
        model.llm.to(device)

    print("\n=== Final Test Set Evaluation (Strict Teacher-Forcing-Free + Per-Subject Breakdown) ===")
    eeg_metrics, eeg_examples, eeg_extra = evaluate_teacher_forcing_free(
        model,
        test_loader,
        device=device,
        max_new_tokens=int(cfg["data"]["max_text_len"]),
        num_beams=int(s2_cfg["eval_num_beams"]),
        repetition_penalty=float(s2_cfg["eval_repetition_penalty"]),
        no_repeat_ngram_size=int(s2_cfg["eval_no_repeat_ngram_size"]),
        replace_with_gaussian_noise=False,
        compute_per_subject=True,
    )
    print(f"[Real EEG Input]  {eeg_metrics}")

    report: dict[str, object] = {
        "experiment": cfg["experiment"]["name"],
        "split_strategy": cfg["data"]["split_strategy"],
        "eeg_only_mode": eeg_only_mode,
        "llm_backbone": s2_cfg["llm_name_or_path"],
        "real_eeg_tf_free_metrics": eeg_metrics,
        "per_subject_real_eeg": eeg_extra.get("per_subject", {}),
        "decoded_examples": eeg_examples,
    }

    if bool(s2_cfg.get("run_gaussian_noise_sanity_check", True)):
        noise_metrics, _, noise_extra = evaluate_teacher_forcing_free(
            model,
            test_loader,
            device=device,
            max_new_tokens=int(cfg["data"]["max_text_len"]),
            num_beams=int(s2_cfg["eval_num_beams"]),
            repetition_penalty=float(s2_cfg["eval_repetition_penalty"]),
            no_repeat_ngram_size=int(s2_cfg["eval_no_repeat_ngram_size"]),
            replace_with_gaussian_noise=True,
            compute_per_subject=True,
        )
        print(f"[Gaussian Noise]  {noise_metrics}")
        report["gaussian_noise_metrics"] = noise_metrics
        report["per_subject_gaussian_noise"] = noise_extra.get("per_subject", {})
        report["sanity_gap_bleu1"] = round(eeg_metrics["BLEU-1"] - noise_metrics["BLEU-1"], 2)
        report["sanity_gap_bleu4"] = round(eeg_metrics["BLEU-4"] - noise_metrics["BLEU-4"], 2)
        report["sanity_gap_semsim"] = round(
            eeg_metrics.get("Semantic_CosSim", 0.0) - noise_metrics.get("Semantic_CosSim", 0.0), 2
        )

        # Compute Per-Subject Sanity Gaps & Mean +/- SD across unseen subjects!
        ps_real = eeg_extra.get("per_subject", {})
        ps_noise = noise_extra.get("per_subject", {})
        ps_summary: dict[str, Any] = {}
        for sub_id, r_m in ps_real.items():
            n_m = ps_noise.get(sub_id, {})
            ps_summary[sub_id] = {
                "n_trials": r_m.get("n_trials", 0),
                "BLEU-1": r_m["BLEU-1"],
                "BLEU-4": r_m["BLEU-4"],
                "ROUGE-1_F1": r_m["ROUGE-1_F1"],
                "WER": r_m["WER"],
                "Semantic_CosSim": r_m.get("Semantic_CosSim", 0.0),
                "Noise_BLEU-1": n_m.get("BLEU-1", 0.0),
                "Noise_BLEU-4": n_m.get("BLEU-4", 0.0),
                "Delta_BLEU-1": round(r_m["BLEU-1"] - n_m.get("BLEU-1", 0.0), 2),
                "Delta_BLEU-4": round(r_m["BLEU-4"] - n_m.get("BLEU-4", 0.0), 2),
            }
        report["per_subject_summary_table"] = ps_summary

        if len(ps_summary) >= 2:
            agg_stats: dict[str, Any] = {}
            for k_stat in ("BLEU-1", "BLEU-4", "ROUGE-1_F1", "WER", "Semantic_CosSim", "Delta_BLEU-1", "Delta_BLEU-4"):
                arr = [float(v[k_stat]) for v in ps_summary.values()]
                agg_stats[k_stat] = {
                    "mean": round(float(np.mean(arr)), 2),
                    "std": round(float(np.std(arr, ddof=1)), 2),
                }
            report["per_subject_mean_std"] = agg_stats

        print(
            f"[Sanity Check Gap] Delta BLEU-1 = +{report['sanity_gap_bleu1']}% | "
            f"Delta BLEU-4 = +{report['sanity_gap_bleu4']}% | Delta SemSim = +{report['sanity_gap_semsim']}%"
        )

    out_report_name = "test_evaluation_report_per_subject.json" if args.eval_only else "test_evaluation_report.json"
    out_report = os.path.join(cfg["experiment"]["log_dir"], out_report_name)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[Saved Evaluation Report] -> {out_report}")


if __name__ == "__main__":
    main()
