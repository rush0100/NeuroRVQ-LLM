"""
Automated Dual-Stream Connectome + Full 8-Metadata Stage 1 Ablation & Optimization Loop on NVIDIA DGX Spark.

Runs on the 11-subject ZuCo 1.0 `task1-SR` Connectome dataset (`./datasets/processed_zuco_connectome_11sub`,
Train=3417 across 9 subjects, Val=338 on unseen `ZJS`, Test=385 on unseen `ZMG`) with zero length leakage:
  - Round 1 (Scientific Stream Ablation — Q1 Paper Table):
      1) `s1_legacy_840_noleak`   : Flat 105-ch x 8-band GD only (with length-leakage removed!)
      2) `s1_node_only_12roi_4stg`: Stream 1 only (3D 12-ROI + FFD/GD/TRT + Reanalysis + L-R diff + Gaze order)
      3) `s1_edge_only_connectome`: Stream 2 only (Welch wPLI + iCoh + Riemannian SPD Tangent + Word Co-fluctuation)
      4) `s1_dual_stream_full8meta`: Full Gated Node + Edge Fusion + All 8 Metadata + Aux Sentiment + CogLoad
  - Round 2 (Contrastive Target & Temperature Geometry):
      Tests `raw_last` vs `centered_last` vs `centered_multilayer`, and fixed/clamped temperature scaling.
  - Round 3 (Encoder Capacity, Regularization & Neuro-Augmentation):
      Sweeps `embed_dim` (`384`, `512`, `768`), depth (`3`, `4`), ROI/edge dropout, and weight decay.
  - Round 4 (Discrete RVQ Bottleneck & Loss-Weight Polish):
      Sweeps `num_latent_queries K` (`16`, `32`), `rvq_num_quantizers M` (`4`, `6`), `codebook_size` (`256`, `512`),
      `quantizer_dropout`, and extended 55-epoch cosine polish.

Saves real-time ranked results to `./logs/stage1_connectome_leaderboard.json` and exports the overall #1
Stage 1 Champion to `./checkpoints/stage1_champion/stage1_cet_rvq_best.pt` (holding before Stage 2).
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
from typing import Any, Dict, List, Tuple

import yaml

from src.train_stage1 import run_stage1_experiment

LEADERBOARD_PATH = "./logs/stage1_connectome_leaderboard.json"
CHAMPION_DIR = "./checkpoints/stage1_champion"
CHAMPION_CONFIG_PATH = "./configs/stage1_champion.yaml"


def base_connectome_config() -> Dict[str, Any]:
    return {
        "experiment": {
            "name": "stage1_connectome_trial",
            "seed": 42,
            "output_dir": "./checkpoints/stage1_connectome/trial",
            "log_dir": "./logs/stage1_connectome/trial",
        },
        "hardware": {
            "device": "cuda",
            "precision": "bf16",
            "uma_drop_caches_hint": False,
            "num_workers": 4,
        },
        "data": {
            "dataset_root": "./datasets/processed_zuco_connectome_11sub",
            "raw_zuco_root": "./datasets/zuco_raw",
            "tasks": ["task1-SR"],
            "eeg_type": "GD",
            "bands": ["_t1", "_t2", "_a1", "_a2", "_b1", "_b2", "_g1", "_g2"],
            "num_channels": 105,
            "eeg_feature_dim": 840,
            "max_eeg_len": 58,
            "max_text_len": 64,
            "split_strategy": "subject_independent",
            "held_out_test_subjects": ["ZMG"],
            "held_out_val_subjects": ["ZJS"],
        },
        "stage1_cet_rvq": {
            "stream_mode": "dual",
            "embed_dim": 512,
            "encoder_layers": 4,
            "encoder_heads": 8,
            "ffn_dim": 1536,
            "dropout": 0.15,
            "proj_dropout": 0.15,
            "use_length_invariant_resampler": True,
            "num_latent_queries": 32,
            "use_attentive_pooler": True,
            "use_task_conditioning": True,
            "use_band_topography_norm": False,
            "channel_dropout": 0.15,
            "band_jitter": 0.10,
            "sensor_noise": 0.04,
            "token_dropout": 0.10,
            "rvq_num_quantizers": 4,
            "rvq_codebook_size": 512,
            "rvq_commitment_weight": 0.25,
            "rvq_quantizer_dropout": 0.20,
            "rvq_ema_decay": 0.98,
            "eeg_mask_ratio": 0.75,
            "compulsory_sentence_token_mask": True,
            "text_target_mode": "raw_last",
            "zca_reg": 0.05,
            "use_multi_positive_infonce": True,
            "paired_subject_sampling": True,
            "use_cog_load_weighting": True,
            "init_temperature": 0.07,
            "learnable_temperature": True,
            "max_logit_scale": 14.2857,
            "mae_loss_weight": 0.35,
            "contrastive_loss_weight": 1.0,
            "rvq_loss_weight": 1.0,
            "pre_rvq_contrast_ratio": 0.30,
            "brain_invariance_weight": 0.20,
            "aux_sentiment_weight": 0.25,
            "label_smoothing": 0.05,
            "epochs": 40,
            "warmup_epochs": 4,
            "early_stop_patience": 12,
            "batch_size": 64,
            "lr": 3.5e-4,
            "weight_decay": 0.04,
        },
        "stage2_llm": {
            "llm_name_or_path": "./models/Qwen2.5-7B-Instruct",
            "quantization": "none",
            "freeze_cet_rvq": False,
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "scheduled_word_dropout_max": 0.50,
            "prompt_template": "Decode the following neural brain signal tokens into an exact English sentence:\n<eeg_prefix>\nDecoded sentence:",
            "epochs": 15,
            "batch_size": 16,
            "grad_accum_steps": 2,
            "lr_projector": 2.0e-4,
            "lr_eeg_encoder": 3.0e-5,
            "lr_lora": 2.0e-5,
            "weight_decay": 0.01,
            "eval_num_beams": 4,
            "eval_repetition_penalty": 1.25,
            "eval_no_repeat_ngram_size": 3,
            "run_gaussian_noise_sanity_check": True,
        },
    }


def composite_val_objective(summary: Dict[str, Any]) -> float:
    post = summary["best_val_unseen_subject"]["post_rvq"]
    sent_acc = float(summary["best_val_unseen_subject"].get("sentiment_acc", 0.0))
    return float(post["MRR"]) + 0.35 * float(post["R@10"]) + 0.04 * sent_acc - 2.0 * float(post["CE_400way"])


def save_leaderboard(entries: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(LEADERBOARD_PATH), exist_ok=True)
    ranked = sorted(entries, key=lambda x: x["val_objective_score"], reverse=True)
    for rank_idx, item in enumerate(ranked, start=1):
        item["rank"] = rank_idx
    with open(LEADERBOARD_PATH, "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2)


def execute_trial(
    trial_name: str,
    round_label: str,
    base_s1: Dict[str, Any],
    overrides: Dict[str, Any],
    entries: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg = base_connectome_config()
    s1 = copy.deepcopy(base_s1)
    s1.update(overrides)
    cfg["stage1_cet_rvq"] = s1
    cfg["experiment"]["name"] = trial_name
    cfg["experiment"]["output_dir"] = f"./checkpoints/stage1_connectome/{trial_name}"
    cfg["experiment"]["log_dir"] = f"./logs/stage1_connectome/{trial_name}"

    print("\n" + "=" * 90)
    print(f">>> [{round_label}] Trial `{trial_name}` | Overrides: {overrides}")
    print("=" * 90)

    summary = run_stage1_experiment(cfg)
    obj_score = round(composite_val_objective(summary), 4)

    val_post = summary["best_val_unseen_subject"]["post_rvq"]
    val_pre = summary["best_val_unseen_subject"]["pre_rvq"]
    test_post = summary["test_unseen_subject_real_eeg"]["post_rvq"]
    noise_post = summary["test_unseen_subject_gaussian_noise"]["post_rvq"]

    entry = {
        "trial_name": trial_name,
        "round": round_label,
        "stream_mode": s1["stream_mode"],
        "val_objective_score": obj_score,
        "best_epoch": summary["best_epoch"],
        "elapsed_sec": summary["elapsed_sec"],
        "val_post_rvq": val_post,
        "val_pre_rvq": val_pre,
        "val_sentiment_acc": summary["best_val_unseen_subject"]["sentiment_acc"],
        "val_perplexity": summary["best_val_unseen_subject"]["mean_perplexity"],
        "val_layer_perplexity": summary["best_val_unseen_subject"]["per_layer_perplexity"],
        "test_zmg_real_post_rvq": test_post,
        "test_zmg_sentiment_acc": summary["test_unseen_subject_real_eeg"]["sentiment_acc"],
        "test_zmg_noise_post_rvq": noise_post,
        "test_sanity_gap_mrr": summary["test_sanity_gap_mrr"],
        "test_sanity_gap_r10": summary["test_sanity_gap_r10"],
        "test_sanity_gap_ce400": summary["test_sanity_gap_ce400"],
        "overrides": overrides,
        "checkpoint_path": os.path.join(cfg["experiment"]["output_dir"], "stage1_cet_rvq_best.pt"),
        "full_config": cfg,
    }
    entries.append(entry)
    save_leaderboard(entries)

    best_so_far = max(entries, key=lambda x: x["val_objective_score"])
    if best_so_far["trial_name"] == trial_name:
        os.makedirs(CHAMPION_DIR, exist_ok=True)
        shutil.copy2(entry["checkpoint_path"], os.path.join(CHAMPION_DIR, "stage1_cet_rvq_best.pt"))
        champ_cfg = copy.deepcopy(cfg)
        champ_cfg["experiment"]["name"] = f"stage1_champion_{trial_name}"
        champ_cfg["experiment"]["output_dir"] = CHAMPION_DIR
        with open(CHAMPION_CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(champ_cfg, f, sort_keys=False)
        print(
            f"*** NEW #1 STAGE 1 CHAMPION: `{trial_name}` | "
            f"Val(MRR={val_post['MRR']:.2f}% R@10={val_post['R@10']:.2f}% SentAcc={entry['val_sentiment_acc']:.1f}%) | "
            f"Test ZMG(Real MRR={test_post['MRR']:.2f}% vs Noise={noise_post['MRR']:.2f}% -> Gap=+{entry['test_sanity_gap_mrr']:.2f}%) ***"
        )

    return s1, entry


def pick_best_s1(candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
    winner_s1, winner_entry = max(candidates, key=lambda pair: pair[1]["val_objective_score"])
    print(
        f"\n---> Promoting `{winner_entry['trial_name']}` "
        f"(Val MRR={winner_entry['val_post_rvq']['MRR']:.2f}%, R@10={winner_entry['val_post_rvq']['R@10']:.2f}%, "
        f"Test Sanity Gap=+{winner_entry['test_sanity_gap_mrr']:.2f}%) to next round!\n"
    )
    return copy.deepcopy(winner_s1)


def main() -> None:
    t0 = time.time()
    entries: List[Dict[str, Any]] = []
    if os.path.exists(LEADERBOARD_PATH):
        with open(LEADERBOARD_PATH, "r", encoding="utf-8") as f:
            entries = json.load(f)
    done_names = {e["trial_name"] for e in entries}

    def run_or_reuse(
        trial_name: str, round_label: str, base_s1: Dict[str, Any], overrides: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if trial_name in done_names:
            existing = next(e for e in entries if e["trial_name"] == trial_name)
            print(f"[Skip Completed] `{trial_name}` -> Score={existing['val_objective_score']}")
            return copy.deepcopy(existing["full_config"]["stage1_cet_rvq"]), existing
        return execute_trial(trial_name, round_label, base_s1, overrides, entries)

    base_s1 = base_connectome_config()["stage1_cet_rvq"]

    # =========================================================================
    # ROUND 1: Core Stream & Metadata Ablation Study (Paper 1 Table)
    # =========================================================================
    r1_cands = []
    r1_cands.append(
        run_or_reuse(
            "r1_ablation_legacy840_noleak",
            "Round 1: Stream Ablation",
            base_s1,
            {
                "stream_mode": "legacy_840",
                "use_task_conditioning": False,
                "use_cog_load_weighting": False,
                "aux_sentiment_weight": 0.0,
            },
        )
    )
    r1_cands.append(
        run_or_reuse(
            "r1_ablation_node_only_12roi",
            "Round 1: Stream Ablation",
            base_s1,
            {"stream_mode": "node_only"},
        )
    )
    r1_cands.append(
        run_or_reuse(
            "r1_ablation_edge_only_connectome",
            "Round 1: Stream Ablation",
            base_s1,
            {"stream_mode": "edge_only"},
        )
    )
    r1_cands.append(
        run_or_reuse(
            "r1_dual_stream_full8meta",
            "Round 1: Stream Ablation",
            base_s1,
            {"stream_mode": "dual"},
        )
    )
    r1_champ = pick_best_s1(r1_cands)

    # =========================================================================
    # ROUND 2: Target Embedding Geometry & Temperature Calibration
    # =========================================================================
    r2_cands = [(r1_champ, max(entries, key=lambda x: x["val_objective_score"]))]
    r2_cands.append(
        run_or_reuse(
            "r2a_centered_last_fixed_tau08",
            "Round 2: Target & Temp",
            r1_champ,
            {"text_target_mode": "centered_last", "learnable_temperature": False, "init_temperature": 0.08},
        )
    )
    r2_cands.append(
        run_or_reuse(
            "r2b_raw_last_fixed_tau09",
            "Round 2: Target & Temp",
            r1_champ,
            {"text_target_mode": "raw_last", "learnable_temperature": False, "init_temperature": 0.09},
        )
    )
    r2_cands.append(
        run_or_reuse(
            "r2c_centered_multi_tau10",
            "Round 2: Target & Temp",
            r1_champ,
            {"text_target_mode": "centered_multilayer", "learnable_temperature": False, "init_temperature": 0.10},
        )
    )
    r2_champ = pick_best_s1(r2_cands)

    # =========================================================================
    # ROUND 3: Encoder Capacity & Neuro-Regularization Sweep
    # =========================================================================
    r3_cands = [(r2_champ, max(entries, key=lambda x: x["val_objective_score"]))]
    r3_cands.append(
        run_or_reuse(
            "r3a_compact384_3layer_strongreg",
            "Round 3: Capacity & Reg",
            r2_champ,
            {
                "embed_dim": 384,
                "encoder_layers": 3,
                "encoder_heads": 6,
                "ffn_dim": 1152,
                "dropout": 0.22,
                "proj_dropout": 0.20,
                "channel_dropout": 0.20,
                "band_jitter": 0.12,
                "weight_decay": 0.06,
                "lr": 4.0e-4,
            },
        )
    )
    r3_cands.append(
        run_or_reuse(
            "r3b_wide768_4layer_modreg",
            "Round 3: Capacity & Reg",
            r2_champ,
            {
                "embed_dim": 768,
                "encoder_layers": 4,
                "encoder_heads": 8,
                "ffn_dim": 2048,
                "dropout": 0.18,
                "proj_dropout": 0.18,
                "channel_dropout": 0.15,
                "weight_decay": 0.05,
                "lr": 3.0e-4,
            },
        )
    )
    r3_champ = pick_best_s1(r3_cands)

    # =========================================================================
    # ROUND 4: RVQ Bottleneck Hierarchy & Extended Cosine Polish
    # =========================================================================
    r4_cands = [(r3_champ, max(entries, key=lambda x: x["val_objective_score"]))]
    r4_cands.append(
        run_or_reuse(
            "r4a_rvq_k24_m6_cb256_qdrop30",
            "Round 4: RVQ & Polish",
            r3_champ,
            {
                "num_latent_queries": 24,
                "rvq_num_quantizers": 6,
                "rvq_codebook_size": 256,
                "rvq_quantizer_dropout": 0.30,
                "epochs": 55,
                "warmup_epochs": 5,
                "early_stop_patience": 16,
            },
        )
    )
    r4_cands.append(
        run_or_reuse(
            "r4b_rvq_k32_m4_cb256_b32_55ep",
            "Round 4: RVQ & Polish",
            r3_champ,
            {
                "num_latent_queries": 32,
                "rvq_num_quantizers": 4,
                "rvq_codebook_size": 256,
                "rvq_quantizer_dropout": 0.25,
                "mae_loss_weight": 0.25,
                "contrastive_loss_weight": 1.35,
                "brain_invariance_weight": 0.30,
                "aux_sentiment_weight": 0.35,
                "batch_size": 32,
                "lr": 2.5e-4,
                "epochs": 55,
                "warmup_epochs": 5,
                "early_stop_patience": 16,
            },
        )
    )
    r4_champ = pick_best_s1(r4_cands)

    # =========================================================================
    # ROUND 5: Residual Zero-Init Dual-Stream Fusion Synthesis
    # Combines #1 Node+RVQ Champion with Zero-Init Residual Edge & Connectome Adapter!
    # =========================================================================
    r5_cands = [(r4_champ, max(entries, key=lambda x: x["val_objective_score"]))]
    r5_cands.append(
        run_or_reuse(
            "r5a_residual_dual_stream_768_m6",
            "Round 5: Residual Dual Synthesis",
            r4_champ,
            {
                "stream_mode": "dual",
                "mae_loss_weight": 0.30,
                "contrastive_loss_weight": 1.25,
                "brain_invariance_weight": 0.25,
                "aux_sentiment_weight": 0.30,
                "epochs": 55,
                "warmup_epochs": 5,
                "early_stop_patience": 16,
            },
        )
    )
    r5_cands.append(
        run_or_reuse(
            "r5b_residual_dual_k32_m4_cb256",
            "Round 5: Residual Dual Synthesis",
            r4_champ,
            {
                "stream_mode": "dual",
                "num_latent_queries": 32,
                "rvq_num_quantizers": 4,
                "rvq_codebook_size": 256,
                "rvq_quantizer_dropout": 0.25,
                "mae_loss_weight": 0.30,
                "contrastive_loss_weight": 1.25,
                "brain_invariance_weight": 0.25,
                "aux_sentiment_weight": 0.30,
                "epochs": 55,
                "warmup_epochs": 5,
                "early_stop_patience": 16,
            },
        )
    )
    r5_cands.append(
        run_or_reuse(
            "r5c_late_connectome_fusion_768_m6",
            "Round 5: Late Connectome Bottleneck Fusion",
            r4_champ,
            {
                "stream_mode": "dual_late_fusion",
                "epochs": 55,
                "warmup_epochs": 5,
                "early_stop_patience": 16,
            },
        )
    )
    _ = pick_best_s1(r5_cands)

    ranked = sorted(entries, key=lambda x: x["val_objective_score"], reverse=True)
    total_min = (time.time() - t0) / 60.0
    print("\n" + "#" * 104)
    print(f"DUAL-STREAM CONNECTOME + 8-METADATA STAGE 1 LEADERBOARD ({len(ranked)} trials in {total_min:.1f} min)")
    print("#" * 104)
    for row in ranked:
        vp = row["val_post_rvq"]
        tp = row["test_zmg_real_post_rvq"]
        np_ = row["test_zmg_noise_post_rvq"]
        print(
            f"#{row['rank']:02d} | {row['trial_name']:<32} | "
            f"Val(MRR={vp['MRR']:5.2f}% R@10={vp['R@10']:5.2f}% Sent={row['val_sentiment_acc']:4.1f}% CE={vp['CE_400way']:.3f} PPL={row['val_perplexity']:5.1f}) | "
            f"TestZMG(Real={tp['MRR']:5.2f}% Noise={np_['MRR']:5.2f}% Gap=+{row['test_sanity_gap_mrr']:5.2f}% R@10={tp['R@10']:5.2f}% Sent={row['test_zmg_sentiment_acc']:4.1f}%)"
        )
    print(f"\nChampion Checkpoint -> {CHAMPION_DIR}/stage1_cet_rvq_best.pt")
    print(f"Champion Config     -> {CHAMPION_CONFIG_PATH}")
    print(f"Full Leaderboard    -> {LEADERBOARD_PATH}")


if __name__ == "__main__":
    main()
