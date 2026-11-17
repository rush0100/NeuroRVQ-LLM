"""
Stage 1 Training & Full-Bank Unseen-Subject Retrieval Evaluation Script (CET-RVQ v3: Dual-Stream + 8-Metadata).
Features:
  - Persistent disk cache for frozen Qwen/Llama hidden states (`raw_last`, `centered_multilayer`, `zca_whitened`)
  - Dual-Stream Node (4-stage 12-ROI + 3D coords + L-R diff) + Edge (`wPLI`, `iCoh`, `Riemannian SPD`, `Co-fluctuation`)
  - Zero-Length-Leakage Normalized Scanpath Progress PE (`rel_pos`, `lex_pos` in `[0, 1]`)
  - Auxiliary Affective Sentiment Accuracy & Loss tracking + Cognitive-Load weighted InfoNCE
  - Strict Pure Gaussian Noise Sanity Check (randomizes nodes, edges, connectomes, & randomized uniform mask lengths!)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.data.dataset import ZuCoEEGTextDataset, collate_eeg_text, load_and_split_zuco
from src.models.cet_rvq_pretrainer import CETRVQPretrainer, multi_positive_infonce


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


class SentencePairedBatchSampler(Sampler[List[int]]):
    """Constructs mini-batches where each chosen sentence contributes 2 distinct subjects reading that sentence."""

    def __init__(self, dataset: ZuCoEEGTextDataset, batch_size: int, seed: int = 42) -> None:
        self.dataset = dataset
        self.batch_size = max(2, batch_size)
        self.seed = seed
        self.epoch = 0

        self.by_sent: Dict[str, List[int]] = defaultdict(list)
        for idx, s in enumerate(dataset.samples):
            self.by_sent[s["target_text"]].append(idx)

        self.num_batches = max(1, len(dataset) // self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        pairs: List[Tuple[int, int]] = []
        leftovers: List[int] = []

        sents = list(self.by_sent.keys())
        rng.shuffle(sents)
        for s in sents:
            idxs = list(self.by_sent[s])
            rng.shuffle(idxs)
            while len(idxs) >= 2:
                pairs.append((idxs.pop(), idxs.pop()))
            if idxs:
                leftovers.append(idxs.pop())

        rng.shuffle(pairs)
        rng.shuffle(leftovers)

        half_b = max(1, self.batch_size // 2)
        for i in range(0, len(pairs), half_b):
            chunk = pairs[i : i + half_b]
            batch: List[int] = []
            for a, b in chunk:
                batch.extend([a, b])
            while len(batch) < self.batch_size and leftovers:
                batch.append(leftovers.pop())
            if len(batch) >= 2:
                yield batch[: self.batch_size]

        while len(leftovers) >= self.batch_size:
            yield [leftovers.pop() for _ in range(self.batch_size)]


def _slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


@torch.no_grad()
def load_or_build_text_cache(
    unique_sentences: List[str],
    dataset_root: str,
    llm_name_or_path: str,
    device: torch.device,
    text_target_mode: str = "raw_last",
    zca_reg: float = 0.05,
    batch_size: int = 32,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, int], int]:
    """Loads or builds persistent frozen LLM sentence cache (checks both current `dataset_root` and `processed_zuco_10sub`)."""
    os.makedirs(dataset_root, exist_ok=True)
    fname = f"llm_stage1_cache_{_slugify(llm_name_or_path)}.pt"
    cache_path = os.path.join(dataset_root, fname)
    alt_cache = os.path.join("./datasets/processed_zuco_10sub", fname)
    full_cache = os.path.join("./datasets/processed_zuco_connectome_full", fname)

    raw_payload: Dict[str, Any] | None = None
    for cand in (cache_path, full_cache, alt_cache):
        if os.path.exists(cand):
            loaded = torch.load(cand, map_location="cpu", weights_only=False)
            cached_sents = loaded.get("sentences", [])
            if cached_sents == unique_sentences:
                raw_payload = loaded
                print(f"[Stage 1] Loaded exact cached frozen LLM features in <0.2s -> {cand}")
                break
            elif set(unique_sentences).issubset(set(cached_sents)):
                idx_lookup = {s: i for i, s in enumerate(cached_sents)}
                sel = torch.tensor([idx_lookup[s] for s in unique_sentences], dtype=torch.long)
                raw_payload = {
                    "sentences": unique_sentences,
                    "hidden_dim": int(loaded["hidden_dim"]),
                    "raw_last": loaded["raw_last"][sel],
                    "raw_multi": loaded["raw_multi"][sel],
                }
                print(f"[Stage 1] Sliced {len(unique_sentences)}/{len(cached_sents)} cached LLM features in <0.2s -> {cand}")
                break

    if raw_payload is None:
        print(f"[Stage 1] Extracting multi-layer frozen text features from `{llm_name_or_path}` ({len(unique_sentences)} sentences)...")
        tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        backbone = AutoModel.from_pretrained(llm_name_or_path, trust_remote_code=True, torch_dtype=dtype).to(device)
        backbone.eval()
        hidden_dim = int(getattr(backbone.config, "d_model", getattr(backbone.config, "hidden_size", 1024)))

        last_vecs: List[torch.Tensor] = []
        multi_vecs: List[torch.Tensor] = []

        for start in tqdm(range(0, len(unique_sentences), batch_size), desc="Caching Multi-Layer LLM Features"):
            batch_sents = unique_sentences[start : start + batch_size]
            tok = tokenizer(batch_sents, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            out = backbone(**tok, output_hidden_states=True)
            mask = tok["attention_mask"].unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp(min=1.0)

            last_h = out.last_hidden_state.float()
            last_vecs.append(((last_h * mask).sum(dim=1) / denom).cpu())

            hs = (
                getattr(out, "hidden_states", None)
                or getattr(out, "decoder_hidden_states", None)
                or getattr(out, "encoder_hidden_states", None)
                or (out.last_hidden_state,)
            )
            layers_to_avg = hs[-4:] if len(hs) >= 4 else hs[-1:]
            avg_h = torch.stack([h.float() for h in layers_to_avg], dim=0).mean(dim=0)
            multi_vecs.append(((avg_h * mask).sum(dim=1) / denom).cpu())

        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()

        raw_payload = {
            "sentences": unique_sentences,
            "hidden_dim": hidden_dim,
            "raw_last": torch.cat(last_vecs, dim=0),
            "raw_multi": torch.cat(multi_vecs, dim=0),
        }
        torch.save(raw_payload, cache_path)
        print(f"[Stage 1] Saved persistent LLM text cache -> {cache_path}")

    hidden_dim = int(raw_payload["hidden_dim"])
    if text_target_mode == "raw_last":
        mat = raw_payload["raw_last"].float()
    elif text_target_mode == "centered_last":
        mat = raw_payload["raw_last"].float()
        mat = F.normalize(mat - mat.mean(dim=0, keepdim=True), dim=-1)
    elif text_target_mode == "centered_multilayer":
        mat = raw_payload["raw_multi"].float()
        mat = F.normalize(mat - mat.mean(dim=0, keepdim=True), dim=-1)
    elif text_target_mode == "zca_whitened":
        mat = raw_payload["raw_multi"].float()
        mat = mat - mat.mean(dim=0, keepdim=True)
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        inv_s = 1.0 / torch.sqrt(s ** 2 / max(1, mat.size(0) - 1) + zca_reg)
        mat = F.normalize((u * inv_s.unsqueeze(0)) @ vh, dim=-1)
    else:
        raise ValueError(f"Unknown text_target_mode: {text_target_mode}")

    sent_to_id = {s: idx for idx, s in enumerate(unique_sentences)}
    text_cache = {s: mat[idx] for idx, s in enumerate(unique_sentences)}
    return text_cache, mat, sent_to_id, hidden_dim


@torch.no_grad()
def evaluate_stage1_comprehensive(
    model: CETRVQPretrainer,
    dataloader: DataLoader,
    text_bank_matrix: torch.Tensor,
    sent_to_id: Dict[str, int],
    device: torch.device,
    mask_ratio: float = 0.75,
    compulsory_last_token: bool = True,
    fixed_eval_tau: float = 0.07,
    replace_with_gaussian_noise: bool = False,
) -> Dict[str, Any]:
    """
    Temperature-invariant 400-way retrieval + sentiment auxiliary accuracy + pure Gaussian noise sanity check.
    When `replace_with_gaussian_noise=True`, replaces ALL neural/eye features AND randomizes token mask lengths
    so any residual sequence-length cue is completely destroyed as well!
    """
    model.eval()
    bank_proj = model.project_text_bank(text_bank_matrix.to(device))
    fixed_scale = torch.tensor(1.0 / fixed_eval_tau, device=device, dtype=torch.float32)

    all_post_emb: List[torch.Tensor] = []
    all_pre_emb: List[torch.Tensor] = []
    all_gt_ids: List[torch.Tensor] = []

    mae_sum, rvq_sum, batch_infonce_sum, ppl_sum = 0.0, 0.0, 0.0, 0.0
    sent_correct, sent_total = 0, 0
    layer_ppl_sums: List[float] | None = None
    n_batches = 0

    for batch in dataloader:
        eeg = batch["eeg_features"].to(device)
        mask = batch["eeg_attn_mask"].to(device)
        nodes = batch["node_features"].to(device)
        edges = batch["word_coupled_edges"].to(device)
        sconn = batch["sent_connectome"].to(device)
        rpos = batch["rel_pos"].to(device)
        lpos = batch["lex_pos"].to(device)
        sal = batch["word_salience"].to(device)
        cog = batch["cog_load"].to(device)
        tid = batch["task_id"].to(device)
        slbl = batch["sentiment_label"].to(device)

        if replace_with_gaussian_noise:
            bsz, max_l = mask.shape
            # Randomize sequence lengths as well as Gaussian values -> true zero-information baseline!
            rand_mask = torch.zeros_like(mask)
            rand_rpos = torch.zeros_like(rpos)
            for bi in range(bsz):
                rlen = int(torch.randint(6, max(8, max_l // 2), (1,)).item())
                rand_mask[bi, :rlen] = 1
                rand_rpos[bi, :rlen] = torch.linspace(0.0, 1.0, steps=rlen, device=device)
            mask = rand_mask
            rpos = rand_rpos
            lpos = rand_rpos
            sal = mask.float()
            eeg = torch.randn_like(eeg) * mask.unsqueeze(-1).float()
            nodes = torch.randn_like(nodes) * mask.unsqueeze(-1).float()
            edges = torch.randn_like(edges) * mask.unsqueeze(-1).float()
            sconn = torch.randn_like(sconn)
            cog = torch.randn_like(cog)

        gt_ids = torch.tensor([sent_to_id[t] for t in batch["target_text"]], dtype=torch.long, device=device)
        txt_vecs = text_bank_matrix.to(device)[gt_ids]

        fwd = model(
            eeg_features=eeg,
            eeg_attn_mask=mask,
            frozen_text_pooled=txt_vecs,
            node_features=nodes,
            word_coupled_edges=edges,
            sent_connectome=sconn,
            rel_pos=rpos,
            lex_pos=lpos,
            word_salience=sal,
            cog_load=cog,
            task_id=tid,
            sentiment_label=slbl,
            sentence_ids=gt_ids,
            mask_ratio=mask_ratio,
            compulsory_last_token=compulsory_last_token,
            use_cog_load_weighting=False,
            label_smoothing=0.0,
        )
        ret = model.extract_brain_retrieval_embeddings(
            eeg_features=eeg,
            eeg_attn_mask=mask,
            node_features=nodes,
            word_coupled_edges=edges,
            sent_connectome=sconn,
            rel_pos=rpos,
            lex_pos=lpos,
            word_salience=sal,
            cog_load=cog,
            task_id=tid,
        )
        post_emb: torch.Tensor = ret["post_rvq_emb"]          # type: ignore[assignment]
        pre_emb: torch.Tensor = ret["pre_rvq_emb"]            # type: ignore[assignment]
        sent_logits: torch.Tensor = ret["sentiment_logits"]   # type: ignore[assignment]

        valid_s = slbl >= 0
        if bool(valid_s.any().item()):
            preds_s = torch.argmax(sent_logits[valid_s], dim=-1)
            sent_correct += int((preds_s == slbl[valid_s]).sum().item())
            sent_total += int(valid_s.sum().item())

        txt_proj = bank_proj[gt_ids]
        batch_ce = multi_positive_infonce(post_emb, txt_proj, scale=fixed_scale, sentence_ids=gt_ids, label_smoothing=0.0)

        all_post_emb.append(post_emb)
        all_pre_emb.append(pre_emb)
        all_gt_ids.append(gt_ids)

        mae_sum += float(fwd["mae_loss"].item())  # type: ignore[union-attr]
        rvq_sum += float(fwd["rvq_loss"].item())  # type: ignore[union-attr]
        batch_infonce_sum += float(batch_ce.item())
        ppl_sum += float(ret["mean_perplexity"])  # type: ignore[arg-type]

        lp = ret.get("per_layer_perplexity", [])
        if isinstance(lp, list) and lp:
            if layer_ppl_sums is None:
                layer_ppl_sums = [0.0] * len(lp)
            for idx_l, v in enumerate(lp):
                layer_ppl_sums[idx_l] += float(v)
        n_batches += 1

    n_b = max(1, n_batches)
    post_mat = torch.cat(all_post_emb, dim=0)
    pre_mat = torch.cat(all_pre_emb, dim=0)
    gt_vec = torch.cat(all_gt_ids, dim=0)

    def _ranking_stats(query_mat: torch.Tensor) -> Dict[str, float]:
        sims = query_mat @ bank_proj.t()
        logits_400 = fixed_scale * sims
        ce_400 = float(F.cross_entropy(logits_400, gt_vec).item())

        gt_scores = sims.gather(1, gt_vec.unsqueeze(1))
        ranks = (sims > gt_scores).sum(dim=1) + 1
        ranks_f = ranks.float()
        return {
            "R@1": round(float((ranks <= 1).float().mean().item()) * 100.0, 2),
            "R@5": round(float((ranks <= 5).float().mean().item()) * 100.0, 2),
            "R@10": round(float((ranks <= 10).float().mean().item()) * 100.0, 2),
            "MRR": round(float((1.0 / ranks_f).mean().item()) * 100.0, 2),
            "MedR": round(float(ranks_f.median().item()), 1),
            "CE_400way": round(ce_400, 4),
        }

    post_stats = _ranking_stats(post_mat)
    pre_stats = _ranking_stats(pre_mat)

    return {
        "post_rvq": post_stats,
        "pre_rvq": pre_stats,
        "quantization_mrr_gap": round(pre_stats["MRR"] - post_stats["MRR"], 2),
        "sentiment_acc": round((sent_correct / max(1, sent_total)) * 100.0, 2) if sent_total > 0 else 0.0,
        "fixed_tau_batch_infonce": round(batch_infonce_sum / n_b, 4),
        "mae_loss": round(mae_sum / n_b, 4),
        "rvq_loss": round(rvq_sum / n_b, 4),
        "mean_perplexity": round(ppl_sum / n_b, 2),
        "per_layer_perplexity": [round(x / n_b, 1) for x in (layer_ppl_sums or [])],
    }


def run_stage1_experiment(cfg: Dict[str, Any], eval_only: bool = False) -> Dict[str, Any]:
    t_start = time.time()
    set_seed(int(cfg["experiment"]["seed"]))
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

    unique_sents = sorted({s["target_text"] for s in (train_ds.samples + val_ds.samples + test_ds.samples)})
    s1_cfg = cfg["stage1_cet_rvq"]
    text_target_mode = str(s1_cfg.get("text_target_mode", "raw_last"))
    zca_reg = float(s1_cfg.get("zca_reg", 0.05))

    text_teacher_path = str(s1_cfg.get("text_teacher_name_or_path", cfg["stage2_llm"]["llm_name_or_path"]))
    text_cache, text_bank_matrix, sent_to_id, llm_hidden_dim = load_or_build_text_cache(
        unique_sentences=unique_sents,
        dataset_root=cfg["data"]["dataset_root"],
        llm_name_or_path=text_teacher_path,
        device=device,
        text_target_mode=text_target_mode,
        zca_reg=zca_reg,
    )

    model = CETRVQPretrainer(
        eeg_feature_dim=int(cfg["data"]["eeg_feature_dim"]),
        num_channels=int(cfg["data"]["num_channels"]),
        num_bands=len(cfg["data"]["bands"]),
        embed_dim=int(s1_cfg["embed_dim"]),
        llm_hidden_dim=llm_hidden_dim,
        encoder_layers=int(s1_cfg["encoder_layers"]),
        encoder_heads=int(s1_cfg["encoder_heads"]),
        ffn_dim=int(s1_cfg["ffn_dim"]),
        dropout=float(s1_cfg.get("dropout", 0.15)),
        use_resampler=bool(s1_cfg.get("use_length_invariant_resampler", True)),
        num_latent_queries=int(s1_cfg.get("num_latent_queries", 32)),
        rvq_num_quantizers=int(s1_cfg.get("rvq_num_quantizers", 4)),
        rvq_codebook_size=int(s1_cfg.get("rvq_codebook_size", 512)),
        rvq_commitment_weight=float(s1_cfg.get("rvq_commitment_weight", 0.25)),
        rvq_quantizer_dropout=float(s1_cfg.get("rvq_quantizer_dropout", 0.2)),
        rvq_ema_decay=float(s1_cfg.get("rvq_ema_decay", 0.98)),
        max_eeg_len=max(128, int(cfg["data"]["max_eeg_len"]) + 8),
        stream_mode=str(s1_cfg.get("stream_mode", "dual")),
        use_task_conditioning=bool(s1_cfg.get("use_task_conditioning", True)),
        use_band_topography_norm=bool(s1_cfg.get("use_band_topography_norm", False)),
        channel_dropout=float(s1_cfg.get("channel_dropout", 0.15)),
        band_jitter=float(s1_cfg.get("band_jitter", 0.10)),
        sensor_noise=float(s1_cfg.get("sensor_noise", 0.05)),
        token_dropout=float(s1_cfg.get("token_dropout", 0.10)),
        use_attentive_pooler=bool(s1_cfg.get("use_attentive_pooler", True)),
        proj_dropout=float(s1_cfg.get("proj_dropout", 0.15)),
        init_temperature=float(s1_cfg.get("init_temperature", 0.07)),
        learnable_temperature=bool(s1_cfg.get("learnable_temperature", True)),
        max_logit_scale=float(s1_cfg.get("max_logit_scale", 16.0)),
    ).to(device)

    batch_size = int(s1_cfg["batch_size"])
    use_paired_sampler = bool(s1_cfg.get("paired_subject_sampling", True))
    paired_sampler: SentencePairedBatchSampler | None = None

    if use_paired_sampler and len(train_ds) >= batch_size * 2:
        paired_sampler = SentencePairedBatchSampler(train_ds, batch_size=batch_size, seed=int(cfg["experiment"]["seed"]))
        train_loader = DataLoader(
            train_ds,
            batch_sampler=paired_sampler,
            num_workers=int(cfg["hardware"]["num_workers"]),
            collate_fn=collate_eeg_text,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(cfg["hardware"]["num_workers"]),
            collate_fn=collate_eeg_text,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["hardware"]["num_workers"]),
        collate_fn=collate_eeg_text,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["hardware"]["num_workers"]),
        collate_fn=collate_eeg_text,
    )

    epochs = int(s1_cfg["epochs"])
    warmup_epochs = int(s1_cfg.get("warmup_epochs", max(2, epochs // 10)))
    patience = int(s1_cfg.get("early_stop_patience", 14))
    base_lr = float(s1_cfg["lr"])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=float(s1_cfg.get("weight_decay", 0.03))
    )

    def _lr_lambda(ep_zero_based: int) -> float:
        if ep_zero_based < warmup_epochs:
            return float(ep_zero_based + 1) / float(max(1, warmup_epochs))
        progress = float(ep_zero_based - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    best_score = -float("inf")
    best_val_snapshot: Dict[str, Any] = {}
    best_epoch = 0
    epochs_no_improve = 0
    ckpt_path = os.path.join(cfg["experiment"]["output_dir"], "stage1_cet_rvq_best.pt")
    history: List[Dict[str, Any]] = []

    use_multipos = bool(s1_cfg.get("use_multi_positive_infonce", True))
    mae_w = float(s1_cfg.get("mae_loss_weight", 0.4))
    con_w = float(s1_cfg.get("contrastive_loss_weight", 1.0))
    rvq_w = float(s1_cfg.get("rvq_loss_weight", 1.0))
    pre_ratio = float(s1_cfg.get("pre_rvq_contrast_ratio", 0.3))
    b2b_w = float(s1_cfg.get("brain_invariance_weight", 0.25))
    aux_w = float(s1_cfg.get("aux_sentiment_weight", 0.20))
    use_cog_w = bool(s1_cfg.get("use_cog_load_weighting", True))
    lbl_smooth = float(s1_cfg.get("label_smoothing", 0.05))
    mask_ratio = float(s1_cfg.get("eeg_mask_ratio", 0.75))
    comp_last = bool(s1_cfg.get("compulsory_sentence_token_mask", True))

    print(
        f"[Stage 1] Exp=`{cfg['experiment']['name']}` | Stream=`{s1_cfg.get('stream_mode', 'dual')}` "
        f"| EEGOnly=`{eeg_only_mode}` | Split=`{cfg['data']['split_strategy']}` "
        f"| Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)} | Target={text_target_mode}"
    )

    if eval_only and os.path.exists(ckpt_path):
        loaded_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        best_epoch = int(loaded_ckpt.get("best_epoch", 0))
        best_val_snapshot = loaded_ckpt.get("best_val_metrics", {})
        epochs = 0

    for epoch in range(1, epochs + 1):
        if paired_sampler is not None:
            paired_sampler.set_epoch(epoch)

        model.train()
        tr_loss, tr_mae, tr_rvq, tr_con, tr_b2b, tr_aux = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for batch in train_loader:
            eeg = batch["eeg_features"].to(device)
            mask = batch["eeg_attn_mask"].to(device)
            nodes = batch["node_features"].to(device)
            edges = batch["word_coupled_edges"].to(device)
            sconn = batch["sent_connectome"].to(device)
            rpos = batch["rel_pos"].to(device)
            lpos = batch["lex_pos"].to(device)
            sal = batch["word_salience"].to(device)
            cog = batch["cog_load"].to(device)
            tid = batch["task_id"].to(device)
            slbl = batch["sentiment_label"].to(device)

            txt_vecs = torch.stack([text_cache[t] for t in batch["target_text"]], dim=0).to(device)
            sent_ids = (
                torch.tensor([sent_to_id[t] for t in batch["target_text"]], dtype=torch.long, device=device)
                if use_multipos
                else None
            )

            out = model(
                eeg_features=eeg,
                eeg_attn_mask=mask,
                frozen_text_pooled=txt_vecs,
                node_features=nodes,
                word_coupled_edges=edges,
                sent_connectome=sconn,
                rel_pos=rpos,
                lex_pos=lpos,
                word_salience=sal,
                cog_load=cog,
                task_id=tid,
                sentiment_label=slbl,
                sentence_ids=sent_ids,
                mask_ratio=mask_ratio,
                compulsory_last_token=comp_last,
                mae_weight=mae_w,
                contrast_weight=con_w,
                rvq_weight=rvq_w,
                pre_rvq_contrast_ratio=pre_ratio,
                brain_invariance_weight=b2b_w,
                aux_sentiment_weight=aux_w,
                use_cog_load_weighting=use_cog_w,
                label_smoothing=lbl_smooth,
            )
            optimizer.zero_grad()
            out["loss"].backward()  # type: ignore[union-attr]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += float(out["loss"].item())          # type: ignore[union-attr]
            tr_mae += float(out["mae_loss"].item())       # type: ignore[union-attr]
            tr_rvq += float(out["rvq_loss"].item())       # type: ignore[union-attr]
            tr_con += float(out["contrast_loss"].item())  # type: ignore[union-attr]
            tr_b2b += float(out["b2b_loss"].item())       # type: ignore[union-attr]
            tr_aux += float(out["aux_loss"].item())       # type: ignore[union-attr]

        scheduler.step()
        n_tr = max(1, len(train_loader))

        val_eval = evaluate_stage1_comprehensive(
            model=model,
            dataloader=val_loader,
            text_bank_matrix=text_bank_matrix,
            sent_to_id=sent_to_id,
            device=device,
            mask_ratio=mask_ratio,
            compulsory_last_token=comp_last,
        )
        post_v = val_eval["post_rvq"]
        pre_v = val_eval["pre_rvq"]

        score = float(post_v["MRR"]) + 0.35 * float(post_v["R@10"]) - 2.0 * float(post_v["CE_400way"])

        print(
            f"[Ep {epoch:02d}/{epochs}] "
            f"Tr(MAE={tr_mae/n_tr:.3f} Con={tr_con/n_tr:.3f} B2B={tr_b2b/n_tr:.3f} Aux={tr_aux/n_tr:.3f}) | "
            f"Val400(CE={post_v['CE_400way']:.4f} R@1={post_v['R@1']:.2f}% R@5={post_v['R@5']:.2f}% "
            f"R@10={post_v['R@10']:.2f}% MRR={post_v['MRR']:.2f}% [Pre={pre_v['MRR']:.2f}%] "
            f"SentAcc={val_eval['sentiment_acc']:.1f}% PPL={val_eval['mean_perplexity']:.1f})"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(tr_loss / n_tr, 4),
                "val": val_eval,
                "selection_score": round(score, 4),
            }
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val_snapshot = val_eval
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "llm_hidden_dim": llm_hidden_dim,
                    "config": cfg,
                    "best_epoch": best_epoch,
                    "best_val_metrics": best_val_snapshot,
                    "best_val_loss": float(val_eval["fixed_tau_batch_infonce"]),
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Stage 1 Early Stop] No improvement for {patience} epochs (best epoch={best_epoch}).")
                break

    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_real = evaluate_stage1_comprehensive(
        model=model,
        dataloader=test_loader,
        text_bank_matrix=text_bank_matrix,
        sent_to_id=sent_to_id,
        device=device,
        mask_ratio=mask_ratio,
        compulsory_last_token=comp_last,
        replace_with_gaussian_noise=False,
    )
    test_noise = evaluate_stage1_comprehensive(
        model=model,
        dataloader=test_loader,
        text_bank_matrix=text_bank_matrix,
        sent_to_id=sent_to_id,
        device=device,
        mask_ratio=mask_ratio,
        compulsory_last_token=comp_last,
        replace_with_gaussian_noise=True,
    )

    # Per-Subject Unseen Test Breakdown (`ZMG`, `ZPH`, `YTL`, `YSL` + Mean +/- SD)
    test_subjects_sorted = sorted({s["subject_id"] for s in test_ds.samples})
    per_subject_breakdown: Dict[str, Any] = {}
    for sub_id in test_subjects_sorted:
        sub_samples = [s for s in test_ds.samples if s["subject_id"] == sub_id]
        sub_ds = ZuCoEEGTextDataset(sub_samples, eeg_only_mode=eeg_only_mode)
        sub_loader = DataLoader(sub_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_eeg_text)
        sub_real = evaluate_stage1_comprehensive(
            model=model,
            dataloader=sub_loader,
            text_bank_matrix=text_bank_matrix,
            sent_to_id=sent_to_id,
            device=device,
            mask_ratio=mask_ratio,
            compulsory_last_token=comp_last,
            replace_with_gaussian_noise=False,
        )
        sub_noise = evaluate_stage1_comprehensive(
            model=model,
            dataloader=sub_loader,
            text_bank_matrix=text_bank_matrix,
            sent_to_id=sent_to_id,
            device=device,
            mask_ratio=mask_ratio,
            compulsory_last_token=comp_last,
            replace_with_gaussian_noise=True,
        )
        per_subject_breakdown[sub_id] = {
            "n_trials": len(sub_samples),
            "tasks": sorted({s["task_name"] for s in sub_samples}),
            "real_eeg": sub_real["post_rvq"],
            "gaussian_noise": sub_noise["post_rvq"],
            "sanity_gap_mrr": round(sub_real["post_rvq"]["MRR"] - sub_noise["post_rvq"]["MRR"], 2),
            "sanity_gap_r10": round(sub_real["post_rvq"]["R@10"] - sub_noise["post_rvq"]["R@10"], 2),
            "sentiment_acc": sub_real["sentiment_acc"],
        }

    per_sub_agg: Dict[str, Any] = {}
    if len(per_subject_breakdown) >= 2:
        for k_m in ("R@1", "R@5", "R@10", "MRR", "MedR"):
            vals = [v["real_eeg"][k_m] for v in per_subject_breakdown.values()]
            per_sub_agg[k_m] = {
                "mean": round(float(np.mean(vals)), 2),
                "std": round(float(np.std(vals, ddof=1)), 2),
            }
        gap_vals = [v["sanity_gap_mrr"] for v in per_subject_breakdown.values()]
        per_sub_agg["sanity_gap_mrr"] = {
            "mean": round(float(np.mean(gap_vals)), 2),
            "std": round(float(np.std(gap_vals, ddof=1)), 2),
        }

    # If strict unseen-sentence split, ALSO evaluate pure zero-shot retrieval restricted to the unseen test sentence pool!
    unseen_sent_pool_eval: Dict[str, Any] | None = None
    test_unique_sents = sorted({s["target_text"] for s in test_ds.samples})
    if len(test_unique_sents) < len(unique_sents) and len(test_unique_sents) >= 10:
        sub_bank_ids = torch.tensor([sent_to_id[s] for s in test_unique_sents], dtype=torch.long)
        sub_bank_mat = text_bank_matrix[sub_bank_ids]
        sub_sent_map = {s: i for i, s in enumerate(test_unique_sents)}
        unseen_real = evaluate_stage1_comprehensive(
            model=model,
            dataloader=test_loader,
            text_bank_matrix=sub_bank_mat,
            sent_to_id=sub_sent_map,
            device=device,
            mask_ratio=mask_ratio,
            compulsory_last_token=comp_last,
            replace_with_gaussian_noise=False,
        )
        unseen_noise = evaluate_stage1_comprehensive(
            model=model,
            dataloader=test_loader,
            text_bank_matrix=sub_bank_mat,
            sent_to_id=sub_sent_map,
            device=device,
            mask_ratio=mask_ratio,
            compulsory_last_token=comp_last,
            replace_with_gaussian_noise=True,
        )
        unseen_sent_pool_eval = {
            "candidate_pool_size": len(test_unique_sents),
            "real_eeg": unseen_real["post_rvq"],
            "gaussian_noise": unseen_noise["post_rvq"],
            "sanity_gap_mrr": round(unseen_real["post_rvq"]["MRR"] - unseen_noise["post_rvq"]["MRR"], 2),
            "sanity_gap_r10": round(unseen_real["post_rvq"]["R@10"] - unseen_noise["post_rvq"]["R@10"], 2),
            "sentiment_acc": unseen_real["sentiment_acc"],
        }

    elapsed_s = round(time.time() - t_start, 1)
    summary = {
        "experiment": cfg["experiment"]["name"],
        "stream_mode": s1_cfg.get("stream_mode", "dual"),
        "eeg_only_mode": eeg_only_mode,
        "split_strategy": cfg["data"]["split_strategy"],
        "best_epoch": best_epoch,
        "elapsed_sec": elapsed_s,
        "num_unique_sentences": len(unique_sents),
        "best_val_unseen_subject": best_val_snapshot,
        "test_unseen_subject_real_eeg": test_real,
        "test_unseen_subject_gaussian_noise": test_noise,
        "test_sanity_gap_mrr": round(test_real["post_rvq"]["MRR"] - test_noise["post_rvq"]["MRR"], 2),
        "test_sanity_gap_r10": round(test_real["post_rvq"]["R@10"] - test_noise["post_rvq"]["R@10"], 2),
        "test_sanity_gap_ce400": round(test_noise["post_rvq"]["CE_400way"] - test_real["post_rvq"]["CE_400way"], 4),
        "per_subject_test_breakdown": per_subject_breakdown,
        "per_subject_mean_std": per_sub_agg,
        "unseen_sentence_pool_eval": unseen_sent_pool_eval,
        "stage1_config": s1_cfg,
    }

    out_summary_name = "stage1_per_subject_eval.json" if eval_only else "stage1_summary.json"
    if not eval_only:
        with open(os.path.join(cfg["experiment"]["log_dir"], "stage1_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    with open(os.path.join(cfg["experiment"]["log_dir"], out_summary_name), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    val_mrr_disp = best_val_snapshot.get("post_rvq", {}).get("MRR", 0.0) if isinstance(best_val_snapshot, dict) else 0.0
    val_r10_disp = best_val_snapshot.get("post_rvq", {}).get("R@10", 0.0) if isinstance(best_val_snapshot, dict) else 0.0
    print(
        f"[Stage 1 Done in {elapsed_s}s | BestEp={best_epoch}] "
        f"Val MRR={val_mrr_disp:.2f}% R@10={val_r10_disp:.2f}% | "
        f"Test Real MRR={test_real['post_rvq']['MRR']:.2f}% R@10={test_real['post_rvq']['R@10']:.2f}% "
        f"SentAcc={test_real['sentiment_acc']:.1f}% vs Noise MRR={test_noise['post_rvq']['MRR']:.2f}% "
        f"(Sanity Delta MRR=+{summary['test_sanity_gap_mrr']:.2f}%)"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Dual-Stream + 8-Metadata CET-RVQ Pretrainer")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--eval-only", action="store_true", help="Evaluate existing Stage 1 checkpoint including per-subject breakdown")
    parser.add_argument("--device", type=str, default=None, help="Optional device override (e.g., cpu or cuda)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.device:
        cfg["hardware"]["device"] = args.device
    run_stage1_experiment(cfg, eval_only=args.eval_only)


if __name__ == "__main__":
    main()
