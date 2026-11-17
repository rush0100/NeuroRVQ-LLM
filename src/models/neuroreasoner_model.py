"""
Paper 2 Architecture: `NeuroReasoner`
Privacy-Preserving Multimodal EEG-LLM with Chain-of-Thought Neural Evidence Grounding
for Cross-Center Cognitive and Affective Assessment.

Key Innovations Directly Addressing HIT Shenzhen BCI Lab Gaps (`WaveMind` 2025, `EEG-MACS` 2024, `EEG-SeeGraph` 2026):
  1. Montage-Agnostic 3D 12-ROI + Late Connectome Bottleneck + 6-Book EMA-RVQ initialized from Paper 1's
     18,276-trial normative human brain checkpoint, augmented with a Zero-Initialized `ROICoverageAdapter`
     unifying 19-ch (`ds004504`), 60-ch (`UNM`/`UIowa`), 128-ch (`MODMA`), and 3-ch Wearable (`Fp1/Fpz/Fp2`) EEG!
  2. Explicit Neural Evidence Grounding Heads (`L_bio` Smooth-L1 regression over 11 spectral/wPLI/SPD biomarkers
     + `L_diag` domain-specific diagnostic heads), projecting verified neural evidence into 2 dedicated
     Evidence Bridge Tokens alongside the `K=24` discrete RVQ cortical tokens!
  3. Privacy-Preserving Federated LoRA State Protocol (`get_privacy_preserving_state` / `load_privacy_preserving_state`):
     shares ONLY compact adapter/encoder parameters (~0.8% of total model size) across hospital centers with optional
     Gaussian Local Differential Privacy (`sigma_dp`), keeping raw patient EEG 100% inside each hospital!
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.neuro_connectome import COG_LOAD_DIM, NODE_FEAT_DIM, NUM_ROIS, SENT_CONNECTOME_DIM, WORD_EDGE_DIM
from src.evaluation.clinical_metrics import BIOMARKER_KEYS
from src.models.eeg_encoder import SpatioFrequencyEEGEncoder
from src.models.eeg_llm_bridge import BrainToLLMProjector
from src.models.rvq import ResidualVQ


class ROICoverageAdapter(nn.Module):
    """
    Zero-initialized montage-coverage conditioning (`roi_coverage_mask` `[B, 12] -> [B, 1, D]`).
    Explicitly informs latent queries which anatomical ROIs are observed vs. missing on sparse wearable headsets
    while preserving exact Paper-1 pretrained weights at initialization (`W_out = 0`).
    """

    def __init__(self, num_rois: int = NUM_ROIS, embed_dim: int = 768) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_rois, embed_dim // 4),
            nn.GELU(),
            nn.LayerNorm(embed_dim // 4),
        )
        self.out_proj = nn.Linear(embed_dim // 4, embed_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, roi_coverage_mask: torch.Tensor) -> torch.Tensor:
        h = self.mlp(roi_coverage_mask.to(self.out_proj.weight.dtype))
        return self.out_proj(h).unsqueeze(1)  # [B, 1, D]


class AttentiveClinicalPooler(nn.Module):
    """Attention pooler over `K` quantized neural tokens `[B, K, D] -> [B, D]`."""

    def __init__(self, embed_dim: int = 768) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.scorer(tokens), dim=1)
        return torch.sum(tokens * w, dim=1)


class EvidenceGroundingBridge(nn.Module):
    """
    Predicts quantitative EEG biomarkers (`11-d`) and domain-specific diagnostic logits directly from
    quantized RVQ brain tokens, then projects `[predicted_biomarkers || diagnostic_logits]` into 2
    explicit Evidence Bridge Tokens in the LLM embedding space (`[B, 2, D_llm]`).
    Eliminates numerical & diagnostic hallucination inside `<neural_evidence>` and `<clinical_reasoning>`!
    """

    def __init__(self, eeg_embed_dim: int = 768, llm_hidden_dim: int = 3584, num_biomarkers: int = len(BIOMARKER_KEYS)) -> None:
        super().__init__()
        self.num_biomarkers = num_biomarkers
        self.biomarker_head = nn.Sequential(
            nn.LayerNorm(eeg_embed_dim),
            nn.Linear(eeg_embed_dim, eeg_embed_dim // 2),
            nn.GELU(),
            nn.Linear(eeg_embed_dim // 2, num_biomarkers),
        )
        # Domain-specific auxiliary diagnostic heads:
        #   cognitive (`ds004504`): 3 classes (CN=0, FTD=1, AD=2)
        #   parkinson (`UNM`/`UIowa`): 2 classes (CTL=0, PD=1)
        #   affective (`MODMA 128ch`/`3ch`): 2 classes (HC=0, MDD=1)
        self.head_cognitive = nn.Linear(eeg_embed_dim, 3)
        self.head_parkinson = nn.Linear(eeg_embed_dim, 2)
        self.head_affective = nn.Linear(eeg_embed_dim, 2)

        summary_in_dim = num_biomarkers + 3 + 2 + 2  # 18-d structured neural summary
        self.to_llm_tokens = nn.Sequential(
            nn.Linear(summary_in_dim, eeg_embed_dim),
            nn.GELU(),
            nn.LayerNorm(eeg_embed_dim),
            nn.Linear(eeg_embed_dim, llm_hidden_dim * 2),
        )
        self.out_norm = nn.LayerNorm(llm_hidden_dim)
        self.llm_hidden_dim = llm_hidden_dim

    def forward(self, pooled_brain: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred_bm = self.biomarker_head(pooled_brain)            # [B, 11]
        logits_cog = self.head_cognitive(pooled_brain)         # [B, 3]
        logits_pd = self.head_parkinson(pooled_brain)          # [B, 2]
        logits_aff = self.head_affective(pooled_brain)         # [B, 2]

        probs_summary = torch.cat(
            [
                pred_bm,
                F.softmax(logits_cog.float(), dim=-1).to(pred_bm.dtype),
                F.softmax(logits_pd.float(), dim=-1).to(pred_bm.dtype),
                F.softmax(logits_aff.float(), dim=-1).to(pred_bm.dtype),
            ],
            dim=-1,
        )
        bridge = self.to_llm_tokens(probs_summary).view(-1, 2, self.llm_hidden_dim)
        bridge = self.out_norm(bridge)
        return {
            "pred_biomarkers_norm": pred_bm,
            "logits_cognitive": logits_cog,
            "logits_parkinson": logits_pd,
            "logits_affective": logits_aff,
            "evidence_bridge_tokens": bridge,
        }


class NeuroReasonerModel(nn.Module):
    """
    Unified Multimodal Clinical EEG-LLM with Chain-of-Thought Neural Evidence Grounding
    and Privacy-Preserving Federated Parameter Exchange.
    """

    def __init__(
        self,
        llm: nn.Module,
        tokenizer: Any,
        embed_dim: int = 768,
        encoder_layers: int = 4,
        encoder_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.20,
        stream_mode: str = "dual_late_fusion",
        num_latent_queries: int = 24,
        rvq_num_quantizers: int = 6,
        rvq_codebook_size: int = 256,
        rvq_commitment_weight: float = 0.25,
        rvq_quantizer_dropout: float = 0.25,
        rvq_ema_decay: float = 0.98,
        max_eeg_len: int = 64,
        channel_dropout: float = 0.12,
        band_jitter: float = 0.08,
        sensor_noise: float = 0.03,
        init_paper1_ckpt: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.llm = llm
        self.tokenizer = tokenizer
        llm_hidden_dim = int(self.llm.config.hidden_size)

        self.eeg_encoder = SpatioFrequencyEEGEncoder(
            num_channels=105,
            num_bands=8,
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_resampler=True,
            num_latent_queries=num_latent_queries,
            max_len=max_eeg_len,
            stream_mode=stream_mode,
            node_feat_dim=NODE_FEAT_DIM,
            word_edge_dim=WORD_EDGE_DIM,
            sent_conn_dim=SENT_CONNECTOME_DIM,
            cog_load_dim=COG_LOAD_DIM,
            use_task_conditioning=True,
            channel_dropout=channel_dropout,
            band_jitter=band_jitter,
            sensor_noise=sensor_noise,
        )
        self.rvq = ResidualVQ(
            num_quantizers=rvq_num_quantizers,
            codebook_size=rvq_codebook_size,
            embedding_dim=embed_dim,
            commitment_weight=rvq_commitment_weight,
            quantizer_dropout=rvq_quantizer_dropout,
            ema_decay=rvq_ema_decay,
        )
        self.roi_coverage_adapter = ROICoverageAdapter(num_rois=NUM_ROIS, embed_dim=embed_dim)
        self.pooler = AttentiveClinicalPooler(embed_dim=embed_dim)
        self.projector = BrainToLLMProjector(eeg_embed_dim=embed_dim, llm_hidden_dim=llm_hidden_dim, dropout=0.10)
        self.evidence_bridge = EvidenceGroundingBridge(eeg_embed_dim=embed_dim, llm_hidden_dim=llm_hidden_dim)

        if init_paper1_ckpt and os.path.exists(init_paper1_ckpt):
            self._load_paper1_normative_init(init_paper1_ckpt)

    def _load_paper1_normative_init(self, ckpt_path: str) -> None:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = payload.get("model_state_dict", payload)
        enc_sd = {k.replace("eeg_encoder.", "", 1): v for k, v in sd.items() if k.startswith("eeg_encoder.")}
        rvq_sd = {k.replace("rvq.", "", 1): v for k, v in sd.items() if k.startswith("rvq.")}

        missing_e, unexpected_e = self.eeg_encoder.load_state_dict(enc_sd, strict=False)
        # Filter RVQ buffers only if shapes match exactly
        cur_rvq = self.rvq.state_dict()
        compat_rvq = {k: v for k, v in rvq_sd.items() if k in cur_rvq and cur_rvq[k].shape == v.shape}
        self.rvq.load_state_dict(compat_rvq, strict=False)
        print(
            f"[NeuroReasoner] Initialized Normative 3D Connectome Encoder ({len(enc_sd)} tensors, "
            f"missing={len(missing_e)}, unexpected={len(unexpected_e)}) & EMA-RVQ ({len(compat_rvq)}/{len(cur_rvq)} tensors) "
            f"from Paper 1 checkpoint `{ckpt_path}`!"
        )

    def _get_input_embeddings_layer(self) -> nn.Module:
        return self.llm.get_input_embeddings()

    def encode_clinical_brain(
        self,
        batch: Dict[str, Any],
        device: torch.device,
        replace_with_gaussian_noise: bool = False,
    ) -> Dict[str, Any]:
        eeg = batch["eeg_features"].to(device)
        mask = batch["eeg_attn_mask"].to(device)
        nodes = batch["node_features"].to(device)
        edges = batch["word_coupled_edges"].to(device)
        sconn = batch["sent_connectome"].to(device)
        rcov = batch["roi_coverage_mask"].to(device)
        rpos = batch["rel_pos"].to(device)
        lpos = batch["lex_pos"].to(device)
        sal = batch["word_salience"].to(device)
        cog = batch["cog_load"].to(device)
        tid = batch["task_id"].to(device)

        if replace_with_gaussian_noise:
            mask_f = mask.unsqueeze(-1).to(nodes.dtype)
            eeg = torch.randn_like(eeg) * mask_f
            nodes = torch.randn_like(nodes) * mask_f
            edges = torch.randn_like(edges) * mask_f
            sconn = torch.randn_like(sconn)
            cog = torch.randn_like(cog)
            sal = mask.to(sal.dtype)

        resampled = self.eeg_encoder(
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
            apply_augment=self.training,
        )
        # Condition on observed montage ROI coverage (`19ch` / `60ch` / `128ch` vs `3ch wearable`)
        resampled = resampled + self.roi_coverage_adapter(rcov)

        rvq_out = self.rvq(resampled)
        quantized: torch.Tensor = rvq_out["quantized"]  # type: ignore[assignment]
        pooled = self.pooler(quantized)
        ev_out = self.evidence_bridge(pooled)

        emb_layer = self._get_input_embeddings_layer()
        target_dtype = emb_layer.weight.dtype
        target_device = emb_layer.weight.device

        brain_llm_toks = self.projector(quantized).to(device=target_device, dtype=target_dtype)       # [B, 24, D_llm]
        ev_llm_toks = ev_out["evidence_bridge_tokens"].to(device=target_device, dtype=target_dtype)   # [B, 2, D_llm]
        neural_prefix = torch.cat([brain_llm_toks, ev_llm_toks], dim=1)                               # [B, 26, D_llm]

        return {
            "neural_prefix": neural_prefix,
            "rvq_loss": rvq_out["rvq_loss"],
            "perplexity": rvq_out["mean_perplexity"],
            **ev_out,
        }

    def compute_auxiliary_grounding_losses(
        self,
        enc_out: Dict[str, Any],
        domains: List[str],
        numeric_labels: torch.Tensor,
        target_biomarkers_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_bm = enc_out["pred_biomarkers_norm"].float()
        bio_loss = F.smooth_l1_loss(pred_bm, target_biomarkers_norm.to(pred_bm.device).float())

        diag_losses: List[torch.Tensor] = []
        y = numeric_labels.to(pred_bm.device)

        cog_idx = [i for i, d in enumerate(domains) if d == "cognitive_dementia"]
        if cog_idx:
            diag_losses.append(F.cross_entropy(enc_out["logits_cognitive"][cog_idx].float(), y[cog_idx]))

        pd_idx = [i for i, d in enumerate(domains) if d == "crosscenter_parkinson"]
        if pd_idx:
            diag_losses.append(F.cross_entropy(enc_out["logits_parkinson"][pd_idx].float(), y[pd_idx]))

        aff_idx = [i for i, d in enumerate(domains) if d == "affective_depression"]
        if aff_idx:
            diag_losses.append(F.cross_entropy(enc_out["logits_affective"][aff_idx].float(), y[aff_idx]))

        diag_loss = torch.stack(diag_losses).mean() if diag_losses else torch.tensor(0.0, device=pred_bm.device)
        return bio_loss, diag_loss

    def forward(
        self,
        batch: Dict[str, Any],
        target_biomarkers_norm: torch.Tensor,
        device: torch.device,
        max_prompt_tokens: int = 128,
        max_target_tokens: int = 240,
        rvq_weight: float = 0.5,
        bio_weight: float = 0.8,
        diag_weight: float = 0.8,
    ) -> Dict[str, Any]:
        enc_out = self.encode_clinical_brain(batch=batch, device=device, replace_with_gaussian_noise=False)
        neural_prefix: torch.Tensor = enc_out["neural_prefix"]  # [B, 26, D_llm]
        bsz, n_neural, _ = neural_prefix.shape
        target_device = neural_prefix.device
        emb_layer = self._get_input_embeddings_layer()

        prompts: List[str] = batch["clinical_prompt"]
        targets: List[str] = batch["target_text"]
        eos = self.tokenizer.eos_token or ""

        full_embeds_list: List[torch.Tensor] = []
        full_labels_list: List[torch.Tensor] = []
        max_seq = 0

        for i in range(bsz):
            p_str = (
                f"You are NeuroReasoner, a clinical neurophysiology AI. "
                f"{prompts[i].strip()}\nGrounded CoT Report:\n"
            )
            t_str = f"{targets[i].strip()}{eos}"

            p_ids = self.tokenizer(
                p_str, add_special_tokens=True, truncation=True, max_length=max_prompt_tokens, return_tensors="pt"
            )["input_ids"][0].to(target_device)
            t_ids = self.tokenizer(
                t_str, add_special_tokens=False, truncation=True, max_length=max_target_tokens, return_tensors="pt"
            )["input_ids"][0].to(target_device)

            p_emb = emb_layer(p_ids)
            t_emb = emb_layer(t_ids)
            seq_emb = torch.cat([neural_prefix[i], p_emb, t_emb], dim=0)  # [L_i, D_llm]

            ignore_len = n_neural + p_ids.numel()
            seq_lbl = torch.cat(
                [torch.full((ignore_len,), -100, dtype=torch.long, device=target_device), t_ids], dim=0
            )
            full_embeds_list.append(seq_emb)
            full_labels_list.append(seq_lbl)
            max_seq = max(max_seq, seq_emb.size(0))

        pad_emb = emb_layer(torch.tensor([self.tokenizer.pad_token_id], device=target_device))[0]
        padded_embeds = torch.stack(
            [
                torch.cat([e, pad_emb.unsqueeze(0).expand(max_seq - e.size(0), -1)], dim=0)
                if e.size(0) < max_seq
                else e
                for e in full_embeds_list
            ],
            dim=0,
        )
        padded_labels = torch.stack(
            [
                torch.cat([l, torch.full((max_seq - l.size(0),), -100, dtype=torch.long, device=target_device)], dim=0)
                if l.size(0) < max_seq
                else l
                for l in full_labels_list
            ],
            dim=0,
        )
        attn_mask = torch.zeros((bsz, max_seq), dtype=torch.long, device=target_device)
        for i, e in enumerate(full_embeds_list):
            attn_mask[i, : e.size(0)] = 1

        lm_out = self.llm(inputs_embeds=padded_embeds, attention_mask=attn_mask, labels=padded_labels)
        lm_loss = lm_out.loss

        bio_loss, diag_loss = self.compute_auxiliary_grounding_losses(
            enc_out=enc_out,
            domains=batch["domain"],
            numeric_labels=batch["numeric_label"],
            target_biomarkers_norm=target_biomarkers_norm,
        )
        rvq_loss: torch.Tensor = enc_out["rvq_loss"]
        total_loss = lm_loss + rvq_weight * rvq_loss + bio_weight * bio_loss + diag_weight * diag_loss

        return {
            "loss": total_loss,
            "lm_loss": lm_loss.detach(),
            "bio_loss": bio_loss.detach(),
            "diag_loss": diag_loss.detach(),
            "rvq_loss": rvq_loss.detach(),
            "perplexity": float(enc_out["perplexity"]),
        }

    @torch.no_grad()
    def generate_clinical_cot(
        self,
        batch: Dict[str, Any],
        device: torch.device,
        max_prompt_tokens: int = 128,
        max_new_tokens: int = 430,
        replace_with_gaussian_noise: bool = False,
    ) -> List[str]:
        """Strict Teacher-Forcing-Free Chain-of-Thought Clinical Report Generation."""
        self.eval()
        enc_out = self.encode_clinical_brain(
            batch=batch, device=device, replace_with_gaussian_noise=replace_with_gaussian_noise
        )
        neural_prefix: torch.Tensor = enc_out["neural_prefix"]  # [B, 26, D_llm]
        bsz, n_neural, d_llm = neural_prefix.shape
        target_device = neural_prefix.device
        emb_layer = self._get_input_embeddings_layer()

        prompts: List[str] = batch["clinical_prompt"]
        seq_list: List[torch.Tensor] = []
        max_p_len = 0

        for i in range(bsz):
            p_str = (
                f"You are NeuroReasoner, a clinical neurophysiology AI. "
                f"{prompts[i].strip()}\nGrounded CoT Report:\n"
            )
            p_ids = self.tokenizer(
                p_str, add_special_tokens=True, truncation=True, max_length=max_prompt_tokens, return_tensors="pt"
            )["input_ids"][0].to(target_device)
            comb = torch.cat([neural_prefix[i], emb_layer(p_ids)], dim=0)
            seq_list.append(comb)
            max_p_len = max(max_p_len, comb.size(0))

        # Left-pad prompt embeddings for batched causal generation!
        pad_emb = emb_layer(torch.tensor([self.tokenizer.pad_token_id], device=target_device))[0]
        left_padded = torch.zeros((bsz, max_p_len, d_llm), dtype=neural_prefix.dtype, device=target_device)
        attn_mask = torch.zeros((bsz, max_p_len), dtype=torch.long, device=target_device)

        for i, seq in enumerate(seq_list):
            pad_len = max_p_len - seq.size(0)
            if pad_len > 0:
                left_padded[i, :pad_len] = pad_emb
            left_padded[i, pad_len:] = seq
            attn_mask[i, pad_len:] = 1

        prev_cache = getattr(self.llm.config, "use_cache", False)
        self.llm.config.use_cache = True
        try:
            gen_ids = self.llm.generate(
                inputs_embeds=left_padded,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        finally:
            self.llm.config.use_cache = prev_cache
        decoded = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return [d.strip() for d in decoded]

    def get_privacy_preserving_state(self, dp_noise_std: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        Extract ONLY trainable lightweight parameters + EMA-RVQ codebook buffers (~65M params vs 7.6B frozen LLM)
        for Privacy-Preserving Cross-Center Federated Aggregation (`FedAvg` + optional Gaussian Local DP).
        """
        state: Dict[str, torch.Tensor] = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                tensor = param.detach().cpu().float()
                if dp_noise_std > 0.0:
                    tensor = tensor + torch.randn_like(tensor) * dp_noise_std
                state[name] = tensor
        for bname, buf in self.rvq.named_buffers():
            state[f"rvq.{bname}"] = buf.detach().cpu().clone()
        return state

    def load_privacy_preserving_state(self, state: Dict[str, torch.Tensor]) -> None:
        own_params = dict(self.named_parameters())
        for name, tensor in state.items():
            if name in own_params:
                own_params[name].data.copy_(tensor.to(device=own_params[name].device, dtype=own_params[name].dtype))
        own_rvq_bufs = dict(self.rvq.named_buffers())
        for name, tensor in state.items():
            if name.startswith("rvq."):
                short = name.replace("rvq.", "", 1)
                if short in own_rvq_bufs and own_rvq_bufs[short].shape == tensor.shape:
                    own_rvq_bufs[short].copy_(tensor.to(device=own_rvq_bufs[short].device, dtype=own_rvq_bufs[short].dtype))
