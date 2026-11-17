"""
Stage 1 Pretraining Architecture: CET-RVQ v3 (Dual-Stream Neuro-Connectome + Full 8-Metadata Edition)
  - Dual Masked Reconstruction (`L_MAE` over Multi-Stage 12-ROI Nodes + Coupled Edges)
  - Coarse-to-Fine Un-inflated EMA Residual VQ (`ResidualVQ`) with Quantizer Dropout
  - Cognitive-Load Weighted Multi-Positive Cross-Subject InfoNCE (`L_Con`)
  - Cross-Subject Brain-to-Brain Neural Invariance (`L_B2B`)
  - Auxiliary Sentiment / Task Semantic Head (`L_Aux` organizing emotional & cognitive latent geometry)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.neuro_connectome import COG_LOAD_DIM, NODE_FEAT_DIM, SENT_CONNECTOME_DIM, WORD_EDGE_DIM
from src.models.eeg_encoder import SpatioFrequencyEEGEncoder
from src.models.rvq import ResidualVQ


class AttentiveBrainPooler(nn.Module):
    """Learnable attention pooling over K resampled neural tokens `[B, K, D] -> [B, D]`."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.attn_weights = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, max(16, embed_dim // 4)),
            nn.Tanh(),
            nn.Linear(max(16, embed_dim // 4), 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        scores = self.attn_weights(tokens)
        weights = F.softmax(scores, dim=1)
        return torch.sum(tokens * weights, dim=1)


def multi_positive_infonce(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    scale: torch.Tensor,
    sentence_ids: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Symmetric Multi-Positive InfoNCE with optional per-trial Cognitive-Load weighting
    (upweights high-engagement / high-pupil-arousal reading trials and downweights high-omission trials).
    """
    bsz = emb_a.size(0)
    logits_ab = scale * (emb_a @ emb_b.t())
    logits_ba = logits_ab.t()

    if sentence_ids is None:
        pos_mask = torch.eye(bsz, device=emb_a.device, dtype=logits_ab.dtype)
    else:
        pos_mask = (sentence_ids.unsqueeze(1) == sentence_ids.unsqueeze(0)).to(logits_ab.dtype)

    target_dist = pos_mask / pos_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    if label_smoothing > 0.0 and bsz > 1:
        target_dist = (1.0 - label_smoothing) * target_dist + (label_smoothing / bsz)

    per_row_ab = -(target_dist * F.log_softmax(logits_ab, dim=-1)).sum(dim=-1)
    per_row_ba = -(target_dist * F.log_softmax(logits_ba, dim=-1)).sum(dim=-1)

    if sample_weights is not None:
        w = sample_weights.to(per_row_ab.dtype)
        w = w / w.mean().clamp(min=1e-5)
        return 0.5 * ((per_row_ab * w).mean() + (per_row_ba * w).mean())

    return 0.5 * (per_row_ab.mean() + per_row_ba.mean())


def cross_subject_brain_invariance_loss(
    eeg_emb: torch.Tensor,
    sentence_ids: Optional[torch.Tensor],
    temperature: float = 0.12,
) -> torch.Tensor:
    """Pulls quantized brain embeddings of DIFFERENT subjects reading the SAME sentence together."""
    if sentence_ids is None or eeg_emb.size(0) < 2:
        return torch.tensor(0.0, device=eeg_emb.device, dtype=eeg_emb.dtype)

    bsz = eeg_emb.size(0)
    eye = torch.eye(bsz, device=eeg_emb.device, dtype=torch.bool)
    same_sent = (sentence_ids.unsqueeze(1) == sentence_ids.unsqueeze(0)) & (~eye)

    has_pair = same_sent.any(dim=1)
    if not bool(has_pair.any().item()):
        return torch.tensor(0.0, device=eeg_emb.device, dtype=eeg_emb.dtype)

    sim = (eeg_emb @ eeg_emb.t()) / temperature
    sim = sim.masked_fill(eye, -1e9)
    log_prob = F.log_softmax(sim, dim=-1)

    pos_float = same_sent[has_pair].to(eeg_emb.dtype)
    log_prob_valid = log_prob[has_pair]
    mean_log_pos = (pos_float * log_prob_valid).sum(dim=-1) / pos_float.sum(dim=-1).clamp(min=1.0)
    return -mean_log_pos.mean()


class CETRVQPretrainer(nn.Module):
    """Dual-Stream Connectome + Full 8-Metadata CET-RVQ Pretrainer."""

    def __init__(
        self,
        eeg_feature_dim: int = 840,
        num_channels: int = 105,
        num_bands: int = 8,
        embed_dim: int = 512,
        llm_hidden_dim: int = 3584,
        encoder_layers: int = 4,
        encoder_heads: int = 8,
        ffn_dim: int = 1536,
        dropout: float = 0.15,
        use_resampler: bool = True,
        num_latent_queries: int = 32,
        rvq_num_quantizers: int = 4,
        rvq_codebook_size: int = 512,
        rvq_commitment_weight: float = 0.25,
        rvq_quantizer_dropout: float = 0.2,
        rvq_ema_decay: float = 0.98,
        max_eeg_len: int = 128,
        stream_mode: str = "dual",
        node_feat_dim: int = NODE_FEAT_DIM,
        word_edge_dim: int = WORD_EDGE_DIM,
        sent_conn_dim: int = SENT_CONNECTOME_DIM,
        cog_load_dim: int = COG_LOAD_DIM,
        use_task_conditioning: bool = True,
        use_band_topography_norm: bool = False,
        channel_dropout: float = 0.15,
        band_jitter: float = 0.10,
        sensor_noise: float = 0.05,
        token_dropout: float = 0.10,
        use_attentive_pooler: bool = True,
        proj_dropout: float = 0.15,
        init_temperature: float = 0.07,
        learnable_temperature: bool = True,
        max_logit_scale: float = 16.0,
        legacy_vanilla_vq: bool = False,
    ) -> None:
        super().__init__()
        self.eeg_feature_dim = eeg_feature_dim
        self.node_feat_dim = node_feat_dim
        self.word_edge_dim = word_edge_dim
        self.sent_conn_dim = sent_conn_dim
        self.embed_dim = embed_dim
        self.stream_mode = stream_mode
        self.max_logit_scale = max_logit_scale

        self.eeg_encoder = SpatioFrequencyEEGEncoder(
            num_channels=num_channels,
            num_bands=num_bands,
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_resampler=use_resampler,
            num_latent_queries=num_latent_queries,
            max_len=max_eeg_len,
            stream_mode=stream_mode,
            node_feat_dim=node_feat_dim,
            word_edge_dim=word_edge_dim,
            sent_conn_dim=sent_conn_dim,
            cog_load_dim=cog_load_dim,
            use_task_conditioning=use_task_conditioning,
            use_band_topography_norm=use_band_topography_norm,
            channel_dropout=channel_dropout,
            band_jitter=band_jitter,
            sensor_noise=sensor_noise,
            token_dropout=token_dropout,
        )

        self.rvq = ResidualVQ(
            num_quantizers=rvq_num_quantizers,
            codebook_size=rvq_codebook_size,
            embedding_dim=embed_dim,
            commitment_weight=rvq_commitment_weight,
            quantizer_dropout=rvq_quantizer_dropout,
            ema_decay=rvq_ema_decay,
            legacy_vanilla_vq=legacy_vanilla_vq,
        )

        self.pooler = AttentiveBrainPooler(embed_dim) if use_attentive_pooler else None

        # Mask vectors for Dual-Stream MAE Reconstruction
        self.mask_legacy_vec = nn.Parameter(torch.zeros(1, 1, eeg_feature_dim))
        self.mask_node_vec = nn.Parameter(torch.zeros(1, 1, node_feat_dim))
        self.mask_edge_vec = nn.Parameter(torch.zeros(1, 1, word_edge_dim))
        nn.init.normal_(self.mask_legacy_vec, std=0.02)
        nn.init.normal_(self.mask_node_vec, std=0.02)
        nn.init.normal_(self.mask_edge_vec, std=0.02)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=encoder_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.eeg_mae_decoder = nn.TransformerEncoder(decoder_layer, num_layers=1)
        self.legacy_recon_head = nn.Linear(embed_dim, eeg_feature_dim)
        self.node_recon_head = nn.Linear(embed_dim, node_feat_dim)
        self.edge_recon_head = nn.Linear(embed_dim, word_edge_dim)

        # Cross-modal contrastive projection heads (Brain <-> Frozen LLM Text Space)
        self.eeg_contrast_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(proj_dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.text_contrast_proj = nn.Sequential(
            nn.LayerNorm(llm_hidden_dim),
            nn.Dropout(proj_dropout),
            nn.Linear(llm_hidden_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Auxiliary Affective/Semantic Sentiment Head (`{0: Neg, 1: Neu, 2: Pos}`)
        self.sentiment_aux_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(proj_dropout),
            nn.Linear(embed_dim, 3),
        )

        init_log_scale = float(np.log(1.0 / max(1e-4, init_temperature)))
        if learnable_temperature:
            self.logit_scale = nn.Parameter(torch.ones([]) * init_log_scale)
        else:
            self.register_buffer("logit_scale", torch.ones([]) * init_log_scale)

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pooler is not None:
            return self.pooler(tokens)
        return tokens.mean(dim=1)

    def get_effective_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(min=1.0, max=self.max_logit_scale)

    def build_token_mask(
        self,
        eeg_attn_mask: torch.Tensor,
        mask_ratio: float = 0.75,
        compulsory_last_token: bool = True,
    ) -> torch.Tensor:
        b, l = eeg_attn_mask.shape
        device = eeg_attn_mask.device
        bool_mask = torch.zeros((b, l), dtype=torch.bool, device=device)

        for i in range(b):
            active_idx = torch.nonzero(eeg_attn_mask[i] > 0, as_tuple=False).squeeze(-1)
            n_active = active_idx.numel()
            if n_active == 0:
                continue
            if compulsory_last_token and n_active >= 2:
                last_pos = active_idx[-1]
                bool_mask[i, last_pos] = True
                word_positions = active_idx[:-1]
                n_mask_words = max(1, int(round(word_positions.numel() * mask_ratio)))
                perm = torch.randperm(word_positions.numel(), device=device)[:n_mask_words]
                bool_mask[i, word_positions[perm]] = True
            else:
                bool_mask[i, active_idx[-1]] = True
        return bool_mask

    @torch.no_grad()
    def encode_quantized_tokens(
        self,
        eeg_features: torch.Tensor,
        eeg_attn_mask: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        word_coupled_edges: Optional[torch.Tensor] = None,
        sent_connectome: Optional[torch.Tensor] = None,
        rel_pos: Optional[torch.Tensor] = None,
        lex_pos: Optional[torch.Tensor] = None,
        word_salience: Optional[torch.Tensor] = None,
        cog_load: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        resampled = self.eeg_encoder(
            eeg_features=eeg_features,
            eeg_attn_mask=eeg_attn_mask,
            node_features=node_features,
            word_coupled_edges=word_coupled_edges,
            sent_connectome=sent_connectome,
            rel_pos=rel_pos,
            lex_pos=lex_pos,
            word_salience=word_salience,
            cog_load=cog_load,
            task_id=task_id,
            apply_augment=False,
        )
        rvq_out = self.rvq(resampled)
        return rvq_out["quantized"], rvq_out["codes"]  # type: ignore[return-value]

    @torch.no_grad()
    def project_text_bank(self, frozen_text_bank: torch.Tensor) -> torch.Tensor:
        param_dtype = next(self.text_contrast_proj.parameters()).dtype
        return F.normalize(self.text_contrast_proj(frozen_text_bank.to(param_dtype)).float(), dim=-1)

    @torch.no_grad()
    def extract_brain_retrieval_embeddings(
        self,
        eeg_features: torch.Tensor,
        eeg_attn_mask: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        word_coupled_edges: Optional[torch.Tensor] = None,
        sent_connectome: Optional[torch.Tensor] = None,
        rel_pos: Optional[torch.Tensor] = None,
        lex_pos: Optional[torch.Tensor] = None,
        word_salience: Optional[torch.Tensor] = None,
        cog_load: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor | float | List[float]]:
        resampled = self.eeg_encoder(
            eeg_features=eeg_features,
            eeg_attn_mask=eeg_attn_mask,
            node_features=node_features,
            word_coupled_edges=word_coupled_edges,
            sent_connectome=sent_connectome,
            rel_pos=rel_pos,
            lex_pos=lex_pos,
            word_salience=word_salience,
            cog_load=cog_load,
            task_id=task_id,
            apply_augment=False,
        )
        rvq_out = self.rvq(resampled)
        quantized: torch.Tensor = rvq_out["quantized"]  # type: ignore[assignment]

        pre_pool = self._pool_tokens(F.layer_norm(resampled, (self.embed_dim,)))
        post_pool = self._pool_tokens(quantized)

        pre_emb = F.normalize(self.eeg_contrast_proj(pre_pool).float(), dim=-1)
        post_emb = F.normalize(self.eeg_contrast_proj(post_pool).float(), dim=-1)
        sent_logits = self.sentiment_aux_head(post_pool).float()
        return {
            "pre_rvq_emb": pre_emb,
            "post_rvq_emb": post_emb,
            "sentiment_logits": sent_logits,
            "rvq_loss": rvq_out["rvq_loss"],
            "mean_perplexity": rvq_out["mean_perplexity"],
            "per_layer_perplexity": rvq_out.get("per_layer_perplexity", []),
        }

    def forward(
        self,
        eeg_features: torch.Tensor,
        eeg_attn_mask: torch.Tensor,
        frozen_text_pooled: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        word_coupled_edges: Optional[torch.Tensor] = None,
        sent_connectome: Optional[torch.Tensor] = None,
        rel_pos: Optional[torch.Tensor] = None,
        lex_pos: Optional[torch.Tensor] = None,
        word_salience: Optional[torch.Tensor] = None,
        cog_load: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        sentiment_label: Optional[torch.Tensor] = None,
        sentence_ids: Optional[torch.Tensor] = None,
        mask_ratio: float = 0.75,
        compulsory_last_token: bool = True,
        mae_weight: float = 0.4,
        contrast_weight: float = 1.0,
        rvq_weight: float = 1.0,
        pre_rvq_contrast_ratio: float = 0.3,
        brain_invariance_weight: float = 0.25,
        aux_sentiment_weight: float = 0.20,
        use_cog_load_weighting: bool = True,
        label_smoothing: float = 0.05,
    ) -> Dict[str, torch.Tensor | float | List[float]]:
        bool_mask = self.build_token_mask(
            eeg_attn_mask, mask_ratio=mask_ratio, compulsory_last_token=compulsory_last_token
        )

        use_connectome = (
            self.stream_mode != "legacy_840"
            and node_features is not None
            and word_coupled_edges is not None
            and sent_connectome is not None
            and float(node_features.abs().sum().item()) > 0.0
        )

        # 1. Dual-Stream Masked EEG Autoencoding (`L_MAE`)
        if use_connectome:
            assert node_features is not None and word_coupled_edges is not None and sent_connectome is not None
            masked_nodes = node_features.clone()
            masked_edges = word_coupled_edges.clone()
            masked_nodes[bool_mask] = self.mask_node_vec.to(masked_nodes.dtype)
            masked_edges[bool_mask] = self.mask_edge_vec.to(masked_edges.dtype)

            masked_full, full_mask, p_glob = self.eeg_encoder.forward_sequence(
                eeg_features=eeg_features,
                eeg_attn_mask=eeg_attn_mask,
                node_features=masked_nodes,
                word_coupled_edges=masked_edges,
                sent_connectome=sent_connectome,
                rel_pos=rel_pos,
                lex_pos=lex_pos,
                word_salience=word_salience,
                cog_load=cog_load,
                task_id=task_id,
                apply_augment=False,
            )
            decoded_full = self.eeg_mae_decoder(masked_full, src_key_padding_mask=(full_mask == 0))
            decoded_tok = decoded_full[:, p_glob:, :]

            pred_nodes = self.node_recon_head(decoded_tok)
            pred_edges = self.edge_recon_head(decoded_tok)
            if bool_mask.any():
                loss_n = F.mse_loss(pred_nodes[bool_mask], node_features[bool_mask])
                loss_e = F.mse_loss(pred_edges[bool_mask], word_coupled_edges[bool_mask])
            else:
                loss_n = F.mse_loss(pred_nodes, node_features)
                loss_e = F.mse_loss(pred_edges, word_coupled_edges)

            if self.stream_mode in ("node_only", "dual_late_fusion"):
                mae_loss = loss_n
            elif self.stream_mode == "edge_only":
                mae_loss = loss_e
            else:
                mae_loss = 0.6 * loss_n + 0.4 * loss_e
        else:
            masked_leg = eeg_features.clone()
            masked_leg[bool_mask] = self.mask_legacy_vec.to(masked_leg.dtype)
            masked_seq, _, _ = self.eeg_encoder.forward_sequence(
                eeg_features=masked_leg, eeg_attn_mask=eeg_attn_mask, rel_pos=rel_pos, apply_augment=False
            )
            decoded_seq = self.eeg_mae_decoder(masked_seq, src_key_padding_mask=(eeg_attn_mask == 0))
            pred_leg = self.legacy_recon_head(decoded_seq)
            mae_loss = F.mse_loss(pred_leg[bool_mask], eeg_features[bool_mask]) if bool_mask.any() else F.mse_loss(pred_leg, eeg_features)

        # 2. Clean/Augmented Dual-Stream Resampling + Coarse-to-Fine Residual VQ
        resampled = self.eeg_encoder(
            eeg_features=eeg_features,
            eeg_attn_mask=eeg_attn_mask,
            node_features=node_features,
            word_coupled_edges=word_coupled_edges,
            sent_connectome=sent_connectome,
            rel_pos=rel_pos,
            lex_pos=lex_pos,
            word_salience=word_salience,
            cog_load=cog_load,
            task_id=task_id,
            apply_augment=self.training,
        )
        rvq_out = self.rvq(resampled)
        quantized_tokens: torch.Tensor = rvq_out["quantized"]  # type: ignore[assignment]
        rvq_loss: torch.Tensor = rvq_out["rvq_loss"]            # type: ignore[assignment]

        # 3. Cognitive-Load Weighted Multi-Positive InfoNCE
        sample_weights: Optional[torch.Tensor] = None
        if use_cog_load_weighting and cog_load is not None and self.training:
            # cog_load[:, 0] = pupil_z/4, cog_load[:, 1] = omission_rate, cog_load[:, 2] = regression_rate
            arousal_bonus = 0.20 * cog_load[:, 0].clamp(-1.0, 1.5) + 0.15 * cog_load[:, 2].clamp(0.0, 1.0)
            skip_penalty = 0.25 * cog_load[:, 1].clamp(0.0, 0.8)
            sample_weights = (1.0 + arousal_bonus - skip_penalty).clamp(0.65, 1.45)

        post_pooled = self._pool_tokens(quantized_tokens)
        post_eeg_emb = F.normalize(self.eeg_contrast_proj(post_pooled), dim=-1)
        txt_emb = F.normalize(self.text_contrast_proj(frozen_text_pooled.to(post_eeg_emb.dtype)), dim=-1)

        scale = self.get_effective_scale()
        post_contrast_loss = multi_positive_infonce(
            post_eeg_emb,
            txt_emb,
            scale=scale,
            sentence_ids=sentence_ids,
            sample_weights=sample_weights,
            label_smoothing=label_smoothing,
        )

        if pre_rvq_contrast_ratio > 0.0:
            pre_pooled = self._pool_tokens(F.layer_norm(resampled, (self.embed_dim,)))
            pre_eeg_emb = F.normalize(self.eeg_contrast_proj(pre_pooled), dim=-1)
            pre_contrast_loss = multi_positive_infonce(
                pre_eeg_emb,
                txt_emb,
                scale=scale,
                sentence_ids=sentence_ids,
                sample_weights=sample_weights,
                label_smoothing=label_smoothing,
            )
            contrast_loss = (1.0 - pre_rvq_contrast_ratio) * post_contrast_loss + pre_rvq_contrast_ratio * pre_contrast_loss
        else:
            contrast_loss = post_contrast_loss

        # 4. Cross-Subject Brain-to-Brain Invariance Loss
        if brain_invariance_weight > 0.0 and sentence_ids is not None:
            b2b_loss = cross_subject_brain_invariance_loss(post_eeg_emb, sentence_ids=sentence_ids)
        else:
            b2b_loss = torch.tensor(0.0, device=eeg_features.device, dtype=post_eeg_emb.dtype)

        # 5. Auxiliary Affective Sentiment Classification Loss (`task1-SR` movie sentiment labels)
        aux_loss = torch.tensor(0.0, device=eeg_features.device, dtype=post_eeg_emb.dtype)
        if aux_sentiment_weight > 0.0 and sentiment_label is not None and bool((sentiment_label >= 0).any().item()):
            sent_logits = self.sentiment_aux_head(post_pooled)
            aux_loss = F.cross_entropy(sent_logits, sentiment_label, ignore_index=-100)

        total_loss = (
            mae_weight * mae_loss
            + rvq_weight * rvq_loss
            + contrast_weight * contrast_loss
            + brain_invariance_weight * b2b_loss
            + aux_sentiment_weight * aux_loss
        )
        return {
            "loss": total_loss,
            "mae_loss": mae_loss.detach(),
            "rvq_loss": rvq_loss.detach(),
            "contrast_loss": post_contrast_loss.detach(),
            "b2b_loss": b2b_loss.detach(),
            "aux_loss": aux_loss.detach(),
            "scale": float(scale.detach().item()),
            "perplexity": rvq_out["mean_perplexity"],
            "per_layer_perplexity": rvq_out.get("per_layer_perplexity", []),
        }
