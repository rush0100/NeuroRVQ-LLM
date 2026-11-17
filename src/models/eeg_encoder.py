"""
Dual-Stream Connectome + 8-Metadata EEG Encoder (`DualStreamConnectomeEncoder`):
  - Zero-Length-Leakage Normalized Progress Fourier Positional Encoding (`rel_pos` & `lex_pos` in `[0, 1]`)
  - 3D Cortical ROI Spatial Embedding (initialized from official `chanlocs105.mat` 3D ROI centroids)
  - Stream 1 (`NodeStream`): 4-Stage Regional Power (`FFD`, `GD`, `TRT`, `TRT-FFD`) + L-R Asymmetry + Eye-Gaze Meta
  - Stream 2 (`EdgeManifoldStream`): Word Co-fluctuation Edges (`528-d`) + Global Sentence `wPLI / iCoh / SPD-Tangent` Tokens (`4 x D`)
  - Cross-Stream Gated Node-Edge Fusion + Task-Context & Cognitive-Load Conditioning
  - Salience-Weighted Length-Invariant `LatentQueryResampler` (`[B, L, D] -> [B, K, D]`)
  - Configurable `stream_mode` (`"dual"`, `"node_only"`, `"edge_only"`, `"legacy_840"`) for clean Q1 paper ablations!
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.neuro_connectome import (
    COG_LOAD_DIM,
    NODE_FEAT_DIM,
    NUM_BANDS,
    NUM_ROIS,
    SENT_CONNECTOME_DIM,
    WORD_EDGE_DIM,
    get_roi_3d_centroids,
)


class NormalizedProgressFourierPE(nn.Module):
    """
    Continuous Fourier Positional Encoding over normalized scanpath progress `u in [0.0, 1.0]`
    and lexical position `v in [0.0, 1.0]`.
    Completely eliminates sentence-length / fixation-count leakage (`u` always spans `[0, 1]` whether 4 words or 45 words!).
    """

    def __init__(self, embed_dim: int, num_freqs: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        freqs = (2.0 ** torch.linspace(0.0, 5.0, steps=num_freqs)) * math.pi
        self.register_buffer("freqs", freqs)  # [F]
        in_dim = num_freqs * 4 + 1            # sin/cos(scanpath) + sin/cos(lexical) + salience
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rel_pos: Optional[torch.Tensor] = None,
        lex_pos: Optional[torch.Tensor] = None,
        word_salience: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, l, _ = x.shape
        device = x.device
        if rel_pos is None:
            rel_pos = torch.linspace(0.0, 1.0, steps=l, device=device).unsqueeze(0).expand(b, -1)
        if lex_pos is None:
            lex_pos = rel_pos
        if word_salience is None:
            word_salience = torch.ones((b, l), device=device, dtype=x.dtype)

        u = rel_pos.float().unsqueeze(-1) * self.freqs.view(1, 1, -1)  # [B, L, F]
        v = lex_pos.float().unsqueeze(-1) * self.freqs.view(1, 1, -1)  # [B, L, F]
        sal = (word_salience.float().clamp(0.0, 3.0) / 3.0).unsqueeze(-1)

        pos_feat = torch.cat([torch.sin(u), torch.cos(u), torch.sin(v), torch.cos(v), sal], dim=-1).to(x.dtype)
        return self.dropout(x + self.proj(pos_feat))


class Cortical3DROINodeStem(nn.Module):
    """
    Structured Node Stem incorporating 3D Anatomical ROI coordinates (`[12, 3]` from `chanlocs105.mat`),
    4-stage cognitive reading windows (`FFD`, `GD`, `TRT`, `TRT-FFD`), L-R hemispheric asymmetry, and gaze metadata.
    """

    def __init__(self, node_dim: int = NODE_FEAT_DIM, embed_dim: int = 512, dropout: float = 0.15) -> None:
        super().__init__()
        self.roi_core_dim = 4 * NUM_BANDS * NUM_ROIS  # 384
        self.extra_dim = node_dim - self.roi_core_dim  # 104 (96 L-R asym + 8 word gaze meta)

        centroids = torch.from_numpy(get_roi_3d_centroids())  # [12, 3]
        self.register_buffer("roi_3d_coords", centroids)

        roi_hidden = max(32, embed_dim // 8)
        self.coord_3d_mlp = nn.Sequential(
            nn.Linear(3, roi_hidden),
            nn.GELU(),
            nn.Linear(roi_hidden, roi_hidden),
        )
        self.roi_band_proj = nn.Linear(4 * NUM_BANDS, roi_hidden)
        self.stage_band_gates = nn.Parameter(torch.ones(1, 1, 4 * NUM_BANDS, 1))

        fused_in = NUM_ROIS * roi_hidden + self.extra_dim
        self.out_proj = nn.Sequential(
            nn.Linear(fused_in, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        b, l, _ = node_features.shape
        core = node_features[:, :, : self.roi_core_dim]   # [B, L, 384]
        extra = node_features[:, :, self.roi_core_dim :]  # [B, L, 104]

        # Reshape core -> [B, L, 4*8, 12] -> transpose to [B, L, 12, 32]
        core_4b_r = core.view(b, l, 4 * NUM_BANDS, NUM_ROIS) * self.stage_band_gates
        core_r_4b = core_4b_r.permute(0, 1, 3, 2).contiguous()  # [B, L, 12, 32]

        spatial_pe = self.coord_3d_mlp(self.roi_3d_coords.to(node_features.dtype)).view(1, 1, NUM_ROIS, -1)
        roi_tokens = F.gelu(self.roi_band_proj(core_r_4b) + spatial_pe)  # [B, L, 12, roi_hidden]

        cat = torch.cat([roi_tokens.reshape(b, l, -1), extra], dim=-1)
        return self.out_proj(cat)


class EdgeAndManifoldStem(nn.Module):
    """
    Projects word-level instantaneous coupled edges (`[B, L, 528]`) AND decomposes the sentence-level
    connectome (`[B, 2064]` = `wPLI` + `iCoh` + `Log-Euclidean SPD Tangent` + `48-pair Asymmetry`)
    into 4 dedicated Global Connectome Tokens `[B, 4, D]`.
    """

    def __init__(
        self,
        word_edge_dim: int = WORD_EDGE_DIM,
        sent_conn_dim: int = SENT_CONNECTOME_DIM,
        embed_dim: int = 512,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.word_edge_proj = nn.Sequential(
            nn.Linear(word_edge_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Split 2064 sentence connectome into 4 specialized neuro-physical tokens:
        #   Token 0: wPLI (528), Token 1: iCoh (528), Token 2: Riemannian SPD Tangent (624), Token 3: L-R Diff (384)
        self.wpli_tok = nn.Sequential(nn.LayerNorm(528), nn.Linear(528, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))
        self.icoh_tok = nn.Sequential(nn.LayerNorm(528), nn.Linear(528, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))
        self.spd_tok = nn.Sequential(nn.LayerNorm(624), nn.Linear(624, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))
        self.asym_tok = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))
        self.sent_conn_dim = sent_conn_dim

    def forward_word_edges(self, word_coupled_edges: torch.Tensor) -> torch.Tensor:
        return self.word_edge_proj(word_coupled_edges)

    def forward_global_connectome_tokens(self, sent_connectome: torch.Tensor) -> torch.Tensor:
        wpli = sent_connectome[:, :528]
        icoh = sent_connectome[:, 528:1056]
        spd = sent_connectome[:, 1056:1680]
        asym = sent_connectome[:, 1680:2064]
        t0 = self.wpli_tok(wpli)
        t1 = self.icoh_tok(icoh)
        t2 = self.spd_tok(spd)
        t3 = self.asym_tok(asym)
        return torch.stack([t0, t1, t2, t3], dim=1)  # [B, 4, D]


class DualStreamNeuroAugmentor(nn.Module):
    """Training-only physiological augmentation across Node, Edge, and Connectome streams."""

    def __init__(
        self,
        roi_dropout: float = 0.0,
        band_jitter: float = 0.0,
        sensor_noise: float = 0.0,
        edge_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.roi_dropout = roi_dropout
        self.band_jitter = band_jitter
        self.sensor_noise = sensor_noise
        self.edge_dropout = edge_dropout

    def forward(
        self,
        node_features: torch.Tensor,
        word_edges: torch.Tensor,
        sent_connectome: torch.Tensor,
        eeg_attn_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training:
            return node_features, word_edges, sent_connectome

        b, l, _ = node_features.shape
        nf = node_features.clone()
        we = word_edges.clone()
        sc = sent_connectome.clone()

        core_len = 4 * NUM_BANDS * NUM_ROIS  # 384
        core = nf[:, :, :core_len].view(b, l, 4, NUM_BANDS, NUM_ROIS)

        if self.band_jitter > 0.0:
            gain = (1.0 + torch.randn(b, 1, 1, NUM_BANDS, 1, device=nf.device, dtype=nf.dtype) * self.band_jitter).clamp(0.6, 1.4)
            core = core * gain

        if self.roi_dropout > 0.0:
            keep = 1.0 - self.roi_dropout
            rmask = (torch.rand(b, 1, 1, 1, NUM_ROIS, device=nf.device) < keep).to(nf.dtype) / keep
            core = core * rmask

        nf[:, :, :core_len] = core.reshape(b, l, core_len)

        if self.sensor_noise > 0.0:
            nf = nf + torch.randn_like(nf) * self.sensor_noise
            sc = sc + torch.randn_like(sc) * (0.5 * self.sensor_noise)

        if self.edge_dropout > 0.0:
            we = F.dropout(we, p=self.edge_dropout, training=True)
            sc = F.dropout(sc, p=0.5 * self.edge_dropout, training=True)

        mask_f = eeg_attn_mask.unsqueeze(-1).to(nf.dtype)
        return nf * mask_f, we * mask_f, sc


class LatentQueryResampler(nn.Module):
    """
    Salience-aware Cross-Attention Resampler compressing variable-length sequences `[B, L_ext, D]`
    into fixed `K` latent brain tokens `[B, K, D]` with zero padding-length leakage.
    """

    def __init__(self, embed_dim: int = 512, num_queries: int = 32, num_heads: int = 8, dropout: float = 0.15) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.latents = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, encoded_seq: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        b = encoded_seq.size(0)
        queries = self.latents.expand(b, -1, -1)
        key_padding_mask = valid_mask == 0

        attn_out, _ = self.cross_attn(
            query=self.norm_q(queries),
            key=self.norm_kv(encoded_seq),
            value=self.norm_kv(encoded_seq),
            key_padding_mask=key_padding_mask,
        )
        x = queries + attn_out
        x = x + self.ffn(x)
        return x


class LateConnectomeFusionAdapter(nn.Module):
    """
    Late Cross-Attention Connectome Fusion Adapter (`stream_mode == "dual_late_fusion"`):
    Lets the 4-Stage 3D 12-ROI Node Transformer run cleanly without quadratic edge-memorization inside word tokens,
    then enriches the K resampled latent brain tokens via a heavily regularized bottleneck cross-attention over the
    4 Global Connectome Tokens (`wPLI`, `iCoh`, `Log-Euclidean SPD Tangent`, `48-pair L-R Asymmetry`)!
    Zero-initialized residual projection guarantees strictly >= Node-Only generalization at initialization.
    """

    def __init__(self, embed_dim: int = 768, bottleneck_dim: int = 128, num_heads: int = 4, dropout: float = 0.30) -> None:
        super().__init__()
        self.conn_down = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, bottleneck_dim),
            nn.GELU(),
        )
        self.query_down = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, bottleneck_dim),
        )
        self.cross_attn = nn.MultiheadAttention(bottleneck_dim, num_heads, dropout=dropout, batch_first=True)
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, resampled_queries: torch.Tensor, global_conn_tokens: torch.Tensor) -> torch.Tensor:
        q = self.query_down(resampled_queries)     # [B, K, B_dim]
        kv = self.conn_down(global_conn_tokens)    # [B, 4, B_dim]
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        return resampled_queries + self.up_proj(attn_out)


class SpatioFrequencyEEGEncoder(nn.Module):
    """
    Unified Dual-Stream Connectome + 8-Metadata Encoder (retaining class name for seamless Stage 2 compatibility).
    Supports `stream_mode` in `{"dual", "node_only", "edge_only", "legacy_840"}`.
    """

    def __init__(
        self,
        num_channels: int = 105,
        num_bands: int = 8,
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1536,
        dropout: float = 0.15,
        use_resampler: bool = True,
        num_latent_queries: int = 32,
        max_len: int = 128,
        stream_mode: str = "dual",
        node_feat_dim: int = NODE_FEAT_DIM,
        word_edge_dim: int = WORD_EDGE_DIM,
        sent_conn_dim: int = SENT_CONNECTOME_DIM,
        cog_load_dim: int = COG_LOAD_DIM,
        use_task_conditioning: bool = True,
        use_band_topography_norm: bool = False,
        channel_dropout: float = 0.0,
        band_jitter: float = 0.0,
        sensor_noise: float = 0.0,
        token_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.stream_mode = stream_mode
        self.use_resampler = use_resampler
        self.use_task_conditioning = use_task_conditioning

        self.augmentor = DualStreamNeuroAugmentor(
            roi_dropout=channel_dropout,
            band_jitter=band_jitter,
            sensor_noise=sensor_noise,
            edge_dropout=token_dropout,
        )

        # Legacy 840-d stem (only active when `stream_mode == "legacy_840"`)
        self.legacy_stem = nn.Sequential(
            nn.Linear(num_channels * num_bands, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.node_stem = Cortical3DROINodeStem(node_dim=node_feat_dim, embed_dim=embed_dim, dropout=dropout)
        self.edge_stem = EdgeAndManifoldStem(
            word_edge_dim=word_edge_dim, sent_conn_dim=sent_conn_dim, embed_dim=embed_dim, dropout=dropout
        )

        # Zero-initialized Residual Edge Adapter (`h = LayerNorm(h_node + alpha_edge * gate * h_edge)`)
        # Starts from clean Node representation (`alpha_edge = 0`) and injects coupled edge/manifold signals only where beneficial!
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.edge_residual_scale = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.global_conn_scale = nn.Parameter(torch.ones(1, 1, embed_dim) * 0.25)
        self.fusion_norm = nn.LayerNorm(embed_dim)

        # Task Embedding (`SR=0, NR=1, TSR=2`) + Oculomotor/Pupil Cognitive Load Token
        self.task_emb = nn.Embedding(4, embed_dim)
        self.cog_load_proj = nn.Sequential(
            nn.LayerNorm(cog_load_dim),
            nn.Linear(cog_load_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        self.progress_pe = NormalizedProgressFourierPE(embed_dim=embed_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.resampler = (
            LatentQueryResampler(embed_dim, num_queries=num_latent_queries, num_heads=num_heads, dropout=dropout)
            if use_resampler
            else None
        )
        self.late_conn_adapter = LateConnectomeFusionAdapter(embed_dim=embed_dim, bottleneck_dim=128, num_heads=4, dropout=0.30)

    def forward_sequence(
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
        apply_augment: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns `(encoded_full_seq, full_valid_mask, prefix_global_tokens_count)` so MAE reconstruction
        slices back the exact `L` word+sentence tokens cleanly while the Resampler attends over both
        Global Connectome Tokens and Word Tokens!
        """
        bsz = eeg_features.size(0)
        device = eeg_features.device

        if self.stream_mode == "legacy_840" or node_features is None or float(node_features.abs().sum().item()) == 0.0:
            x_tok = self.legacy_stem(eeg_features)
            x_tok = self.progress_pe(x_tok, rel_pos=rel_pos, lex_pos=lex_pos, word_salience=word_salience)
            out_seq = self.transformer(x_tok, src_key_padding_mask=(eeg_attn_mask == 0))
            return out_seq, eeg_attn_mask, 0

        assert word_coupled_edges is not None and sent_connectome is not None
        nf, we, sc = (
            self.augmentor(node_features, word_coupled_edges, sent_connectome, eeg_attn_mask)
            if apply_augment
            else (node_features, word_coupled_edges, sent_connectome)
        )

        if self.stream_mode in ("node_only", "dual_late_fusion"):
            x_tok = self.node_stem(nf)
            global_tokens_list = []
        elif self.stream_mode == "edge_only":
            x_tok = self.edge_stem.forward_word_edges(we)
            global_tokens_list = [self.edge_stem.forward_global_connectome_tokens(sc)]
        else:
            # Residual Dual-Stream Fusion: preserves 100% of Node signal at init while adding gated Edge & Connectome residuals!
            h_node = self.node_stem(nf)
            h_edge = self.edge_stem.forward_word_edges(we)
            gate = self.fusion_gate(torch.cat([h_node, h_edge], dim=-1))
            x_tok = self.fusion_norm(h_node + torch.tanh(self.edge_residual_scale) * (gate * h_edge))
            global_tokens_list = [self.edge_stem.forward_global_connectome_tokens(sc) * self.global_conn_scale]

        # Apply zero-length-leakage relative scanpath positional encoding + salience modulation
        x_tok = self.progress_pe(x_tok, rel_pos=rel_pos, lex_pos=lex_pos, word_salience=word_salience)

        if self.use_task_conditioning:
            t_ids = task_id.clamp(0, 3) if task_id is not None else torch.ones((bsz,), dtype=torch.long, device=device)
            t_vec = self.task_emb(t_ids).unsqueeze(1)  # [B, 1, D]
            if cog_load is not None:
                c_vec = self.cog_load_proj(cog_load.to(x_tok.dtype)).unsqueeze(1)  # [B, 1, D]
                global_tokens_list.append(t_vec + c_vec)
            else:
                global_tokens_list.append(t_vec)

        if global_tokens_list:
            global_prefix = torch.cat(global_tokens_list, dim=1)  # [B, P_glob, D]
            p_glob = global_prefix.size(1)
            full_seq = torch.cat([global_prefix, x_tok], dim=1)
            glob_mask = torch.ones((bsz, p_glob), dtype=eeg_attn_mask.dtype, device=device)
            full_mask = torch.cat([glob_mask, eeg_attn_mask], dim=1)
        else:
            p_glob = 0
            full_seq = x_tok
            full_mask = eeg_attn_mask

        encoded = self.transformer(full_seq, src_key_padding_mask=(full_mask == 0))
        return encoded, full_mask, p_glob

    def forward(
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
        apply_augment: bool = True,
    ) -> torch.Tensor:
        encoded, full_mask, _ = self.forward_sequence(
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
            apply_augment=apply_augment,
        )
        if self.resampler is not None:
            out_q = self.resampler(encoded, full_mask)
        else:
            out_q = encoded

        if self.stream_mode == "dual_late_fusion" and sent_connectome is not None:
            sc_in = F.dropout(sent_connectome, p=0.25, training=self.training)
            glob_conn = self.edge_stem.forward_global_connectome_tokens(sc_in)  # [B, 4, D]
            out_q = self.late_conn_adapter(out_q, glob_conn)
        return out_q
