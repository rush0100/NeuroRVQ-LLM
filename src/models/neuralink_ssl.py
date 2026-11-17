"""Unlabeled Neuralink waveform SSL with a RUSH-shaped representation interface.

IMPORTANT: The exported 768-d tokens are NOT semantically aligned with RUSH
EEG tokens. This module does not infer text, behavior, or electrode topology.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class NeuralinkMaskedEncoder(nn.Module):
    def __init__(
        self,
        window_samples: int = 16384,
        patch_size: int = 256,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mask_ratio: float = 0.5,
        rush_dim: int = 768,
        rush_queries: int = 24,
    ) -> None:
        super().__init__()
        if window_samples <= 0 or patch_size <= 0 or window_samples % patch_size:
            raise ValueError("window_samples must divide into equal patches")
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError("mask_ratio must be strictly between 0 and 1")
        self.patch_size = patch_size
        self.num_patches = window_samples // patch_size
        self.mask_ratio = mask_ratio
        self.patch_embed = nn.Sequential(nn.Linear(patch_size, embed_dim), nn.LayerNorm(embed_dim))
        self.positions = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        block = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=4 * embed_dim,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=depth, enable_nested_tensor=False)
        self.decoder = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, patch_size))
        self.rush_projection = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, rush_dim), nn.GELU(),
            nn.LayerNorm(rush_dim),
        )
        self.rush_queries = rush_queries

    def _patches(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2 or waveform.size(1) != self.num_patches * self.patch_size:
            raise ValueError("Expected waveform shape [B, window_samples]")
        if not torch.isfinite(waveform).all():
            raise ValueError("Non-finite waveform samples detected")
        return waveform.unfold(-1, self.patch_size, self.patch_size)

    def forward(self, waveform: torch.Tensor, mask_seed: int | None = None) -> dict:
        patches = self._patches(waveform)
        tokens = self.patch_embed(patches) + self.positions
        if mask_seed is None:
            scores = torch.rand(patches.shape[:2], device=waveform.device)
        else:
            rng = torch.Generator(device=waveform.device)
            rng.manual_seed(mask_seed)
            scores = torch.rand(patches.shape[:2], device=waveform.device, generator=rng)
        n_mask = max(1, min(self.num_patches - 1, round(self.mask_ratio * self.num_patches)))
        mask_idx = scores.argsort(dim=1)[:, :n_mask]
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, mask_idx, True)
        masked_tokens = torch.where(mask.unsqueeze(-1), self.mask_token + self.positions, tokens)
        hidden = self.encoder(masked_tokens)
        prediction = self.decoder(hidden)
        # Robust patch-level scaling avoids domination by large-amplitude files.
        target = (patches - patches.mean(dim=-1, keepdim=True)) / patches.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(1e-4)
        error = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
        loss = error[mask].mean()
        return {"loss": loss, "mask": mask, "prediction": prediction}

    @torch.no_grad()
    def encode_rush_tokens(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return [B, K=24, D=768], shape-compatible but NOT aligned with EEG/RUSH."""
        patches = self._patches(waveform)
        hidden = self.encoder(self.patch_embed(patches) + self.positions)
        source = self.rush_projection(hidden)
        # Fixed size K; differentiable interpolation but no positional semantics transfer.
        return F.adaptive_avg_pool1d(source.transpose(1, 2), self.rush_queries).transpose(1, 2)
