"""
True Coarse-to-Fine Residual Vector Quantizer (RVQ) with:
  1. Initial L2-Sphere Projection + Un-inflated Euclidean EMA Residual Hierarchy
     (Layer 1 captures coarse unit-scale semantics; Layers 2..M capture decaying fine residuals without norm inflation!)
  2. Dead-Code Revival matched to true residual scale (prevents codebook collapse across all M codebooks)
  3. Proper L2 Squared Error Scaling (summed over D, averaged over B*L) so commitment gradients match InfoNCE/MAE scale
  4. Quantizer Dropout (SoundStream / EnCodec style) forcing early codebooks to pack maximal semantic information
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EuclideanEMAQuantizerLayer(nn.Module):
    """Single-stage Euclidean EMA codebook quantizer operating on true (un-inflated) residual magnitudes."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_weight: float = 0.25,
        decay: float = 0.98,
        eps: float = 1e-5,
        dead_code_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.eps = eps
        self.dead_code_threshold = dead_code_threshold

        init_embed = torch.randn(num_embeddings, embedding_dim) * (1.0 / (embedding_dim ** 0.5))
        self.register_buffer("codebook", init_embed)
        self.register_buffer("ema_cluster_size", torch.ones(num_embeddings))
        self.register_buffer("ema_embed_sum", init_embed.clone())
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def _init_from_batch(self, flat_r: torch.Tensor) -> None:
        n = flat_r.size(0)
        if n >= self.num_embeddings:
            perm = torch.randperm(n, device=flat_r.device)[: self.num_embeddings]
            chosen = flat_r[perm].clone()
        else:
            reps = (self.num_embeddings + n - 1) // n
            std = flat_r.std().clamp(min=1e-4)
            chosen = flat_r.repeat(reps, 1)[: self.num_embeddings]
            chosen = chosen + 0.05 * std * torch.randn_like(chosen)
        self.codebook.copy_(chosen)
        self.ema_embed_sum.copy_(chosen)
        self.ema_cluster_size.fill_(1.0)
        self.initialized.fill_(True)

    @torch.no_grad()
    def _ema_update_and_revive(self, flat_r: torch.Tensor, encodings: torch.Tensor) -> None:
        batch_cluster_size = encodings.sum(dim=0)  # [K]
        batch_embed_sum = encodings.t() @ flat_r   # [K, D]

        self.ema_cluster_size.mul_(self.decay).add_(batch_cluster_size, alpha=1.0 - self.decay)
        self.ema_embed_sum.mul_(self.decay).add_(batch_embed_sum, alpha=1.0 - self.decay)

        dead_mask = self.ema_cluster_size < self.dead_code_threshold
        n_dead = int(dead_mask.sum().item())
        if n_dead > 0 and flat_r.size(0) > 0:
            rand_idx = torch.randint(0, flat_r.size(0), (n_dead,), device=flat_r.device)
            noise_scale = 0.02 * flat_r.std().clamp(min=1e-5)
            replacements = flat_r[rand_idx] + noise_scale * torch.randn_like(flat_r[rand_idx])
            self.ema_embed_sum[dead_mask] = replacements
            self.ema_cluster_size[dead_mask] = 1.0

        n_total = self.ema_cluster_size.sum()
        smoothed_size = (
            (self.ema_cluster_size + self.eps)
            / (n_total + self.num_embeddings * self.eps)
            * n_total
        )
        updated_codes = self.ema_embed_sum / smoothed_size.unsqueeze(-1).clamp(min=self.eps)
        self.codebook.copy_(updated_codes)

    def forward(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Args:
            r: Residual tensor [B, L, D] at its natural residual magnitude.
        Returns:
            z_q_st: Straight-through quantized residual [B, L, D]
            indices: Code indices [B, L]
            loss: Commitment loss summed over D, averaged over B*L (scale ~0.05-0.50)
            perplexity: Active codebook utilization perplexity
        """
        b, l, d = r.shape
        r_fp32 = r.float()
        flat_r = r_fp32.reshape(-1, d)  # [B*L, D]

        if self.training and not bool(self.initialized.item()):
            self._init_from_batch(flat_r.detach())

        cb = self.codebook.float()
        # Squared Euclidean distance ||r - e||_2^2
        distances = (
            torch.sum(flat_r ** 2, dim=1, keepdim=True)
            + torch.sum(cb ** 2, dim=1)
            - 2.0 * torch.matmul(flat_r, cb.t())
        )
        indices = torch.argmin(distances, dim=1)  # [B*L]
        encodings = F.one_hot(indices, self.num_embeddings).float()

        if self.training:
            self._ema_update_and_revive(flat_r.detach(), encodings)
            cb = self.codebook.float()

        z_q = F.embedding(indices, cb).view(b, l, d)

        # Sum over D (since ||r_0||_2 = 1.0), mean over B*L -> well-scaled O(0.1) commitment loss!
        sq_err = torch.sum((r_fp32 - z_q.detach()) ** 2, dim=-1).mean()
        commitment_loss = self.commitment_weight * sq_err

        z_q_st = (r_fp32 + (z_q - r_fp32).detach()).to(r.dtype)

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()

        return z_q_st, indices.view(b, l), commitment_loss.to(r.dtype), perplexity


class LegacyVectorQuantizerLayer(nn.Module):
    """Legacy vanilla Euclidean VQ layer retained strictly for loading Ablation Row-1 checkpoints."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_weight: float = 0.25) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        b, l, d = z.shape
        flat_z = z.reshape(-1, d)
        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight ** 2, dim=1)
            - 2.0 * torch.matmul(flat_z, self.codebook.weight.t())
        )
        indices = torch.argmin(distances, dim=1)
        z_q = self.codebook(indices).view(b, l, d)
        loss = F.mse_loss(z_q, z.detach()) + self.commitment_weight * F.mse_loss(z, z_q.detach())
        z_q_st = z + (z_q - z).detach()
        encodings = F.one_hot(indices, self.num_embeddings).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()
        return z_q_st, indices.view(b, l), loss, perplexity


class ResidualVQ(nn.Module):
    """
    Multi-stage Coarse-to-Fine EMA Residual Vector Quantizer (RVQ) with Quantizer Dropout.
    """

    def __init__(
        self,
        num_quantizers: int = 4,
        codebook_size: int = 512,
        embedding_dim: int = 1024,
        commitment_weight: float = 0.25,
        quantizer_dropout: float = 0.0,
        ema_decay: float = 0.98,
        legacy_vanilla_vq: bool = False,
    ) -> None:
        super().__init__()
        self.num_quantizers = num_quantizers
        self.quantizer_dropout = quantizer_dropout
        self.legacy_vanilla_vq = legacy_vanilla_vq

        if legacy_vanilla_vq:
            self.layers = nn.ModuleList(
                [
                    LegacyVectorQuantizerLayer(
                        num_embeddings=codebook_size,
                        embedding_dim=embedding_dim,
                        commitment_weight=commitment_weight,
                    )
                    for _ in range(num_quantizers)
                ]
            )
            self.out_norm = nn.Identity()
        else:
            self.layers = nn.ModuleList(
                [
                    EuclideanEMAQuantizerLayer(
                        num_embeddings=codebook_size,
                        embedding_dim=embedding_dim,
                        commitment_weight=commitment_weight,
                        decay=ema_decay,
                    )
                    for _ in range(num_quantizers)
                ]
            )
            self.out_norm = nn.LayerNorm(embedding_dim)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor | float | List[float]]:
        """Quantize [B, L, D] across M coarse-to-fine residual codebooks."""
        if self.legacy_vanilla_vq:
            residual = z
        else:
            # Project initial continuous latent onto unit sphere ONCE; subsequent residuals decay naturally!
            residual = F.normalize(z.float(), dim=-1).to(z.dtype)

        active_q = self.num_quantizers
        if self.training and self.quantizer_dropout > 0.0 and self.num_quantizers > 1:
            if random.random() < self.quantizer_dropout:
                active_q = random.randint(1, self.num_quantizers)

        quantized_out = torch.zeros_like(residual)
        total_loss = torch.tensor(0.0, device=z.device, dtype=z.dtype)
        all_indices = []
        perplexities: List[float] = []

        for q_idx, layer in enumerate(self.layers):
            z_q, idx, layer_loss, ppl = layer(residual)
            # True residual subtraction without re-inflation
            residual = residual - z_q.detach()
            if q_idx < active_q:
                quantized_out = quantized_out + z_q
                total_loss = total_loss + layer_loss
            all_indices.append(idx)
            perplexities.append(ppl)

        denom = max(1, active_q)
        return {
            "quantized": self.out_norm(quantized_out),           # [B, L, D]
            "codes": torch.stack(all_indices, dim=-1),           # [B, L, M]
            "rvq_loss": total_loss / denom,
            "mean_perplexity": sum(perplexities) / len(perplexities),
            "per_layer_perplexity": perplexities,
        }
