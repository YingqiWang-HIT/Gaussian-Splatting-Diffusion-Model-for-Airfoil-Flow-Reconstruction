from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ConditionEncoder(nn.Module):
    """Encode airfoil surface coordinates and incoming-flow conditions."""

    def __init__(self, embed_dim: int = 512, geometry_conv_layers: int = 4, condition_mlp_layers: int = 3, heads: int = 8):
        super().__init__()
        conv = []
        in_ch = 2
        for _ in range(geometry_conv_layers):
            conv += [nn.Conv1d(in_ch, embed_dim, kernel_size=3, padding=1), nn.GELU()]
            in_ch = embed_dim
        self.geometry_encoder = nn.Sequential(*conv)

        mlp = []
        in_dim = 3
        for i in range(condition_mlp_layers - 1):
            mlp += [nn.Linear(in_dim, embed_dim), nn.GELU()]
            in_dim = embed_dim
        mlp += [nn.Linear(in_dim, embed_dim)]
        self.condition_encoder = nn.Sequential(*mlp)

        self.fusion = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, geometry: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # geometry: [B, Ns, 2], condition: [B, 3] = [Ma, Re, alpha_deg]
        geom = geometry.transpose(1, 2)
        h_geom = self.geometry_encoder(geom).mean(dim=-1)
        cond_input = torch.stack([
            torch.log(condition[:, 0].clamp_min(1.0e-6)),
            torch.log(condition[:, 1].clamp_min(1.0e-6)),
            condition[:, 2] / 10.0,
        ], dim=-1)
        h_op = self.condition_encoder(cond_input)
        tokens = torch.stack([h_geom, h_op], dim=1)
        fused, _ = self.fusion(tokens, tokens, tokens)
        return self.norm(fused.mean(dim=1))


class AdaLayerNorm(nn.Module):
    def __init__(self, embed_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.to_scale_shift = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * embed_dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale[:, None, :]) + shift[:, None, :]


class AdaTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, heads: int, ffn_dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.adaln1 = AdaLayerNorm(embed_dim, cond_dim)
        self.attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.adaln2 = AdaLayerNorm(embed_dim, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.adaln1(x, cond)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        x = x + self.ffn(self.adaln2(x, cond))
        return x


class DenoiserTransformer(nn.Module):
    """Set-to-set Transformer noise predictor in Gaussian parameter space."""

    def __init__(self,
                 primitive_dim: int = 9,
                 embed_dim: int = 512,
                 layers: int = 8,
                 heads: int = 8,
                 ffn_dim: int = 2048,
                 dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_proj = nn.Linear(primitive_dim, embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.SiLU(), nn.Linear(embed_dim * 4, embed_dim)
        )
        self.blocks = nn.ModuleList([
            AdaTransformerBlock(embed_dim, heads, ffn_dim, cond_dim=embed_dim, dropout=dropout)
            for _ in range(layers)
        ])
        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, primitive_dim)

    def forward(self, theta_t: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(sinusoidal_timestep_embedding(timesteps, self.embed_dim))
        cond = cond + t_emb
        x = self.in_proj(theta_t)
        for block in self.blocks:
            x = block(x, cond)
        return self.out_proj(self.out_norm(x))


class PGausDiff(nn.Module):
    """CGF-CDG module: conditional Gaussian primitive diffusion generator."""

    def __init__(self,
                 primitive_dim: int = 9,
                 embed_dim: int = 512,
                 geometry_conv_layers: int = 4,
                 condition_mlp_layers: int = 3,
                 transformer_layers: int = 8,
                 attention_heads: int = 8,
                 ffn_dim: int = 2048,
                 dropout: float = 0.0):
        super().__init__()
        self.condition_encoder = ConditionEncoder(
            embed_dim=embed_dim,
            geometry_conv_layers=geometry_conv_layers,
            condition_mlp_layers=condition_mlp_layers,
            heads=attention_heads,
        )
        self.denoiser = DenoiserTransformer(
            primitive_dim=primitive_dim,
            embed_dim=embed_dim,
            layers=transformer_layers,
            heads=attention_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(self, theta_t: torch.Tensor, timesteps: torch.Tensor, geometry: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        cond = self.condition_encoder(geometry, condition)
        return self.denoiser(theta_t, timesteps, cond)
