from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class GaussianDiffusion:
    """DDPM-style diffusion process in normalized Gaussian parameter space."""

    def __init__(self, steps: int = 2000, beta_start: float = 1e-4, beta_end: float = 2e-2, device: str | torch.device = "cpu"):
        self.steps = steps
        self.device = torch.device(device)
        betas = torch.linspace(beta_start, beta_end, steps, device=self.device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def to(self, device: str | torch.device):
        self.device = torch.device(device)
        for name in ["betas", "alphas", "alpha_bars", "sqrt_alpha_bars", "sqrt_one_minus_alpha_bars"]:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        b = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return a * x0 + b * noise

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        b = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return (xt - b * eps_pred) / (a + 1.0e-12)

    def training_loss(self, model, theta0_norm: torch.Tensor, geometry: torch.Tensor, condition: torch.Tensor):
        b = theta0_norm.shape[0]
        t = torch.randint(0, self.steps, (b,), device=theta0_norm.device)
        noise = torch.randn_like(theta0_norm)
        xt = self.q_sample(theta0_norm, t, noise)
        eps_pred = model(xt, t, geometry, condition)
        loss = F.mse_loss(eps_pred, noise)
        x0_pred = self.predict_x0(xt, t, eps_pred)
        return loss, x0_pred, t

    @torch.no_grad()
    def p_sample(self, model, xt: torch.Tensor, t_value: int, geometry: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        b = xt.shape[0]
        t = torch.full((b,), t_value, device=xt.device, dtype=torch.long)
        beta_t = self.betas[t].view(-1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
        eps_pred = model(xt, t, geometry, condition)
        mean = (xt - beta_t / torch.sqrt(1.0 - alpha_bar_t + 1.0e-12) * eps_pred) / torch.sqrt(alpha_t + 1.0e-12)
        if t_value > 0:
            return mean + torch.sqrt(beta_t) * torch.randn_like(xt)
        return mean

    @torch.no_grad()
    def sample(self, model, shape, geometry: torch.Tensor, condition: torch.Tensor, sample_steps: int | None = None) -> torch.Tensor:
        sample_steps = sample_steps or self.steps
        if sample_steps >= self.steps:
            schedule = list(range(self.steps - 1, -1, -1))
        else:
            schedule = torch.linspace(self.steps - 1, 0, sample_steps, device=self.device).long().tolist()
        xt = torch.randn(shape, device=self.device)
        for t_value in schedule:
            xt = self.p_sample(model, xt, int(t_value), geometry, condition)
        return xt.clamp(0.0, 1.0)
