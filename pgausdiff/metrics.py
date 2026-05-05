from __future__ import annotations

import numpy as np
import torch

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:  # pragma: no cover
    ssim_fn = None

from .physics import compressible_residuals
from .gaussian import render_with_derivatives


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float | None = None) -> float:
    err = mse(pred, target).item()
    if err <= 0:
        return float("inf")
    if data_range is None:
        data_range = float(target.max().item() - target.min().item())
        if data_range <= 1.0e-12:
            data_range = float(target.max().item())
        if data_range <= 1.0e-12:
            data_range = 1.0
    return 10.0 * np.log10((data_range ** 2) / err)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()
    if pred_np.ndim == 3 and pred_np.shape[-1] <= 8:
        vals = []
        for c in range(pred_np.shape[-1]):
            dr = float(tgt_np[..., c].max() - tgt_np[..., c].min()) or 1.0
            if ssim_fn is None:
                vals.append(1.0 - np.mean((pred_np[..., c] - tgt_np[..., c]) ** 2) / (dr ** 2 + 1e-12))
            else:
                vals.append(ssim_fn(tgt_np[..., c], pred_np[..., c], data_range=dr))
        return float(np.mean(vals))
    dr = float(tgt_np.max() - tgt_np.min()) or 1.0
    if ssim_fn is None:
        return float(1.0 - np.mean((pred_np - tgt_np) ** 2) / (dr ** 2 + 1e-12))
    return float(ssim_fn(tgt_np, pred_np, data_range=dr))


@torch.no_grad()
def average_physical_residual(theta: torch.Tensor, xy: torch.Tensor, gamma: float = 1.4) -> float:
    q, dqdx, dqdy = render_with_derivatives(theta, xy, chunk_size=4096)
    residuals = compressible_residuals(q, dqdx, dqdy, gamma=gamma)
    value = torch.sqrt(residuals.pow(2).sum(dim=-1)).mean()
    return float(value.item())
