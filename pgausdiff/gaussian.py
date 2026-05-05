from __future__ import annotations

import math
from typing import Tuple

import torch


# Theta layout per primitive:
# [x, y, sx, sy, theta, rho, u, v, p]


def normalize_theta(theta: torch.Tensor, theta_min: torch.Tensor, theta_max: torch.Tensor) -> torch.Tensor:
    return (theta - theta_min) / (theta_max - theta_min + 1.0e-12)


def denormalize_theta(theta_norm: torch.Tensor, theta_min: torch.Tensor, theta_max: torch.Tensor) -> torch.Tensor:
    return theta_norm * (theta_max - theta_min + 1.0e-12) + theta_min


def sanitize_theta(theta: torch.Tensor, eps: float = 1.0e-5) -> torch.Tensor:
    """Clamp parameters to numerically valid ranges while keeping gradients.

    The implementation avoids in-place writes, so it is safe inside losses that
    back-propagate to the denoising network.
    """
    xy = theta[..., 0:2].clamp(0.0, 1.0)
    scales = theta[..., 2:4].clamp(min=eps)
    angle = torch.atan2(torch.sin(theta[..., 4:5]), torch.cos(theta[..., 4:5]))
    rho = theta[..., 5:6].clamp(min=eps)
    uv = theta[..., 6:8]
    pressure = theta[..., 8:9].clamp(min=eps)
    return torch.cat([xy, scales, angle, rho, uv, pressure], dim=-1)


def _kernel_and_grad(theta: torch.Tensor, xy: torch.Tensor, eps: float = 1.0e-7) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Gaussian kernel phi and its spatial derivatives.

    Args:
        theta: [B, N, 9]
        xy: [B, M, 2] or [M, 2]
    Returns:
        phi: [B, M, N]
        dphidx: [B, M, N]
        dphidy: [B, M, N]
    """
    if theta.dim() == 2:
        theta = theta.unsqueeze(0)
    if xy.dim() == 2:
        xy = xy.unsqueeze(0).expand(theta.shape[0], -1, -1)

    theta = sanitize_theta(theta, eps=eps)
    mu = theta[..., 0:2]  # [B, N, 2]
    sx = theta[..., 2].clamp_min(eps)
    sy = theta[..., 3].clamp_min(eps)
    angle = theta[..., 4]

    dx = xy[:, :, None, 0] - mu[:, None, :, 0]
    dy = xy[:, :, None, 1] - mu[:, None, :, 1]
    c = torch.cos(angle)[:, None, :]
    s = torch.sin(angle)[:, None, :]

    xloc = dx * c + dy * s
    yloc = -dx * s + dy * c
    invsx2 = 1.0 / (sx[:, None, :] ** 2 + eps)
    invsy2 = 1.0 / (sy[:, None, :] ** 2 + eps)
    exponent = -0.5 * (xloc**2 * invsx2 + yloc**2 * invsy2)
    phi = torch.exp(exponent.clamp(min=-80.0, max=20.0))

    # Sigma^{-1}(x-mu) in world coordinates.
    # dphi/dx_j = -[Sigma^{-1}(x-mu)]_j * phi
    a_world_x = (xloc * invsx2) * c + (yloc * invsy2) * (-s)
    a_world_y = (xloc * invsx2) * s + (yloc * invsy2) * c
    dphidx = -a_world_x * phi
    dphidy = -a_world_y * phi
    return phi, dphidx, dphidy


def render_gaussian_field(theta: torch.Tensor, xy: torch.Tensor, eps: float = 1.0e-7, chunk_size: int | None = None) -> torch.Tensor:
    """Render primitive variables [rho, u, v, p] at query coordinates.

    Args:
        theta: [B, N, 9] or [N, 9]
        xy: [B, M, 2] or [M, 2]
    Returns:
        q: [B, M, 4]
    """
    if theta.dim() == 2:
        theta = theta.unsqueeze(0)
    if xy.dim() == 2:
        xy = xy.unsqueeze(0).expand(theta.shape[0], -1, -1)

    if chunk_size is not None and xy.shape[1] > chunk_size:
        outs = []
        for start in range(0, xy.shape[1], chunk_size):
            outs.append(render_gaussian_field(theta, xy[:, start:start + chunk_size], eps=eps, chunk_size=None))
        return torch.cat(outs, dim=1)

    theta = sanitize_theta(theta, eps=eps)
    phi, _, _ = _kernel_and_grad(theta, xy, eps=eps)
    weights = theta[..., 5:9]
    numerator = torch.einsum("bmn,bnc->bmc", phi, weights)
    denominator = phi.sum(dim=-1, keepdim=True) + eps
    return numerator / denominator


def render_with_derivatives(theta: torch.Tensor, xy: torch.Tensor, eps: float = 1.0e-7, chunk_size: int | None = None):
    """Render q and closed-form spatial derivatives dq/dx and dq/dy.

    Returns:
        q, dqdx, dqdy, each [B, M, 4]
    """
    if theta.dim() == 2:
        theta = theta.unsqueeze(0)
    if xy.dim() == 2:
        xy = xy.unsqueeze(0).expand(theta.shape[0], -1, -1)

    if chunk_size is not None and xy.shape[1] > chunk_size:
        qs, dxs, dys = [], [], []
        for start in range(0, xy.shape[1], chunk_size):
            q, dqdx, dqdy = render_with_derivatives(theta, xy[:, start:start + chunk_size], eps=eps, chunk_size=None)
            qs.append(q); dxs.append(dqdx); dys.append(dqdy)
        return torch.cat(qs, 1), torch.cat(dxs, 1), torch.cat(dys, 1)

    theta = sanitize_theta(theta, eps=eps)
    phi, dphidx, dphidy = _kernel_and_grad(theta, xy, eps=eps)
    weights = theta[..., 5:9]

    numerator = torch.einsum("bmn,bnc->bmc", phi, weights)
    denominator = phi.sum(dim=-1, keepdim=True) + eps
    q = numerator / denominator

    dnum_dx = torch.einsum("bmn,bnc->bmc", dphidx, weights)
    dnum_dy = torch.einsum("bmn,bnc->bmc", dphidy, weights)
    dden_dx = dphidx.sum(dim=-1, keepdim=True)
    dden_dy = dphidy.sum(dim=-1, keepdim=True)

    dqdx = (dnum_dx * denominator - numerator * dden_dx) / (denominator**2 + eps)
    dqdy = (dnum_dy * denominator - numerator * dden_dy) / (denominator**2 + eps)
    return q, dqdx, dqdy


def append_temperature(q: torch.Tensor, gas_constant: float = 287.05, eps: float = 1.0e-7) -> torch.Tensor:
    rho = q[..., 0].clamp_min(eps)
    p = q[..., 3].clamp_min(eps)
    temperature = p / (rho * gas_constant)
    return torch.cat([q, temperature.unsqueeze(-1)], dim=-1)
