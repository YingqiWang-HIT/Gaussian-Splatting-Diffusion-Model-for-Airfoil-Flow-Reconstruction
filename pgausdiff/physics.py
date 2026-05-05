from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from .gaussian import render_gaussian_field, render_with_derivatives


@dataclass
class PhysicsLossOutput:
    strong: torch.Tensor
    weak: torch.Tensor
    boundary: torch.Tensor
    total: torch.Tensor


def compressible_residuals(q: torch.Tensor, dqdx: torch.Tensor, dqdy: torch.Tensor, gamma: float = 1.4, eps: float = 1.0e-7) -> torch.Tensor:
    """Closed-form residuals for 2D steady compressible Euler equations.

    q = [rho, u, v, p].
    Returns [R_mass, R_mom_x, R_mom_y, R_energy].
    """
    rho = q[..., 0].clamp_min(eps)
    u = q[..., 1]
    v = q[..., 2]
    p = q[..., 3].clamp_min(eps)

    drdx, dudx, dvdx, dpdx = [dqdx[..., i] for i in range(4)]
    drdy, dudy, dvdy, dpdy = [dqdy[..., i] for i in range(4)]

    r_mass = drdx * u + rho * dudx + drdy * v + rho * dvdy

    r_mom_x = drdx * u**2 + 2.0 * rho * u * dudx + dpdx
    r_mom_x = r_mom_x + drdy * u * v + rho * dudy * v + rho * u * dvdy

    r_mom_y = drdx * u * v + rho * dudx * v + rho * u * dvdx
    r_mom_y = r_mom_y + drdy * v**2 + 2.0 * rho * v * dvdy + dpdy

    kinetic = 0.5 * rho * (u**2 + v**2)
    h = gamma / (gamma - 1.0) * p + kinetic
    dhdx = gamma / (gamma - 1.0) * dpdx + 0.5 * drdx * (u**2 + v**2) + rho * (u * dudx + v * dvdx)
    dhdy = gamma / (gamma - 1.0) * dpdy + 0.5 * drdy * (u**2 + v**2) + rho * (u * dudy + v * dvdy)
    r_energy = dudx * h + u * dhdx + dvdy * h + v * dhdy

    return torch.stack([r_mass, r_mom_x, r_mom_y, r_energy], dim=-1)


def strong_form_loss(theta: torch.Tensor, xy: torch.Tensor, gamma: float = 1.4, eps: float = 1.0e-7, shock_weight: bool = True) -> torch.Tensor:
    q, dqdx, dqdy = render_with_derivatives(theta, xy, eps=eps, chunk_size=4096)
    residuals = compressible_residuals(q, dqdx, dqdy, gamma=gamma, eps=eps)
    loss = residuals.pow(2).sum(dim=-1)
    if shock_weight:
        grad_p = torch.sqrt(dqdx[..., 3].pow(2) + dqdy[..., 3].pow(2) + eps)
        # De-emphasize pointwise strong residual near shocks; weak form handles discontinuities.
        weight = 1.0 / (1.0 + grad_p / (grad_p.mean(dim=1, keepdim=True) + eps))
        loss = loss * weight
    return loss.mean()


def flux_vectors(q: torch.Tensor, gamma: float = 1.4, eps: float = 1.0e-7) -> Tuple[torch.Tensor, torch.Tensor]:
    rho = q[..., 0].clamp_min(eps)
    u = q[..., 1]
    v = q[..., 2]
    p = q[..., 3].clamp_min(eps)
    E = p / (rho * (gamma - 1.0)) + 0.5 * (u**2 + v**2)
    fx = torch.stack([rho * u, rho * u**2 + p, rho * u * v, u * (rho * E + p)], dim=-1)
    fy = torch.stack([rho * v, rho * u * v, rho * v**2 + p, v * (rho * E + p)], dim=-1)
    return fx, fy


def weak_form_loss(theta: torch.Tensor, centers: torch.Tensor, half_size: float = 0.015, gamma: float = 1.4, eps: float = 1.0e-7) -> torch.Tensor:
    """Approximate finite-volume weak loss on square control volumes.

    For each center, samples midpoint on four sides and computes net flux.
    """
    if centers.dim() == 2:
        centers = centers.unsqueeze(0).expand(theta.shape[0], -1, -1)
    offsets = torch.tensor([[half_size, 0.0], [-half_size, 0.0], [0.0, half_size], [0.0, -half_size]], device=centers.device, dtype=centers.dtype)
    normals = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], device=centers.device, dtype=centers.dtype)
    points = centers[:, :, None, :] + offsets[None, None, :, :]
    b, n, s, _ = points.shape
    q = render_gaussian_field(theta, points.reshape(b, n * s, 2), eps=eps, chunk_size=4096).reshape(b, n, s, 4)
    fx, fy = flux_vectors(q, gamma=gamma, eps=eps)
    normal_flux = fx * normals[None, None, :, 0:1] + fy * normals[None, None, :, 1:2]
    net = normal_flux.sum(dim=2) * (2.0 * half_size)
    return net.pow(2).sum(dim=-1).mean()


def boundary_loss(theta: torch.Tensor, wall_xy: torch.Tensor | None = None, wall_normals: torch.Tensor | None = None, eps: float = 1.0e-7) -> torch.Tensor:
    if wall_xy is None:
        return theta.new_tensor(0.0)
    if wall_xy.dim() == 2:
        wall_xy = wall_xy.unsqueeze(0).expand(theta.shape[0], -1, -1)
    q = render_gaussian_field(theta, wall_xy, eps=eps, chunk_size=4096)
    u = q[..., 1]
    v = q[..., 2]
    if wall_normals is None:
        return (u.pow(2) + v.pow(2)).mean()
    if wall_normals.dim() == 2:
        wall_normals = wall_normals.unsqueeze(0).expand(theta.shape[0], -1, -1)
    un = u * wall_normals[..., 0] + v * wall_normals[..., 1]
    return un.pow(2).mean()


def physics_loss(theta: torch.Tensor,
                 collocation_xy: torch.Tensor,
                 weak_centers: torch.Tensor | None = None,
                 wall_xy: torch.Tensor | None = None,
                 wall_normals: torch.Tensor | None = None,
                 gamma: float = 1.4,
                 lambda_weak: float = 0.01,
                 lambda_bc: float = 1.0,
                 eps: float = 1.0e-7) -> PhysicsLossOutput:
    strong = strong_form_loss(theta, collocation_xy, gamma=gamma, eps=eps)
    weak = theta.new_tensor(0.0) if weak_centers is None else weak_form_loss(theta, weak_centers, gamma=gamma, eps=eps)
    bc = boundary_loss(theta, wall_xy=wall_xy, wall_normals=wall_normals, eps=eps)
    total = strong + lambda_weak * weak + lambda_bc * bc
    return PhysicsLossOutput(strong=strong, weak=weak, boundary=bc, total=total)
