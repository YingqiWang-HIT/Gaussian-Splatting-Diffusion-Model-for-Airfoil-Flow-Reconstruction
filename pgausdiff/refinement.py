from __future__ import annotations

from dataclasses import dataclass

import torch

from .gaussian import render_with_derivatives, sanitize_theta
from .physics import physics_loss


@dataclass
class RefinementStats:
    initial_count: int
    final_count: int
    split_count: int
    merge_count: int


@torch.no_grad()
def split_primitives_by_pressure_gradient(theta: torch.Tensor,
                                          split_percentile: float = 90.0,
                                          split_scale: float = 0.5,
                                          eps: float = 1.0e-7,
                                          max_primitives: int = 6000):
    """Directional primitive splitting along analytical pressure-gradient direction.

    This function is intended for one sample theta [N, 9].
    """
    theta = sanitize_theta(theta, eps=eps)
    centers = theta[:, 0:2]
    _, dqdx, dqdy = render_with_derivatives(theta, centers, eps=eps, chunk_size=512)
    dpdx = dqdx[0, :, 3]
    dpdy = dqdy[0, :, 3]
    grad_norm = torch.sqrt(dpdx**2 + dpdy**2 + eps)
    scales = theta[:, 2:4].max(dim=-1).values
    scores = grad_norm * scales
    threshold = torch.quantile(scores, split_percentile / 100.0)
    to_split = scores > threshold

    # Respect max primitive count.
    if theta.shape[0] + int(to_split.sum().item()) > max_primitives:
        allowed = max(0, max_primitives - theta.shape[0])
        idx = torch.argsort(scores, descending=True)[:allowed]
        mask = torch.zeros_like(to_split)
        mask[idx] = True
        to_split = mask

    new = []
    split_count = 0
    for i in range(theta.shape[0]):
        row = theta[i]
        if bool(to_split[i]):
            n = torch.stack([dpdx[i], dpdy[i]]) / (grad_norm[i] + eps)
            delta = split_scale * scales[i]
            plus = row.clone(); minus = row.clone()
            plus[0:2] = (row[0:2] + delta * n).clamp(0.0, 1.0)
            minus[0:2] = (row[0:2] - delta * n).clamp(0.0, 1.0)
            plus[2:4] = row[2:4] * torch.tensor([0.5, 1.0], device=row.device, dtype=row.dtype)
            minus[2:4] = row[2:4] * torch.tensor([0.5, 1.0], device=row.device, dtype=row.dtype)
            plus[4] = torch.atan2(n[1], n[0])
            minus[4] = plus[4]
            new.extend([plus, minus])
            split_count += 1
        else:
            new.append(row)
    return torch.stack(new, dim=0), split_count


@torch.no_grad()
def merge_similar_primitives(theta: torch.Tensor,
                             merge_threshold: float = 0.01,
                             distance_threshold: float = 0.05,
                             max_pairs: int = 20000):
    """Greedy merge of nearby primitives with similar primitive variables."""
    theta = sanitize_theta(theta)
    n = theta.shape[0]
    used = torch.zeros(n, dtype=torch.bool, device=theta.device)
    merged = []
    merge_count = 0
    if n == 0:
        return theta, 0
    dmat = torch.cdist(theta[:, 0:2], theta[:, 0:2])
    dmat.fill_diagonal_(float("inf"))
    pairs = torch.nonzero(dmat < distance_threshold, as_tuple=False)
    if pairs.shape[0] > max_pairs:
        pairs = pairs[:max_pairs]
    pair_set = {(int(i), int(j)) for i, j in pairs.tolist() if i < j}
    partner = {}
    for i, j in pair_set:
        if i in partner or j in partner:
            continue
        if torch.norm(theta[i, 5:9] - theta[j, 5:9]) < merge_threshold:
            partner[i] = j
            partner[j] = i
    for i in range(n):
        if used[i]:
            continue
        if i in partner and not used[partner[i]]:
            j = partner[i]
            row = theta[i].clone()
            row[0:2] = 0.5 * (theta[i, 0:2] + theta[j, 0:2])
            row[2:4] = torch.sqrt(theta[i, 2:4] ** 2 + theta[j, 2:4] ** 2)
            row[5:9] = 0.5 * (theta[i, 5:9] + theta[j, 5:9])
            merged.append(row)
            used[i] = True; used[j] = True
            merge_count += 1
        else:
            merged.append(theta[i])
            used[i] = True
    return torch.stack(merged, dim=0), merge_count


def shock_aware_refine(theta: torch.Tensor,
                       collocation_xy: torch.Tensor | None = None,
                       rounds: int = 3,
                       fine_tune_steps: int = 50,
                       split_percentile: float = 90.0,
                       split_scale: float = 0.5,
                       merge_threshold: float = 0.01,
                       distance_threshold: float = 0.05,
                       fine_tune_lr: float = 1.0e-4,
                       max_primitives: int = 6000,
                       gamma: float = 1.4,
                       eps: float = 1.0e-7):
    """SA-GTR refinement for one sample [N, 9]."""
    initial = theta.shape[0]
    cur = theta.detach().clone()
    split_total = 0
    merge_total = 0
    for _ in range(rounds):
        cur, split_count = split_primitives_by_pressure_gradient(
            cur, split_percentile=split_percentile, split_scale=split_scale, eps=eps, max_primitives=max_primitives
        )
        split_total += split_count
        cur, merge_count = merge_similar_primitives(
            cur, merge_threshold=merge_threshold, distance_threshold=distance_threshold
        )
        merge_total += merge_count
        if collocation_xy is not None and fine_tune_steps > 0:
            cur = torch.nn.Parameter(cur)
            opt = torch.optim.Adam([cur], lr=fine_tune_lr)
            for _ in range(fine_tune_steps):
                opt.zero_grad(set_to_none=True)
                loss = physics_loss(cur.unsqueeze(0), collocation_xy.unsqueeze(0), gamma=gamma, eps=eps).total
                loss.backward()
                opt.step()
                with torch.no_grad():
                    cur.data = sanitize_theta(cur.data, eps=eps)
            cur = cur.detach()
    stats = RefinementStats(initial_count=initial, final_count=cur.shape[0], split_count=split_total, merge_count=merge_total)
    return cur, stats
