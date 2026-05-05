from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pgausdiff.config import load_config
from pgausdiff.data import AirfoilFlowDataset
from pgausdiff.gaussian import render_gaussian_field, sanitize_theta
from pgausdiff.utils import get_device, set_seed


def init_theta(num_primitives: int, device: torch.device):
    theta = torch.zeros(num_primitives, 9, device=device)
    theta[:, 0:2] = torch.rand(num_primitives, 2, device=device)
    theta[:, 2:4] = 0.04 + 0.06 * torch.rand(num_primitives, 2, device=device)
    theta[:, 4] = (torch.rand(num_primitives, device=device) - 0.5) * 2 * torch.pi
    theta[:, 5] = 1.0
    theta[:, 6] = 0.8
    theta[:, 7] = 0.0
    theta[:, 8] = 1.0
    return torch.nn.Parameter(theta)


def prefit_one(sample, num_primitives: int, iterations: int, lr: float, device: torch.device):
    if "field" not in sample or "query_xy" not in sample:
        raise ValueError("Prefitting requires sample keys `field` and `query_xy`.")
    field = sample["field"].to(device)
    query_xy = sample["query_xy"].reshape(-1, 2).to(device)
    target = field.reshape(-1, 4).to(device)
    theta = init_theta(num_primitives, device)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(iterations):
        opt.zero_grad(set_to_none=True)
        pred = render_gaussian_field(theta, query_xy, chunk_size=4096)[0]
        loss = F.mse_loss(pred, target)
        loss.backward()
        opt.step()
        with torch.no_grad():
            theta.data = sanitize_theta(theta.data)
    return theta.detach().cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pgausdiff.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_root", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    ds = AirfoilFlowDataset(cfg.data.root, split=args.split)
    output_root = Path(args.output_root or cfg.data.root) / f"{args.split}_prefit"
    output_root.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(ds)), desc="prefitting"):
        sample = ds[i]
        theta = prefit_one(sample, cfg.data.primitive_count, args.iterations, args.lr, device)
        src = np.load(ds.files[i])
        data = {k: src[k] for k in src.files}
        data["theta"] = theta
        np.savez_compressed(output_root / ds.files[i].name, **data)
    print(f"Prefitted Gaussian parameters written to {output_root}")


if __name__ == "__main__":
    main()
