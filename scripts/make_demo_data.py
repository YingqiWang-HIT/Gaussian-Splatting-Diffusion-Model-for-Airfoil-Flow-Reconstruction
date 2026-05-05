from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pgausdiff.config import load_config
from pgausdiff.gaussian import render_gaussian_field
from pgausdiff.utils import make_query_grid, set_seed


def naca_like_geometry(n: int = 200, thickness: float = 0.12, camber: float = 0.02):
    x = np.linspace(0.0, 1.0, n // 2)
    yt = 5 * thickness * (0.2969 * np.sqrt(np.maximum(x, 1e-6)) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    yc = camber * np.sin(np.pi * x)
    upper = np.stack([x, 0.5 + yc + yt], axis=-1)
    lower = np.stack([x[::-1], 0.5 + yc[::-1] - yt[::-1]], axis=-1)
    return np.concatenate([upper, lower], axis=0).astype(np.float32)


def synthetic_theta(num_primitives: int, ma: float, alpha: float, rng: np.random.Generator):
    x = rng.uniform(0.0, 1.0, num_primitives)
    y = rng.uniform(0.15, 0.85, num_primitives)
    sx = rng.uniform(0.02, 0.10, num_primitives)
    sy = rng.uniform(0.02, 0.10, num_primitives)
    angle = rng.uniform(-np.pi, np.pi, num_primitives)

    shock_x = 0.45 + 0.25 * (ma - 0.8) / 0.4 + 0.04 * np.sin(np.deg2rad(alpha))
    shock = np.exp(-((x - shock_x) ** 2) / 0.002)
    rho = 1.0 + 0.25 * shock + 0.05 * np.sin(2 * np.pi * y)
    u = ma * np.cos(np.deg2rad(alpha)) * (1.0 - 0.45 * shock)
    v = ma * np.sin(np.deg2rad(alpha)) + 0.08 * np.sin(2 * np.pi * x)
    p = 1.0 + 0.35 * shock + 0.10 * np.cos(np.pi * y)
    return np.stack([x, y, sx, sy, angle, rho, u, v, p], axis=-1).astype(np.float32)


def make_samples(root: Path, split: str, count: int, cfg, seed_offset: int = 0):
    out = root / split
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.train.seed + seed_offset)
    grid = make_query_grid(48, 48, device="cpu").numpy().astype(np.float32)
    for i in tqdm(range(count), desc=f"creating {split}"):
        ma = rng.uniform(0.8, 1.2)
        re = rng.uniform(1.0e6, 6.0e6)
        alpha = rng.uniform(-10.0, 10.0)
        thickness = rng.uniform(0.09, 0.15)
        camber = rng.uniform(-0.03, 0.03)
        geom = naca_like_geometry(cfg.data.geometry_points, thickness, camber)
        theta = synthetic_theta(cfg.data.primitive_count, ma, alpha, rng)
        with torch.no_grad():
            field = render_gaussian_field(torch.from_numpy(theta), torch.from_numpy(grid), chunk_size=4096)[0]
        field = field.reshape(48, 48, 4).numpy().astype(np.float32)
        np.savez_compressed(
            out / f"sample_{i:04d}.npz",
            geometry=geom,
            condition=np.array([ma, re, alpha], dtype=np.float32),
            theta=theta,
            query_xy=grid.reshape(48, 48, 2),
            field=field,
            wall_xy=geom,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--train_samples", type=int, default=12)
    parser.add_argument("--test_samples", type=int, default=4)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    root = Path(cfg.data.root)
    make_samples(root, cfg.data.train_split, args.train_samples, cfg, seed_offset=0)
    make_samples(root, cfg.data.test_split, args.test_samples, cfg, seed_offset=1000)
    print(f"Demo data written to {root}")


if __name__ == "__main__":
    main()
