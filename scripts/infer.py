from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pgausdiff.config import load_config
from pgausdiff.data import AirfoilFlowDataset, collate_airfoil_batch
from pgausdiff.diffusion import GaussianDiffusion
from pgausdiff.gaussian import denormalize_theta, render_gaussian_field
from pgausdiff.model import PGausDiff
from pgausdiff.refinement import shock_aware_refine
from pgausdiff.utils import ensure_dir, get_device, make_query_grid, set_seed


def build_model(cfg):
    return PGausDiff(
        primitive_dim=cfg.model.primitive_dim,
        embed_dim=cfg.model.embed_dim,
        geometry_conv_layers=cfg.model.geometry_conv_layers,
        condition_mlp_layers=cfg.model.condition_mlp_layers,
        transformer_layers=cfg.model.transformer_layers,
        attention_heads=cfg.model.attention_heads,
        ffn_dim=cfg.model.ffn_dim,
        dropout=cfg.model.dropout,
    )


@torch.no_grad()
def load_model(cfg, checkpoint, device):
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pgausdiff.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--no_refine", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    split = args.split or cfg.data.test_split
    ds = AirfoilFlowDataset(cfg.data.root, split=split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_airfoil_batch)

    model = load_model(cfg, args.checkpoint, device)
    diffusion = GaussianDiffusion(cfg.diffusion.steps, cfg.diffusion.beta_start, cfg.diffusion.beta_end, device=device)
    theta_min = torch.tensor(cfg.data.theta_min, device=device).view(1, 1, -1)
    theta_max = torch.tensor(cfg.data.theta_max, device=device).view(1, 1, -1)
    out_dir = ensure_dir(Path(cfg.train.output_dir) / "predictions")
    grid = make_query_grid(args.height, args.width, device=device)

    for idx, batch in enumerate(tqdm(loader, desc="inference")):
        if args.num_samples is not None and idx >= args.num_samples:
            break
        geometry = batch["geometry"].to(device)
        condition = batch["condition"].to(device)
        shape = (1, cfg.data.primitive_count, cfg.model.primitive_dim)
        theta_norm = diffusion.sample(model, shape, geometry, condition, sample_steps=cfg.diffusion.sample_steps)
        theta = denormalize_theta(theta_norm, theta_min, theta_max)[0]
        stats_dict = {}
        if not args.no_refine:
            colloc = torch.rand(min(cfg.physics.collocation_points, 2048), 2, device=device)
            theta, stats = shock_aware_refine(
                theta,
                collocation_xy=colloc,
                rounds=cfg.refinement.rounds,
                fine_tune_steps=cfg.refinement.fine_tune_steps,
                split_percentile=cfg.refinement.split_percentile,
                split_scale=cfg.refinement.split_scale,
                merge_threshold=cfg.refinement.merge_threshold,
                distance_threshold=cfg.refinement.distance_threshold,
                fine_tune_lr=cfg.refinement.fine_tune_lr,
                max_primitives=cfg.refinement.max_primitives,
                gamma=cfg.physics.gamma,
                eps=cfg.physics.eps,
            )
            stats_dict = stats.__dict__
        field = render_gaussian_field(theta, grid, eps=cfg.physics.eps, chunk_size=4096)[0].reshape(args.height, args.width, 4)
        np.savez_compressed(
            out_dir / f"{batch['id'][0]}.npz",
            theta=theta.detach().cpu().numpy().astype(np.float32),
            pred_field=field.detach().cpu().numpy().astype(np.float32),
            query_xy=grid.detach().cpu().numpy().reshape(args.height, args.width, 2).astype(np.float32),
            condition=batch["condition"][0].numpy().astype(np.float32),
            geometry=batch["geometry"][0].numpy().astype(np.float32),
            refinement_stats=np.array([stats_dict.get("initial_count", theta.shape[0]), stats_dict.get("final_count", theta.shape[0]), stats_dict.get("split_count", 0), stats_dict.get("merge_count", 0)], dtype=np.int32),
        )
    print(f"Predictions written to {out_dir}")


if __name__ == "__main__":
    main()
