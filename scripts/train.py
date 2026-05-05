from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pgausdiff.config import load_config, save_config
from pgausdiff.data import AirfoilFlowDataset, collate_airfoil_batch
from pgausdiff.diffusion import GaussianDiffusion
from pgausdiff.gaussian import denormalize_theta, normalize_theta
from pgausdiff.model import PGausDiff
from pgausdiff.physics import physics_loss
from pgausdiff.utils import ensure_dir, get_device, make_query_grid, set_seed, count_parameters


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pgausdiff.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    out_dir = ensure_dir(cfg.train.output_dir)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    save_config(cfg, out_dir / "config.yaml")

    ds = AirfoilFlowDataset(cfg.data.root, split=cfg.data.train_split)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, collate_fn=collate_airfoil_batch)

    model = build_model(cfg).to(device)
    diffusion = GaussianDiffusion(cfg.diffusion.steps, cfg.diffusion.beta_start, cfg.diffusion.beta_end, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, betas=(0.9, 0.999), weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.train.epochs))
    theta_min = torch.tensor(cfg.data.theta_min, device=device).view(1, 1, -1)
    theta_max = torch.tensor(cfg.data.theta_max, device=device).view(1, 1, -1)
    print(f"Model parameters: {count_parameters(model):,}")

    best_loss = float("inf")
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total = 0.0
        pbar = tqdm(loader, desc=f"epoch {epoch}/{cfg.train.epochs}")
        for batch in pbar:
            geometry = batch["geometry"].to(device)
            condition = batch["condition"].to(device)
            theta0 = batch["theta"].to(device)
            theta0_norm = normalize_theta(theta0, theta_min, theta_max).clamp(0.0, 1.0)
            diff_loss, x0_pred_norm, _ = diffusion.training_loss(model, theta0_norm, geometry, condition)
            loss = diff_loss
            phys_value = torch.tensor(0.0, device=device)
            if cfg.train.use_physics_loss:
                pred_theta = denormalize_theta(x0_pred_norm.clamp(0.0, 1.0), theta_min, theta_max)
                xy = torch.rand(pred_theta.shape[0], cfg.train.render_points_for_physics, 2, device=device)
                weak_centers = torch.rand(pred_theta.shape[0], max(1, cfg.physics.control_volumes // 4), 2, device=device)
                phys = physics_loss(
                    pred_theta,
                    xy,
                    weak_centers=weak_centers,
                    gamma=cfg.physics.gamma,
                    lambda_weak=cfg.physics.lambda_weak,
                    lambda_bc=cfg.physics.lambda_bc,
                    eps=cfg.physics.eps,
                )
                phys_value = phys.total
                loss = loss + cfg.physics.lambda_phys * phys.total
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            total += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4e}", diff=f"{diff_loss.item():.4e}", phys=f"{phys_value.item():.4e}")
        scheduler.step()
        avg = total / max(1, len(loader))
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "config": args.config,
            "theta_min": cfg.data.theta_min,
            "theta_max": cfg.data.theta_max,
        }
        if epoch % cfg.train.save_every == 0:
            torch.save(ckpt, ckpt_dir / f"epoch_{epoch:04d}.pt")
        if avg < best_loss:
            best_loss = avg
            torch.save(ckpt, ckpt_dir / "best.pt")
        print(f"epoch={epoch} avg_loss={avg:.6e} best={best_loss:.6e}")


if __name__ == "__main__":
    main()
