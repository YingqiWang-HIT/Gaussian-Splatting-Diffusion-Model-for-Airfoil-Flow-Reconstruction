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
from pgausdiff.data import AirfoilFlowDataset
from pgausdiff.gaussian import render_gaussian_field
from pgausdiff.metrics import average_physical_residual, psnr, ssim
from pgausdiff.utils import get_device, make_query_grid, save_json, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pgausdiff.yaml")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    split = args.split or cfg.data.test_split
    ds = AirfoilFlowDataset(cfg.data.root, split=split)
    pred_dir = Path(args.pred_dir)

    records = []
    for i in tqdm(range(len(ds)), desc="evaluating"):
        sample = ds[i]
        pred_file = pred_dir / f"{sample['id']}.npz"
        if not pred_file.exists():
            continue
        pred = np.load(pred_file)
        pred_field = torch.from_numpy(pred["pred_field"]).float().to(device)
        theta = torch.from_numpy(pred["theta"]).float().to(device)
        if "field" in sample:
            target = sample["field"].float().to(device)
            if target.shape[:2] != pred_field.shape[:2]:
                h, w = pred_field.shape[:2]
                query = make_query_grid(h, w, device=device)
                # If theta ground truth exists, render same resolution as reference target proxy.
                theta_ref = sample["theta"].float().to(device)
                target = render_gaussian_field(theta_ref, query, chunk_size=4096)[0].reshape(h, w, 4)
            psnr_value = psnr(pred_field, target)
            ssim_value = ssim(pred_field, target)
        else:
            psnr_value = float("nan")
            ssim_value = float("nan")
        xy = torch.from_numpy(pred["query_xy"].reshape(-1, 2)).float().to(device)
        residual = average_physical_residual(theta, xy, gamma=cfg.physics.gamma)
        records.append({"id": sample["id"], "psnr": psnr_value, "ssim": ssim_value, "physical_residual": residual})

    summary = {
        "num_samples": len(records),
        "psnr_mean": float(np.nanmean([r["psnr"] for r in records])) if records else float("nan"),
        "ssim_mean": float(np.nanmean([r["ssim"] for r in records])) if records else float("nan"),
        "physical_residual_mean": float(np.nanmean([r["physical_residual"] for r in records])) if records else float("nan"),
        "records": records,
    }
    out_file = pred_dir / "metrics.json"
    save_json(summary, out_file)
    print(summary)
    print(f"Metrics written to {out_file}")


if __name__ == "__main__":
    main()
