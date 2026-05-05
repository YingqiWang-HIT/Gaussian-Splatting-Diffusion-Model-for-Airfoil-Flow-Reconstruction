from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class AirfoilFlowDataset(Dataset):
    """Dataset for airfoil geometry, operating condition, Gaussian theta, and optional CFD field.

    Each sample is an `.npz` file. Required keys:
        geometry: [Ns, 2]
        condition: [3] = [Ma, Re, alpha_deg]
        theta: [N, 9]

    Optional keys:
        field: [H, W, 4]
        query_xy: [H, W, 2] or [M, 2]
        wall_xy: [Nw, 2]
        wall_normals: [Nw, 2]
    """

    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root)
        self.split = split
        self.files = sorted((self.root / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz samples found under {self.root / split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        path = self.files[idx]
        data = np.load(path)
        sample: Dict[str, torch.Tensor | str] = {
            "id": path.stem,
            "geometry": torch.from_numpy(data["geometry"]).float(),
            "condition": torch.from_numpy(data["condition"]).float(),
            "theta": torch.from_numpy(data["theta"]).float(),
        }
        for key in ["field", "query_xy", "wall_xy", "wall_normals"]:
            if key in data:
                sample[key] = torch.from_numpy(data[key]).float()
        return sample


def collate_airfoil_batch(batch: List[Dict]) -> Dict[str, torch.Tensor | List[str]]:
    ids = [b["id"] for b in batch]
    out = {
        "id": ids,
        "geometry": torch.stack([b["geometry"] for b in batch], dim=0),
        "condition": torch.stack([b["condition"] for b in batch], dim=0),
        "theta": torch.stack([b["theta"] for b in batch], dim=0),
    }
    for key in ["field", "query_xy", "wall_xy", "wall_normals"]:
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out
