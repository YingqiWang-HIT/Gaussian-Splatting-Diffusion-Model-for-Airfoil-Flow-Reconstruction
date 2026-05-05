from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class DataConfig:
    root: str = "data/demo"
    train_split: str = "train"
    test_split: str = "test"
    geometry_points: int = 200
    primitive_count: int = 128
    field_channels: int = 4
    theta_min: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.005, 0.005, -3.14159, 0.0, -2.0, -2.0, 0.0])
    theta_max: list[float] = field(default_factory=lambda: [1.0, 1.0, 0.25, 0.25, 3.14159, 2.5, 2.0, 2.0, 2.5])


@dataclass
class ModelConfig:
    primitive_dim: int = 9
    flow_dim: int = 4
    embed_dim: int = 256
    geometry_conv_layers: int = 3
    condition_mlp_layers: int = 3
    transformer_layers: int = 4
    attention_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.0


@dataclass
class DiffusionConfig:
    steps: int = 1000
    sample_steps: int = 100
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2


@dataclass
class PhysicsConfig:
    gamma: float = 1.4
    gas_constant: float = 287.05
    lambda_phys: float = 0.1
    lambda_weak: float = 0.01
    lambda_bc: float = 1.0
    collocation_points: int = 2048
    control_volumes: int = 64
    integral_points_per_side: int = 4
    eps: float = 1.0e-7


@dataclass
class RefinementConfig:
    rounds: int = 3
    fine_tune_steps: int = 50
    split_percentile: float = 90.0
    split_scale: float = 0.5
    merge_threshold: float = 0.01
    distance_threshold: float = 0.05
    fine_tune_lr: float = 1.0e-4
    max_primitives: int = 6000


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 500
    batch_size: int = 32
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    num_workers: int = 2
    save_every: int = 20
    output_dir: str = "outputs"
    device: str = "cuda"
    use_physics_loss: bool = True
    render_points_for_physics: int = 1024


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _update_dataclass(obj: Any, values: Dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: str | Path) -> Config:
    cfg = Config()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    _update_dataclass(cfg, raw)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_dict(cfg), f, sort_keys=False)


def _to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
    return obj
