"""LeHome experiment configuration and JSON loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .modeling import LiquidPolicyConfig, SharedBackboneConfig


@dataclass(frozen=True)
class LeHomeConfig:
    # -- Dataset --
    dataset_root: str = "Datasets/example/four_types_merged"
    dataset_repo_id: str = "lehome"

    # -- Observation keys --
    rgb_keys: Tuple[str, ...] = (
        "observation.images.top_rgb",
        "observation.images.left_rgb",
        "observation.images.right_rgb",
    )
    low_dim_keys: Tuple[str, ...] = ("observation.state",)

    # -- Image preprocessing --
    rgb_image_size: int = 96
    rgb_encoder_kind: str = "clip"
    freeze_rgb_encoder: bool = True
    clip_model_name: str = "openai/clip-vit-base-patch32"

    # -- Backbone dims --
    model_dim: int = 512
    temporal_num_layers: int = 5
    temporal_num_heads: int = 8
    temporal_mlp_ratio: int = 4
    dropout: float = 0.1

    # -- Policy / horizons --
    action_dim: int = 12
    observation_horizon: int = 2
    action_horizon: int = 8
    prediction_horizon: int = 16
    control_hz: int = 30
    deployment_env_hz: int = 90

    # -- MDN / Liquid --
    num_mixtures: int = 5
    liquid_hidden_dim: int = 960
    decoder_hidden_dim: int = 960
    action_embed_dim: int = 512
    use_cfc: bool = True

    # -- Inference --
    sample_selection_mode: str = "best_of_k"
    sample_selection_k: int = 10

    # -- Training --
    batch_size: int = 64
    num_epochs: int = 120
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0
    warmup_epochs: int = 3
    warmup_steps: Optional[int] = None
    num_data_workers: int = 4

    # -- Teacher forcing schedule --
    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.15
    free_running_weight_start: float = 0.20
    free_running_weight_end: float = 0.75

    # -- Normalization --
    normalize_actions: bool = True
    normalize_state: bool = True

    # -- Visual learning --
    state_dropout_p: float = 0.15
    enable_image_augmentation: bool = True

    # -- Output --
    output_dir: str = "outputs/liquid_lehome"
    save_every_n_epochs: int = 25
    log_every_n_steps: int = 50
    val_split_ratio: float = 0.1
    seed: int = 42
    device: str = "cuda"


def compute_env_steps_per_policy_step(control_hz: int, env_step_hz: int) -> int:
    """Return how many simulator steps should reuse one policy action."""
    if control_hz <= 0 or env_step_hz <= 0:
        raise ValueError(
            f"control_hz and env_step_hz must be positive, got {control_hz} and {env_step_hz}"
        )

    ratio = float(env_step_hz) / float(control_hz)
    repeat = int(round(ratio))
    if repeat < 1 or abs(ratio - repeat) > 1e-6:
        raise ValueError(
            "LeHome control alignment requires env_step_hz to be an integer multiple "
            f"of control_hz, got control_hz={control_hz}, env_step_hz={env_step_hz}"
        )
    return repeat


def load_config(path: str) -> LeHomeConfig:
    """Load a LeHomeConfig from a JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    # Convert list fields to tuples
    for key in ("rgb_keys", "low_dim_keys"):
        if key in raw and isinstance(raw[key], list):
            raw[key] = tuple(raw[key])
    return LeHomeConfig(**raw)


def config_to_backbone(cfg: LeHomeConfig) -> SharedBackboneConfig:
    """Build a SharedBackboneConfig from LeHomeConfig fields."""
    return SharedBackboneConfig(
        rgb_keys=cfg.rgb_keys,
        low_dim_keys=cfg.low_dim_keys,
        model_dim=cfg.model_dim,
        temporal_num_layers=cfg.temporal_num_layers,
        temporal_num_heads=cfg.temporal_num_heads,
        temporal_mlp_ratio=cfg.temporal_mlp_ratio,
        dropout=cfg.dropout,
        rgb_encoder_kind=cfg.rgb_encoder_kind,
        freeze_rgb_encoder=cfg.freeze_rgb_encoder,
        clip_model_name=cfg.clip_model_name,
        rgb_image_size=cfg.rgb_image_size,
    )


def config_to_policy(cfg: LeHomeConfig) -> LiquidPolicyConfig:
    """Build a LiquidPolicyConfig from LeHomeConfig fields."""
    return LiquidPolicyConfig(
        action_dim=cfg.action_dim,
        observation_horizon=cfg.observation_horizon,
        action_horizon=cfg.action_horizon,
        prediction_horizon=cfg.prediction_horizon,
        num_mixtures=cfg.num_mixtures,
        liquid_hidden_dim=cfg.liquid_hidden_dim,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        action_embed_dim=cfg.action_embed_dim,
        use_cfc=cfg.use_cfc,
        sample_selection_mode=cfg.sample_selection_mode,
        sample_selection_k=cfg.sample_selection_k,
    )
