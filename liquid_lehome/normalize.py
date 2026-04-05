"""Action and state normalization utilities for LeHome data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch


@dataclass
class NormalizationStats:
    min: np.ndarray
    max: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def load_stats(dataset_root: str) -> Dict[str, NormalizationStats]:
    """Load normalization statistics from ``meta/stats.json``."""
    stats_path = Path(dataset_root) / "meta" / "stats.json"
    raw = json.loads(stats_path.read_text(encoding="utf-8"))
    result: Dict[str, NormalizationStats] = {}
    for key in ("observation.state", "action"):
        if key not in raw:
            continue
        entry = raw[key]
        result[key] = NormalizationStats(
            min=np.asarray(entry["min"], dtype=np.float32).flatten(),
            max=np.asarray(entry["max"], dtype=np.float32).flatten(),
            mean=np.asarray(entry["mean"], dtype=np.float32).flatten(),
            std=np.asarray(entry["std"], dtype=np.float32).flatten(),
        )
    return result


def _to_tensor(arr: np.ndarray, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def min_max_normalize(
    x: torch.Tensor, stats: NormalizationStats
) -> torch.Tensor:
    """Map values from [min, max] to [-1, 1]."""
    lo = _to_tensor(stats.min, x.device)
    hi = _to_tensor(stats.max, x.device)
    span = (hi - lo).clamp(min=1e-8)
    return 2.0 * (x - lo) / span - 1.0


def min_max_denormalize(
    x: torch.Tensor, stats: NormalizationStats
) -> torch.Tensor:
    """Map values from [-1, 1] back to [min, max]."""
    lo = _to_tensor(stats.min, x.device)
    hi = _to_tensor(stats.max, x.device)
    span = hi - lo
    return (x + 1.0) * 0.5 * span + lo
