"""Enhanced observation backbone that replaces mean fusion with concat + projection.

Subclasses the local ``SharedObsBackbone`` implementation used by
``liquid_lehome``. Key changes:

1. **Concat fusion** — RGB and state features are concatenated along the
   feature dimension and projected through a learned linear layer, instead of
   being averaged.  This prevents the model from ignoring either modality.
2. **State dropout** — During training, the state feature vector is randomly
   zeroed with probability ``state_dropout_p``, forcing the model to rely on
   visual features when state is unavailable.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence

import torch.nn as nn
import torch

from .modeling import BackboneOutput, SharedBackboneConfig, SharedObsBackbone

TensorDict = Mapping[str, torch.Tensor]


class EnhancedObsBackbone(SharedObsBackbone):
    """Drop-in replacement for SharedObsBackbone with better multi-modal fusion."""

    def __init__(
        self,
        obs_shapes: Mapping[str, Sequence[int]],
        config: SharedBackboneConfig,
        state_dropout_p: float = 0.0,
    ):
        super().__init__(obs_shapes, config)
        n_modalities = int(bool(self.rgb_keys)) + int(bool(self.low_dim_keys))
        if n_modalities > 1:
            self.fusion_proj = nn.Linear(config.model_dim * n_modalities, config.model_dim)
        else:
            self.fusion_proj = nn.Identity()
        self.state_dropout_p = state_dropout_p

    def forward(self, obs_dict: TensorDict) -> BackboneOutput:
        components = []
        rgb_features = self._encode_rgb(obs_dict)
        if rgb_features is not None:
            components.append(rgb_features)

        low_dim_features = self._encode_low_dim(obs_dict)
        if low_dim_features is not None:
            if self.training and self.state_dropout_p > 0:
                mask = (
                    torch.rand(
                        low_dim_features.shape[0], 1, 1,
                        device=low_dim_features.device,
                    )
                    > self.state_dropout_p
                ).float()
                low_dim_features = low_dim_features * mask
            components.append(low_dim_features)

        if not components:
            raise ValueError(
                "No configured observation keys were found in the observation dictionary."
            )

        if len(components) == 1:
            fused = components[0]
        else:
            fused = torch.cat(components, dim=-1)
            fused = self.fusion_proj(fused)

        fused = self.pre_norm(fused)
        encoded = self.temporal(fused)
        summary = encoded[:, -1]
        return BackboneOutput(sequence=encoded, summary=summary)
