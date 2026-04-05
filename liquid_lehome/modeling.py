"""Self-contained modeling components for liquid_lehome."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import CLIPVisionModel
except ImportError:  # pragma: no cover - optional dependency
    CLIPVisionModel = None


@dataclass(frozen=True)
class SharedBackboneConfig:
    rgb_keys: Sequence[str]
    low_dim_keys: Sequence[str]
    model_dim: int
    temporal_num_layers: int
    temporal_num_heads: int
    temporal_mlp_ratio: int
    dropout: float
    rgb_encoder_kind: str = "tiny_cnn"
    freeze_rgb_encoder: bool = False
    clip_model_name: str = "openai/clip-vit-base-patch32"
    rgb_image_size: int = 96


@dataclass(frozen=True)
class LiquidPolicyConfig:
    action_dim: int
    observation_horizon: int
    action_horizon: int
    prediction_horizon: int
    num_mixtures: int
    liquid_hidden_dim: int
    decoder_hidden_dim: int
    action_embed_dim: int
    use_cfc: bool = True
    sample_selection_mode: str = "mean"
    sample_selection_k: int = 10


@dataclass
class BackboneOutput:
    sequence: torch.Tensor
    summary: torch.Tensor


class TinyVisualEncoder(nn.Module):
    """Simple 3-layer CNN encoder used by the original LeHome baseline."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(128, model_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if images.ndim == 4:
            images = images.unsqueeze(0)
            squeeze_batch = True
        if images.ndim != 5:
            raise ValueError(f"Expected image tensor of shape (B, T, C, H, W) or (T, C, H, W), got {tuple(images.shape)}")

        batch, horizon, channels, height, width = images.shape
        flat = images.reshape(batch * horizon, channels, height, width)
        encoded = self.net(flat)
        encoded = self.pool(encoded).flatten(1)
        encoded = self.proj(encoded)
        encoded = encoded.view(batch, horizon, -1)
        if squeeze_batch:
            encoded = encoded.squeeze(0)
        return encoded


class ClipVisualEncoder(nn.Module):
    """CLIP vision encoder wrapper for frozen visual features."""

    def __init__(self, model_name: str, model_dim: int):
        super().__init__()
        if CLIPVisionModel is None:
            raise ImportError(
                "rgb_encoder_kind='clip' requires the transformers package to be installed."
            )
        self.encoder = CLIPVisionModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden_size, model_dim)
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if images.ndim == 4:
            images = images.unsqueeze(0)
            squeeze_batch = True
        if images.ndim != 5:
            raise ValueError(f"Expected image tensor of shape (B, T, C, H, W) or (T, C, H, W), got {tuple(images.shape)}")

        batch, horizon, channels, height, width = images.shape
        flat = images.reshape(batch * horizon, channels, height, width)
        if height != 224 or width != 224:
            flat = F.interpolate(flat, size=(224, 224), mode="bilinear", align_corners=False)
        flat = (flat - self.pixel_mean.view(1, 3, 1, 1)) / self.pixel_std.view(1, 3, 1, 1)
        hidden = self.encoder(pixel_values=flat).pooler_output
        encoded = self.proj(hidden).view(batch, horizon, -1)
        if squeeze_batch:
            encoded = encoded.squeeze(0)
        return encoded


class SharedObsBackbone(nn.Module):
    """Legacy shared backbone with equal-weight mean fusion."""

    def __init__(
        self,
        obs_shapes: Mapping[str, Sequence[int]],
        config: SharedBackboneConfig,
    ) -> None:
        super().__init__()
        self.obs_shapes = dict(obs_shapes)
        self.config = config
        self.rgb_keys = tuple(key for key in config.rgb_keys if key in obs_shapes)
        self.low_dim_keys = tuple(key for key in config.low_dim_keys if key in obs_shapes)

        if self.rgb_keys:
            if config.rgb_encoder_kind == "tiny_cnn":
                self.visual_encoder: Optional[nn.Module] = TinyVisualEncoder(config.model_dim)
            elif config.rgb_encoder_kind == "clip":
                self.visual_encoder = ClipVisualEncoder(config.clip_model_name, config.model_dim)
            else:
                raise NotImplementedError(
                    f"Unsupported rgb_encoder_kind: {config.rgb_encoder_kind}"
                )
            if config.freeze_rgb_encoder:
                for parameter in self.visual_encoder.parameters():
                    parameter.requires_grad = False
        else:
            self.visual_encoder = None

        if self.low_dim_keys:
            first_shape = self.obs_shapes[self.low_dim_keys[0]]
            input_dim = 1
            for dim in first_shape:
                input_dim *= int(dim)
            self.low_dim_proj: Optional[nn.Module] = nn.Linear(input_dim, config.model_dim)
        else:
            self.low_dim_proj = None

        self.pre_norm = nn.LayerNorm(config.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.temporal_num_heads,
            dim_feedforward=config.model_dim * config.temporal_mlp_ratio,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.temporal_num_layers,
        )

    def _encode_rgb(self, obs_dict: Mapping[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if self.visual_encoder is None:
            return None

        components = []
        for key in self.rgb_keys:
            if key not in obs_dict:
                continue
            rgb = obs_dict[key]
            if rgb.ndim == 4:
                rgb = rgb.unsqueeze(0)
            components.append(self.visual_encoder(rgb))

        if not components:
            return None
        return torch.stack(components, dim=0).mean(dim=0)

    def _encode_low_dim(self, obs_dict: Mapping[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if self.low_dim_proj is None:
            return None

        components = []
        for key in self.low_dim_keys:
            if key not in obs_dict:
                continue
            value = obs_dict[key]
            if value.ndim == 2:
                value = value.unsqueeze(0)
            if value.ndim < 3:
                raise ValueError(f"Expected low-dim tensor with shape (B, T, D) or (T, D), got {tuple(value.shape)}")
            flat = value.reshape(value.shape[0], value.shape[1], -1)
            components.append(self.low_dim_proj(flat))

        if not components:
            return None
        return torch.stack(components, dim=0).mean(dim=0)

    def forward(self, obs_dict: Mapping[str, torch.Tensor]) -> BackboneOutput:
        components = []
        rgb_features = self._encode_rgb(obs_dict)
        if rgb_features is not None:
            components.append(rgb_features)
        low_dim_features = self._encode_low_dim(obs_dict)
        if low_dim_features is not None:
            components.append(low_dim_features)

        if not components:
            raise ValueError("No configured observation keys were present in obs_dict.")

        if len(components) == 1:
            fused = components[0]
        else:
            fused = torch.stack(components, dim=0).mean(dim=0)

        fused = self.pre_norm(fused)
        encoded = self.temporal(fused)
        summary = encoded[:, -1]
        return BackboneOutput(sequence=encoded, summary=summary)


class ObservationCore(nn.Module):
    """Temporal core used by the LeHome policy head.

    The original checkpoint currently stored in this repo uses a 2-layer GRU
    fallback rather than a CfC layer, so the self-contained version keeps that
    implementation for checkpoint compatibility.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.core = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, hidden = self.core(sequence)
        return outputs, hidden[-1]


class MixtureDensityHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, num_mixtures: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.num_mixtures = num_mixtures
        self.proj = nn.Linear(hidden_dim, num_mixtures * (1 + 2 * action_dim))

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.proj(hidden)
        logits = raw[..., : self.num_mixtures]
        params = raw[..., self.num_mixtures :].reshape(
            *raw.shape[:-1],
            self.num_mixtures,
            2 * self.action_dim,
        )
        mu = params[..., : self.action_dim]
        log_sigma = params[..., self.action_dim :]
        log_sigma = torch.clamp(log_sigma, min=-7.0, max=2.0)
        return logits, mu, log_sigma


def mdn_mean_actions(logits: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
    return torch.sum(weights * mu, dim=-2)


def mdn_log_prob(
    logits: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    target = actions.unsqueeze(-2)
    inv_sigma = torch.exp(-log_sigma)
    log_component = -0.5 * (
        ((target - mu) * inv_sigma) ** 2 + 2.0 * log_sigma + math.log(2.0 * math.pi)
    ).sum(dim=-1)
    log_mixture = F.log_softmax(logits, dim=-1) + log_component
    return torch.logsumexp(log_mixture, dim=-1)


def mdn_nll_loss(
    logits: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return (-mdn_log_prob(logits, mu, log_sigma, targets)).mean()


def mdn_sample_actions(
    logits: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mixture = torch.distributions.Categorical(logits=logits)
    component_idx = mixture.sample()
    gather_idx = component_idx.unsqueeze(-1).unsqueeze(-1).expand(
        *component_idx.shape,
        1,
        mu.shape[-1],
    )
    chosen_mu = torch.gather(mu, dim=-2, index=gather_idx).squeeze(-2)
    chosen_log_sigma = torch.gather(log_sigma, dim=-2, index=gather_idx).squeeze(-2)
    sampled = chosen_mu + torch.randn_like(chosen_mu) * torch.exp(chosen_log_sigma)
    log_prob = mdn_log_prob(logits, mu, log_sigma, sampled)
    return sampled, log_prob


class LiquidMDNPolicy(nn.Module):
    """Autoregressive MDN policy head with a recurrent observation core."""

    def __init__(self, backbone: nn.Module, config: LiquidPolicyConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.observation_core = ObservationCore(
            input_dim=backbone.config.model_dim,
            hidden_dim=config.liquid_hidden_dim,
        )
        self.core_to_decoder = (
            nn.Identity()
            if config.liquid_hidden_dim == config.decoder_hidden_dim
            else nn.Linear(config.liquid_hidden_dim, config.decoder_hidden_dim)
        )
        self.action_embed = nn.Linear(config.action_dim, config.action_embed_dim)
        self.decoder = nn.GRUCell(config.action_embed_dim, config.decoder_hidden_dim)
        self.mdn = MixtureDensityHead(
            hidden_dim=config.decoder_hidden_dim,
            action_dim=config.action_dim,
            num_mixtures=config.num_mixtures,
        )

    def _initial_decoder_state(self, obs_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        backbone_out = self.backbone(obs_dict)
        _, hidden = self.observation_core(backbone_out.sequence)
        hidden = self.core_to_decoder(hidden)
        return hidden

    def _decode(
        self,
        decoder_state: torch.Tensor,
        horizon: int,
        action_target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
        sample_actions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch = decoder_state.shape[0]
        prev_action = torch.zeros(
            batch,
            self.config.action_dim,
            device=decoder_state.device,
            dtype=decoder_state.dtype,
        )

        logits_list = []
        mu_list = []
        log_sigma_list = []
        action_list = []
        sample_log_probs = []

        for step in range(horizon):
            embedded = self.action_embed(prev_action)
            decoder_state = self.decoder(embedded, decoder_state)
            logits, mu, log_sigma = self.mdn(decoder_state)
            if sample_actions:
                next_action, next_log_prob = mdn_sample_actions(logits, mu, log_sigma)
                sample_log_probs.append(next_log_prob)
            else:
                next_action = mdn_mean_actions(logits, mu)

            logits_list.append(logits)
            mu_list.append(mu)
            log_sigma_list.append(log_sigma)
            action_list.append(next_action)

            if action_target is not None:
                target_action = action_target[:, step]
                if teacher_forcing_ratio >= 1.0:
                    prev_action = target_action
                elif teacher_forcing_ratio <= 0.0:
                    prev_action = next_action.detach()
                else:
                    teacher_mask = (
                        torch.rand(batch, 1, device=decoder_state.device) < teacher_forcing_ratio
                    ).to(next_action.dtype)
                    prev_action = teacher_mask * target_action + (1.0 - teacher_mask) * next_action.detach()
            else:
                prev_action = next_action.detach()

        logits_tensor = torch.stack(logits_list, dim=1)
        mu_tensor = torch.stack(mu_list, dim=1)
        log_sigma_tensor = torch.stack(log_sigma_list, dim=1)
        action_tensor = torch.stack(action_list, dim=1)
        sample_log_prob_tensor = (
            torch.stack(sample_log_probs, dim=1).sum(dim=1) if sample_log_probs else None
        )
        return logits_tensor, mu_tensor, log_sigma_tensor, action_tensor, sample_log_prob_tensor

    def forward(
        self,
        obs_dict: Mapping[str, torch.Tensor],
        action_target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> Mapping[str, torch.Tensor]:
        decoder_state = self._initial_decoder_state(obs_dict)
        horizon = action_target.shape[1] if action_target is not None else self.config.prediction_horizon
        logits, mu, log_sigma, pred_actions, _ = self._decode(
            decoder_state=decoder_state,
            horizon=horizon,
            action_target=action_target,
            teacher_forcing_ratio=teacher_forcing_ratio,
            sample_actions=False,
        )
        return {
            "logits": logits,
            "mu": mu,
            "log_sigma": log_sigma,
            "pred_actions": pred_actions,
        }

    def rollout_actions(
        self,
        obs_dict: Mapping[str, torch.Tensor],
        mode: Optional[str] = None,
        sample_k: Optional[int] = None,
    ) -> torch.Tensor:
        rollout_mode = mode or self.config.sample_selection_mode
        num_samples = int(sample_k or self.config.sample_selection_k)
        decoder_state = self._initial_decoder_state(obs_dict)
        horizon = self.config.action_horizon

        if rollout_mode == "mean":
            _, _, _, actions, _ = self._decode(
                decoder_state=decoder_state,
                horizon=horizon,
                sample_actions=False,
            )
            return actions

        if rollout_mode in {"sample", "stochastic"}:
            _, _, _, actions, _ = self._decode(
                decoder_state=decoder_state,
                horizon=horizon,
                sample_actions=True,
            )
            return actions

        if rollout_mode == "best_of_k":
            candidate_actions = []
            candidate_scores = []
            for _ in range(max(1, num_samples)):
                _, _, _, actions, score = self._decode(
                    decoder_state=decoder_state.clone(),
                    horizon=horizon,
                    sample_actions=True,
                )
                candidate_actions.append(actions)
                candidate_scores.append(score if score is not None else torch.zeros(actions.shape[0], device=actions.device))

            stacked_actions = torch.stack(candidate_actions, dim=1)
            stacked_scores = torch.stack(candidate_scores, dim=1)
            best_idx = torch.argmax(stacked_scores, dim=1)
            batch_idx = torch.arange(stacked_actions.shape[0], device=stacked_actions.device)
            return stacked_actions[batch_idx, best_idx]

        raise ValueError(f"Unsupported rollout mode: {rollout_mode}")
