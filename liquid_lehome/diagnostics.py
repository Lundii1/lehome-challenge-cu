"""Fast diagnostics for liquid_lehome checkpoints and short training runs."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import LeHomeConfig, load_config
from .dataset import LeHomeSequenceDataset
from .modeling import LiquidMDNPolicy, mdn_nll_loss
from .normalize import NormalizationStats, load_stats, min_max_denormalize
from .train import (
    _build_model,
    _build_model_for_state_dict,
    _get_cosine_lr,
    _make_normalize_fn,
    _schedule,
    _seed_everything,
)


@dataclass
class LoadedArtifacts:
    cfg: LeHomeConfig
    model: LiquidMDNPolicy
    device: torch.device
    dataset_root: str
    action_stats: Optional[NormalizationStats]
    state_stats: Optional[NormalizationStats]


@dataclass
class PredictionSensitivitySummary:
    num_windows: int
    compare_steps: int
    target_mse_mean: float
    target_mse_max: float
    zero_rgb_delta_mean: float
    rgb_swap_delta_mean: float
    state_swap_delta_mean: float
    rgb_state_ratio: float
    pairwise_chunk_delta_mean: float
    pairwise_chunk_delta_max: float


@dataclass
class TrainSmokeSummary:
    steps: int
    loss_start: float
    loss_end: float
    mse_start: float
    mse_end: float
    rgb_grad_mean: float
    state_grad_mean: float
    temporal_grad_mean: float
    other_grad_mean: float


def _resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _load_total_episodes(dataset_root: str) -> int:
    info_path = Path(dataset_root) / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return int(info["total_episodes"])


def _load_stats_if_available(
    dataset_root: str,
) -> Tuple[Optional[NormalizationStats], Optional[NormalizationStats]]:
    try:
        stats = load_stats(dataset_root)
    except FileNotFoundError:
        return None, None
    return stats.get("action"), stats.get("observation.state")


def _build_dataset(
    cfg: LeHomeConfig,
    dataset_root: str,
    state_stats: Optional[NormalizationStats],
    action_stats: Optional[NormalizationStats],
    episode_indices: Optional[Sequence[int]] = None,
    enable_image_augmentation: bool = False,
) -> LeHomeSequenceDataset:
    return LeHomeSequenceDataset(
        dataset_root=dataset_root,
        obs_horizon=cfg.observation_horizon,
        pred_horizon=cfg.prediction_horizon,
        rgb_keys=cfg.rgb_keys,
        low_dim_keys=cfg.low_dim_keys,
        rgb_image_size=cfg.rgb_image_size,
        state_normalize_fn=_make_normalize_fn(state_stats)
        if cfg.normalize_state and state_stats is not None
        else None,
        action_normalize_fn=_make_normalize_fn(action_stats)
        if cfg.normalize_actions and action_stats is not None
        else None,
        episode_indices=list(episode_indices) if episode_indices is not None else None,
        enable_image_augmentation=enable_image_augmentation,
    )


def _load_artifacts_from_checkpoint(
    checkpoint_path: str,
    device: str,
    config_path: Optional[str] = None,
    dataset_root: Optional[str] = None,
) -> LoadedArtifacts:
    resolved_device = _resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    if config_path is not None:
        cfg = load_config(config_path)
    else:
        cfg = LeHomeConfig(**checkpoint["config"])

    root = dataset_root or cfg.dataset_root
    action_stats, state_stats = _load_stats_if_available(root)
    model = _build_model_for_state_dict(cfg, checkpoint["model_state_dict"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()

    return LoadedArtifacts(
        cfg=cfg,
        model=model,
        device=resolved_device,
        dataset_root=root,
        action_stats=action_stats if cfg.normalize_actions else None,
        state_stats=state_stats if cfg.normalize_state else None,
    )


def _load_artifacts_from_config(
    config_path: str,
    device: str,
    dataset_root: Optional[str] = None,
) -> LoadedArtifacts:
    cfg = load_config(config_path)
    overrides = {}
    if dataset_root is not None:
        overrides["dataset_root"] = dataset_root
    if device is not None:
        overrides["device"] = device
    if overrides:
        cfg = LeHomeConfig(**{**cfg.__dict__, **overrides})

    resolved_device = _resolve_device(cfg.device)
    action_stats, state_stats = _load_stats_if_available(cfg.dataset_root)
    model = _build_model(cfg)
    model.to(resolved_device)

    return LoadedArtifacts(
        cfg=cfg,
        model=model,
        device=resolved_device,
        dataset_root=cfg.dataset_root,
        action_stats=action_stats if cfg.normalize_actions else None,
        state_stats=state_stats if cfg.normalize_state else None,
    )


def _denormalize_actions(
    actions: torch.Tensor,
    cfg: LeHomeConfig,
    action_stats: Optional[NormalizationStats],
) -> torch.Tensor:
    if cfg.normalize_actions and action_stats is not None:
        return min_max_denormalize(actions, action_stats)
    return actions


def _predict_chunk(artifacts: LoadedArtifacts, obs_dict: Dict[str, torch.Tensor]) -> np.ndarray:
    batched_obs = {key: value.unsqueeze(0).to(artifacts.device) for key, value in obs_dict.items()}
    with torch.inference_mode():
        pred = artifacts.model.rollout_actions(
            obs_dict=batched_obs,
            mode=artifacts.cfg.sample_selection_mode,
            sample_k=artifacts.cfg.sample_selection_k,
        )[0].detach().cpu()
    pred = _denormalize_actions(pred, artifacts.cfg, artifacts.action_stats)
    return pred.numpy()


def _clone_obs_dict(obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in obs_dict.items()}


def _zero_rgb(
    obs_dict: Dict[str, torch.Tensor],
    rgb_keys: Sequence[str],
) -> Dict[str, torch.Tensor]:
    modified = _clone_obs_dict(obs_dict)
    for key in rgb_keys:
        if key in modified:
            modified[key] = torch.zeros_like(modified[key])
    return modified


def _swap_modalities(
    base_obs: Dict[str, torch.Tensor],
    donor_obs: Dict[str, torch.Tensor],
    rgb_keys: Sequence[str],
    low_dim_keys: Sequence[str],
    swap_rgb: bool = False,
    swap_state: bool = False,
) -> Dict[str, torch.Tensor]:
    modified = _clone_obs_dict(base_obs)
    if swap_rgb:
        for key in rgb_keys:
            if key in donor_obs:
                modified[key] = donor_obs[key].clone()
    if swap_state:
        for key in low_dim_keys:
            if key in donor_obs:
                modified[key] = donor_obs[key].clone()
    return modified


def _window_label(dataset: LeHomeSequenceDataset, idx: int) -> str:
    ep_i, offset = dataset._windows[idx]
    episode = dataset._episodes[ep_i]
    return f"episode={episode.episode_index} offset={offset}"


def _select_window_indices(
    dataset: LeHomeSequenceDataset,
    num_windows: int,
    seed: int,
) -> List[int]:
    indices = [start for start, _ in dataset.episode_window_ranges]
    if len(indices) >= num_windows:
        return indices[:num_windows]

    rng = random.Random(seed)
    seen = set(indices)
    remaining = [idx for idx in range(len(dataset)) if idx not in seen]
    rng.shuffle(remaining)
    indices.extend(remaining[: max(0, num_windows - len(indices))])
    return indices


def _chunk_mean_abs_delta(chunk_a: np.ndarray, chunk_b: np.ndarray, compare_steps: int) -> float:
    steps = min(compare_steps, chunk_a.shape[0], chunk_b.shape[0])
    return float(np.mean(np.abs(chunk_a[:steps] - chunk_b[:steps])))


def _chunk_mse(chunk_a: np.ndarray, chunk_b: np.ndarray, compare_steps: int) -> float:
    steps = min(compare_steps, chunk_a.shape[0], chunk_b.shape[0])
    return float(np.mean((chunk_a[:steps] - chunk_b[:steps]) ** 2))


def _pairwise_chunk_deltas(chunks: Sequence[np.ndarray], compare_steps: int) -> List[float]:
    deltas: List[float] = []
    for left_idx in range(len(chunks)):
        for right_idx in range(left_idx + 1, len(chunks)):
            deltas.append(_chunk_mean_abs_delta(chunks[left_idx], chunks[right_idx], compare_steps))
    return deltas


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _run_prediction_sensitivity_suite(
    artifacts: LoadedArtifacts,
    num_windows: int,
    compare_steps: int,
    seed: int,
    episode_indices: Optional[Sequence[int]] = None,
    title: str = "Prediction Sensitivity",
) -> PredictionSensitivitySummary:
    dataset = _build_dataset(
        artifacts.cfg,
        artifacts.dataset_root,
        artifacts.state_stats,
        artifacts.action_stats,
        episode_indices=episode_indices,
        enable_image_augmentation=False,
    )
    try:
        window_indices = _select_window_indices(dataset, num_windows, seed)
        if len(window_indices) < 2:
            raise ValueError("Need at least two dataset windows to run modality-swap diagnostics.")

        compare_steps = max(1, compare_steps)
        base_chunks: List[np.ndarray] = []
        target_mses: List[float] = []
        zero_rgb_deltas: List[float] = []
        rgb_swap_deltas: List[float] = []
        state_swap_deltas: List[float] = []

        print(f"\n=== {title} ===")
        print(
            "windows="
            f"{len(window_indices)} compare_steps={compare_steps} mode={artifacts.cfg.sample_selection_mode}"
        )

        for idx, base_idx in enumerate(window_indices):
            donor_idx = window_indices[(idx + 1) % len(window_indices)]
            base_obs, base_actions = dataset[base_idx]
            donor_obs, _ = dataset[donor_idx]

            base_pred = _predict_chunk(artifacts, base_obs)
            zero_rgb_pred = _predict_chunk(
                artifacts,
                _zero_rgb(base_obs, artifacts.cfg.rgb_keys),
            )
            donor_rgb_pred = _predict_chunk(
                artifacts,
                _swap_modalities(
                    base_obs,
                    donor_obs,
                    artifacts.cfg.rgb_keys,
                    artifacts.cfg.low_dim_keys,
                    swap_rgb=True,
                    swap_state=False,
                ),
            )
            donor_state_pred = _predict_chunk(
                artifacts,
                _swap_modalities(
                    base_obs,
                    donor_obs,
                    artifacts.cfg.rgb_keys,
                    artifacts.cfg.low_dim_keys,
                    swap_rgb=False,
                    swap_state=True,
                ),
            )

            target = _denormalize_actions(base_actions, artifacts.cfg, artifacts.action_stats).numpy()
            base_chunks.append(base_pred)
            target_mses.append(_chunk_mse(base_pred, target, compare_steps))
            zero_rgb_deltas.append(_chunk_mean_abs_delta(base_pred, zero_rgb_pred, compare_steps))
            rgb_swap_deltas.append(_chunk_mean_abs_delta(base_pred, donor_rgb_pred, compare_steps))
            state_swap_deltas.append(_chunk_mean_abs_delta(base_pred, donor_state_pred, compare_steps))

            if idx < 3:
                print(
                    f"example[{idx}] base={_window_label(dataset, base_idx)} "
                    f"donor={_window_label(dataset, donor_idx)}"
                )
                print(
                    "  target_mse="
                    f"{_format_float(target_mses[-1])} zero_rgb_delta={_format_float(zero_rgb_deltas[-1])} "
                    f"rgb_swap_delta={_format_float(rgb_swap_deltas[-1])} "
                    f"state_swap_delta={_format_float(state_swap_deltas[-1])}"
                )

        pairwise_deltas = _pairwise_chunk_deltas(base_chunks, compare_steps)
        rgb_state_ratio = float(np.mean(rgb_swap_deltas) / max(np.mean(state_swap_deltas), 1e-8))

        summary = PredictionSensitivitySummary(
            num_windows=len(window_indices),
            compare_steps=compare_steps,
            target_mse_mean=float(np.mean(target_mses)),
            target_mse_max=float(np.max(target_mses)),
            zero_rgb_delta_mean=float(np.mean(zero_rgb_deltas)),
            rgb_swap_delta_mean=float(np.mean(rgb_swap_deltas)),
            state_swap_delta_mean=float(np.mean(state_swap_deltas)),
            rgb_state_ratio=rgb_state_ratio,
            pairwise_chunk_delta_mean=float(np.mean(pairwise_deltas)) if pairwise_deltas else 0.0,
            pairwise_chunk_delta_max=float(np.max(pairwise_deltas)) if pairwise_deltas else 0.0,
        )

        print("summary:")
        print(
            "  target_mse mean/max="
            f"{_format_float(summary.target_mse_mean)} / {_format_float(summary.target_mse_max)}"
        )
        print(f"  zero_rgb_delta mean={_format_float(summary.zero_rgb_delta_mean)}")
        print(f"  rgb_swap_delta mean={_format_float(summary.rgb_swap_delta_mean)}")
        print(f"  state_swap_delta mean={_format_float(summary.state_swap_delta_mean)}")
        print(f"  rgb_state_ratio={_format_float(summary.rgb_state_ratio)}")
        print(
            "  pairwise_chunk_delta mean/max="
            f"{_format_float(summary.pairwise_chunk_delta_mean)} / "
            f"{_format_float(summary.pairwise_chunk_delta_max)}"
        )

        if summary.rgb_state_ratio < 0.20:
            print("  warning: RGB perturbations are much weaker than state perturbations.")
        if summary.pairwise_chunk_delta_mean < 0.02:
            print("  warning: predicted chunks are very similar across different windows.")

        return summary
    finally:
        dataset.close()


def _bucket_gradients(model: LiquidMDNPolicy) -> Tuple[Dict[str, float], List[Tuple[float, str]]]:
    bucket_sq_norms: Dict[str, float] = defaultdict(float)
    top_params: List[Tuple[float, str]] = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_norm = float(param.grad.norm().item())
        lname = name.lower()
        if any(token in lname for token in ("rgb", "image", "clip", "vision", "cnn")):
            bucket = "rgb"
        elif any(token in lname for token in ("low_dim", "state")):
            bucket = "state"
        elif any(token in lname for token in ("temporal", "fusion", "pre_norm")):
            bucket = "temporal"
        else:
            bucket = "other"
        bucket_sq_norms[bucket] += grad_norm * grad_norm
        top_params.append((grad_norm, name))

    bucket_norms = {key: math.sqrt(value) for key, value in bucket_sq_norms.items()}
    top_params.sort(reverse=True)
    return bucket_norms, top_params[:5]


def _sample_episode_subset(dataset_root: str, num_episodes: int, seed: int) -> List[int]:
    total_episodes = _load_total_episodes(dataset_root)
    indices = list(range(total_episodes))
    rng = random.Random(seed)
    rng.shuffle(indices)
    selected = sorted(indices[: min(num_episodes, total_episodes)])
    if not selected:
        raise ValueError(f"No episodes were found in {dataset_root}")
    return selected


def _run_train_smoke(args: argparse.Namespace) -> TrainSmokeSummary:
    if args.config is None:
        raise ValueError("--config is required when --train_smoke_steps is set.")

    artifacts = _load_artifacts_from_config(
        config_path=args.config,
        device=args.device,
        dataset_root=args.dataset_root,
    )
    _seed_everything(args.seed)

    selected_episodes = _sample_episode_subset(
        artifacts.dataset_root,
        args.train_smoke_episodes,
        args.seed,
    )
    train_dataset = _build_dataset(
        artifacts.cfg,
        artifacts.dataset_root,
        artifacts.state_stats,
        artifacts.action_stats,
        episode_indices=selected_episodes,
        enable_image_augmentation=(
            artifacts.cfg.enable_image_augmentation and not args.train_smoke_disable_image_augmentation
        ),
    )
    try:
        batch_size = args.train_smoke_batch_size or min(artifacts.cfg.batch_size, max(1, len(train_dataset)))
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(train_dataset) >= batch_size,
            num_workers=0,
        )

        model = artifacts.model
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=artifacts.cfg.learning_rate,
            weight_decay=artifacts.cfg.weight_decay,
        )

        tf_ratio, free_weight = _schedule(0, artifacts.cfg.num_epochs, artifacts.cfg)
        losses: List[float] = []
        mses: List[float] = []
        grad_history: List[Dict[str, float]] = []
        loader_iter = iter(train_loader)
        total_steps = max(1, args.train_smoke_steps)

        print("\n=== Train Smoke ===")
        print(
            f"episodes={selected_episodes} steps={total_steps} batch_size={batch_size} "
            f"augmentation={'off' if args.train_smoke_disable_image_augmentation else artifacts.cfg.enable_image_augmentation}"
        )

        for step in range(1, total_steps + 1):
            try:
                obs_dict, actions = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                obs_dict, actions = next(loader_iter)

            obs_dict = {key: value.to(artifacts.device) for key, value in obs_dict.items()}
            actions = actions.to(artifacts.device)

            teacher_out = model(obs_dict, action_target=actions, teacher_forcing_ratio=tf_ratio)
            teacher_loss = mdn_nll_loss(
                teacher_out["logits"],
                teacher_out["mu"],
                teacher_out["log_sigma"],
                actions,
            )

            free_out = model(obs_dict, action_target=None, teacher_forcing_ratio=0.0)
            free_loss = mdn_nll_loss(
                free_out["logits"],
                free_out["mu"],
                free_out["log_sigma"],
                actions,
            )
            total_loss = (1.0 - free_weight) * teacher_loss + free_weight * free_loss

            optimizer.zero_grad()
            total_loss.backward()
            bucket_norms, top_params = _bucket_gradients(model)
            grad_history.append(bucket_norms)

            torch.nn.utils.clip_grad_norm_(model.parameters(), artifacts.cfg.max_grad_norm)
            warmup_frac = artifacts.cfg.warmup_epochs / max(1, artifacts.cfg.num_epochs)
            warmup_steps = int(warmup_frac * total_steps)
            lr = _get_cosine_lr(
                step - 1,
                min(warmup_steps, total_steps),
                total_steps,
                artifacts.cfg.learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()

            mse = F.mse_loss(free_out["pred_actions"], actions).item()
            losses.append(float(total_loss.item()))
            mses.append(float(mse))

            if step == 1 or step == total_steps or step % args.train_smoke_log_every == 0:
                print(
                    f"  step {step}/{total_steps} loss={total_loss.item():.4f} "
                    f"mse={mse:.6f} lr={lr:.2e} "
                    f"grad[rgb]={bucket_norms.get('rgb', 0.0):.4e} "
                    f"grad[state]={bucket_norms.get('state', 0.0):.4e} "
                    f"grad[temporal]={bucket_norms.get('temporal', 0.0):.4e}"
                )
                if top_params:
                    formatted = ", ".join(
                        f"{name}={norm:.3e}"
                        for norm, name in top_params[:3]
                    )
                    print(f"    top_grads: {formatted}")

        model.eval()
        summary = TrainSmokeSummary(
            steps=total_steps,
            loss_start=losses[0],
            loss_end=losses[-1],
            mse_start=mses[0],
            mse_end=mses[-1],
            rgb_grad_mean=float(np.mean([entry.get("rgb", 0.0) for entry in grad_history])),
            state_grad_mean=float(np.mean([entry.get("state", 0.0) for entry in grad_history])),
            temporal_grad_mean=float(np.mean([entry.get("temporal", 0.0) for entry in grad_history])),
            other_grad_mean=float(np.mean([entry.get("other", 0.0) for entry in grad_history])),
        )

        print("summary:")
        print(
            f"  loss start/end={summary.loss_start:.4f} / {summary.loss_end:.4f} "
            f"mse start/end={summary.mse_start:.6f} / {summary.mse_end:.6f}"
        )
        print(
            "  mean_grad_norms "
            f"rgb={summary.rgb_grad_mean:.4e} "
            f"state={summary.state_grad_mean:.4e} "
            f"temporal={summary.temporal_grad_mean:.4e} "
            f"other={summary.other_grad_mean:.4e}"
        )

        _run_prediction_sensitivity_suite(
            artifacts,
            num_windows=args.num_windows,
            compare_steps=args.compare_steps,
            seed=args.seed,
            episode_indices=selected_episodes,
            title="Post-Train-Smoke Sensitivity",
        )
        return summary
    finally:
        train_dataset.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fast diagnostics for liquid_lehome checkpoints and short training runs"
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint for offline diagnostics")
    parser.add_argument("--config", type=str, default=None, help="Config path for train-smoke runs or checkpoint override")
    parser.add_argument("--dataset_root", type=str, default=None, help="Optional dataset root override")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_windows", type=int, default=8, help="Number of dataset windows for checkpoint diagnostics")
    parser.add_argument("--compare_steps", type=int, default=3, help="How many predicted steps to compare")
    parser.add_argument("--train_smoke_steps", type=int, default=0, help="Run a short training smoke for this many optimizer steps")
    parser.add_argument("--train_smoke_episodes", type=int, default=4, help="How many episodes to use for train smoke")
    parser.add_argument("--train_smoke_batch_size", type=int, default=None, help="Optional batch size override for train smoke")
    parser.add_argument("--train_smoke_log_every", type=int, default=25, help="Train smoke logging cadence")
    parser.add_argument(
        "--train_smoke_disable_image_augmentation",
        action="store_true",
        help="Disable image augmentation during the train smoke even if the config enables it",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.checkpoint is None and args.train_smoke_steps <= 0:
        raise SystemExit("Specify --checkpoint and/or --train_smoke_steps to run diagnostics.")

    if args.checkpoint is not None:
        checkpoint_artifacts = _load_artifacts_from_checkpoint(
            checkpoint_path=args.checkpoint,
            device=args.device,
            config_path=args.config,
            dataset_root=args.dataset_root,
        )
        _run_prediction_sensitivity_suite(
            checkpoint_artifacts,
            num_windows=args.num_windows,
            compare_steps=args.compare_steps,
            seed=args.seed,
            title="Checkpoint Sensitivity",
        )

    if args.train_smoke_steps > 0:
        _run_train_smoke(args)


if __name__ == "__main__":
    main()
