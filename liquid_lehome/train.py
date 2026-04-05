"""Standalone training loop for Liquid+MDN on LeHome data."""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from liquid_robomimic.modeling import (
    LiquidMDNPolicy,
    SharedObsBackbone,
    mdn_nll_loss,
)

from .config import LeHomeConfig, config_to_backbone, config_to_policy, load_config
from .dataset import LeHomeSequenceDataset
from .normalize import NormalizationStats, load_stats, min_max_normalize


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_obs_shapes(cfg: LeHomeConfig) -> Dict[str, tuple]:
    shapes: Dict[str, tuple] = {}
    for key in cfg.rgb_keys:
        shapes[key] = (3, cfg.rgb_image_size, cfg.rgb_image_size)
    for key in cfg.low_dim_keys:
        if key == "observation.state":
            shapes[key] = (cfg.action_dim,)  # state dim == action dim for LeHome
        else:
            shapes[key] = (cfg.action_dim,)
    return shapes


def _build_model(cfg: LeHomeConfig) -> LiquidMDNPolicy:
    obs_shapes = _build_obs_shapes(cfg)
    backbone = SharedObsBackbone(obs_shapes, config_to_backbone(cfg))
    return LiquidMDNPolicy(backbone, config_to_policy(cfg))


def _get_cosine_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _schedule(epoch: int, num_epochs: int, cfg: LeHomeConfig):
    """Compute teacher-forcing ratio and free-running weight for this epoch.

    Mirrors ``LiquidMDNAlgo._schedule`` in algo.py (lines 128-136).
    """
    progress = float(max(0, epoch)) / float(max(1, num_epochs - 1))
    tf_ratio = cfg.teacher_forcing_start + progress * (cfg.teacher_forcing_end - cfg.teacher_forcing_start)
    free_weight = cfg.free_running_weight_start + progress * (cfg.free_running_weight_end - cfg.free_running_weight_start)
    return tf_ratio, free_weight


def _make_normalize_fn(stats: Optional[NormalizationStats]):
    if stats is None:
        return None

    def fn(x: torch.Tensor) -> torch.Tensor:
        return min_max_normalize(x, stats)

    return fn


class EpisodeOrderedBatchSampler(Sampler[List[int]]):
    """Shuffle episode order while keeping windows contiguous within each batch."""

    def __init__(
        self,
        episode_window_ranges: List[tuple[int, int]],
        batch_size: int,
        drop_last: bool,
        seed: int,
    ) -> None:
        self.episode_window_ranges = list(episode_window_ranges)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        episode_ranges = list(self.episode_window_ranges)
        rng.shuffle(episode_ranges)

        batch: List[int] = []
        for start, end in episode_ranges:
            for idx in range(start, end):
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        total_windows = sum(end - start for start, end in self.episode_window_ranges)
        if self.drop_last:
            return total_windows // self.batch_size
        return math.ceil(total_windows / self.batch_size)


def train(cfg: LeHomeConfig) -> None:
    _seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Normalization stats --
    norm_stats = load_stats(cfg.dataset_root)
    action_stats = norm_stats.get("action") if cfg.normalize_actions else None
    state_stats = norm_stats.get("observation.state") if cfg.normalize_state else None

    # -- Datasets --
    # Split episodes: last val_split_ratio fraction for validation
    import json
    info_path = Path(cfg.dataset_root) / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    total_episodes = int(info["total_episodes"])
    all_episodes = list(range(total_episodes))
    n_val = max(1, int(total_episodes * cfg.val_split_ratio))
    train_episodes = all_episodes[: total_episodes - n_val]
    val_episodes = all_episodes[total_episodes - n_val :]

    common_kwargs = dict(
        dataset_root=cfg.dataset_root,
        obs_horizon=cfg.observation_horizon,
        pred_horizon=cfg.prediction_horizon,
        rgb_keys=cfg.rgb_keys,
        low_dim_keys=cfg.low_dim_keys,
        rgb_image_size=cfg.rgb_image_size,
        state_normalize_fn=_make_normalize_fn(state_stats),
        action_normalize_fn=_make_normalize_fn(action_stats),
    )
    train_ds = LeHomeSequenceDataset(episode_indices=train_episodes, **common_kwargs)
    val_ds = LeHomeSequenceDataset(episode_indices=val_episodes, **common_kwargs)
    train_batch_sampler = EpisodeOrderedBatchSampler(
        episode_window_ranges=train_ds.episode_window_ranges,
        batch_size=cfg.batch_size,
        drop_last=True,
        seed=cfg.seed,
    )
    loader_kwargs = dict(
        num_workers=cfg.num_data_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_data_workers > 0,
    )

    try:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        print(f"Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")
        print(f"Train episodes: {len(train_episodes)}, Val episodes: {len(val_episodes)}")

        # -- Model --
        policy = _build_model(cfg).to(device)
        num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")

        # -- Optimizer --
        optimizer = torch.optim.AdamW(
            policy.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        total_steps = cfg.num_epochs * len(train_loader)
        global_step = 0
        best_val_loss = float("inf")

        # -- Training loop --
        for epoch in range(cfg.num_epochs):
            train_batch_sampler.set_epoch(epoch)
            policy.train()
            tf_ratio, free_weight = _schedule(epoch, cfg.num_epochs, cfg)

            epoch_loss = 0.0
            epoch_steps = 0
            t0 = time.time()

            for obs_dict, actions in train_loader:
                # Move to device
                obs_dict = {k: v.to(device) for k, v in obs_dict.items()}
                actions = actions.to(device)

                # Teacher-forced pass
                teacher_out = policy(obs_dict, action_target=actions, teacher_forcing_ratio=tf_ratio)
                teacher_loss = mdn_nll_loss(
                    teacher_out["logits"], teacher_out["mu"], teacher_out["log_sigma"], actions
                )

                # Free-running pass
                free_out = policy(obs_dict, action_target=None, teacher_forcing_ratio=0.0)
                free_loss = mdn_nll_loss(
                    free_out["logits"], free_out["mu"], free_out["log_sigma"], actions
                )

                total_loss = (1.0 - free_weight) * teacher_loss + free_weight * free_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)

                # Cosine LR
                lr = _get_cosine_lr(global_step, cfg.warmup_steps, total_steps, cfg.learning_rate)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_steps += 1
                global_step += 1

                if global_step % cfg.log_every_n_steps == 0:
                    mse = F.mse_loss(free_out["pred_actions"], actions).item()
                    print(
                        f"  step {global_step}/{total_steps} | "
                        f"loss={total_loss.item():.4f} tf_nll={teacher_loss.item():.4f} "
                        f"fr_nll={free_loss.item():.4f} mse={mse:.6f} "
                        f"tf={tf_ratio:.3f} fw={free_weight:.3f} lr={lr:.2e}"
                    )

            elapsed = time.time() - t0
            avg_loss = epoch_loss / max(1, epoch_steps)

            # -- Validation --
            policy.eval()
            val_loss_sum = 0.0
            val_mse_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for obs_dict, actions in val_loader:
                    obs_dict = {k: v.to(device) for k, v in obs_dict.items()}
                    actions = actions.to(device)
                    free_out = policy(obs_dict, action_target=None, teacher_forcing_ratio=0.0)
                    loss = mdn_nll_loss(
                        free_out["logits"], free_out["mu"], free_out["log_sigma"], actions
                    )
                    mse = F.mse_loss(free_out["pred_actions"], actions)
                    val_loss_sum += loss.item()
                    val_mse_sum += mse.item()
                    val_steps += 1

            val_loss = val_loss_sum / max(1, val_steps)
            val_mse = val_mse_sum / max(1, val_steps)

            print(
                f"Epoch {epoch + 1}/{cfg.num_epochs} | "
                f"train_loss={avg_loss:.4f} val_loss={val_loss:.4f} val_mse={val_mse:.6f} "
                f"time={elapsed:.1f}s"
            )

            # -- Checkpointing --
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg.__dict__,
                "val_loss": val_loss,
            }

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt, output_dir / "best.pt")
                print(f"  -> Saved best checkpoint (val_loss={val_loss:.4f})")

            if (epoch + 1) % cfg.save_every_n_epochs == 0:
                torch.save(ckpt, output_dir / f"epoch_{epoch + 1:04d}.pt")

        # Save final checkpoint
        torch.save(ckpt, output_dir / "last.pt")
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    finally:
        train_ds.close()
        val_ds.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Liquid+MDN on LeHome data")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--dataset_root", type=str, default=None, help="Override dataset_root")
    parser.add_argument(
        "--num_data_workers", type=int, default=None, help="Override num_data_workers"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    # Apply CLI overrides by constructing a new config
    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.epochs is not None:
        overrides["num_epochs"] = args.epochs
    if args.dataset_root is not None:
        overrides["dataset_root"] = args.dataset_root
    if args.num_data_workers is not None:
        overrides["num_data_workers"] = args.num_data_workers

    if overrides:
        cfg = LeHomeConfig(**{**cfg.__dict__, **overrides})

    train(cfg)


if __name__ == "__main__":
    main()
