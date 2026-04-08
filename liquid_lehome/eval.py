"""Evaluation entrypoint and policy class for LeHome Liquid+MDN inference."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Dict, Optional
import math

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    LeHomeConfig,
    compute_env_steps_per_policy_step,
    load_config,
)
from .dataset import LeHomeSequenceDataset
from .modeling import LiquidMDNPolicy
from .normalize import NormalizationStats, load_stats, min_max_denormalize, min_max_normalize
from .train import _build_model_for_state_dict


class LiquidLeHomePolicy:
    """Inference-time policy wrapper compatible with LeHome evaluation.

    Implements:
    * ``reset()`` — clear internal queues.
    * ``select_action(observation) -> np.ndarray`` — return a single 12-D
      action from the receding-horizon action queue.

    The observation dict should contain numpy arrays with keys matching the
    config's ``rgb_keys`` and ``low_dim_keys`` (e.g.
    ``observation.images.top_rgb`` as ``(H, W, 3)`` uint8 and
    ``observation.state`` as ``(12,)`` float32).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        config_path: Optional[str] = None,
        dataset_root: Optional[str] = None,
        env_step_hz: Optional[int] = None,
        control_hz: Optional[int] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if config_path is not None:
            cfg = load_config(config_path)
        else:
            cfg = LeHomeConfig(**ckpt["config"])
        self.cfg = cfg
        self.control_hz = int(control_hz or cfg.control_hz)
        self.env_step_hz = int(env_step_hz) if env_step_hz is not None else None
        self.env_steps_per_policy_step = (
            compute_env_steps_per_policy_step(self.control_hz, self.env_step_hz)
            if self.env_step_hz is not None
            else 1
        )

        # Build model. Older checkpoints use the legacy mean-fusion backbone,
        # while newer ones include ``backbone.fusion_proj`` from the enhanced backbone.
        self.policy = _build_model_for_state_dict(cfg, ckpt["model_state_dict"])
        self.policy.load_state_dict(ckpt["model_state_dict"])
        self.policy.to(self.device)
        self.policy.eval()

        # Normalization stats
        root = dataset_root or cfg.dataset_root
        try:
            norm_stats = load_stats(root)
            self.action_stats: Optional[NormalizationStats] = norm_stats.get("action")
            self.state_stats: Optional[NormalizationStats] = norm_stats.get("observation.state")
        except FileNotFoundError:
            self.action_stats = None
            self.state_stats = None

        # Internal queues (mirrors algo.py lines 210-241)
        self.obs_queue: Optional[Dict[str, deque]] = None
        self.action_queue: deque = deque()
        self.held_action: Optional[torch.Tensor] = None
        self.env_steps_until_next_policy_step = 0
        self._last_debug_snapshot: Optional[Dict[str, object]] = None
        self.reset()

    def reset(self) -> None:
        """Clear observation and action queues."""
        obs_horizon = self.cfg.observation_horizon
        self.obs_queue = {}
        for key in list(self.cfg.rgb_keys) + list(self.cfg.depth_keys) + list(self.cfg.low_dim_keys):
            self.obs_queue[key] = deque(maxlen=obs_horizon)
        self.action_queue = deque(maxlen=self.cfg.action_horizon)
        self.held_action = None
        self.env_steps_until_next_policy_step = 0
        self._last_debug_snapshot = None

    def get_debug_snapshot(self) -> Optional[Dict[str, object]]:
        """Return structured debug information for the last selected action."""
        if self._last_debug_snapshot is None:
            return None

        snapshot: Dict[str, object] = {}
        for key, value in self._last_debug_snapshot.items():
            if isinstance(value, np.ndarray):
                snapshot[key] = value.copy()
            elif isinstance(value, list):
                copied = []
                for item in value:
                    if isinstance(item, np.ndarray):
                        copied.append(item.copy())
                    else:
                        copied.append(item)
                snapshot[key] = copied
            else:
                snapshot[key] = value
        return snapshot

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Return a single action from the receding-horizon queue.

        Parameters
        ----------
        observation : dict
            Maps observation keys to numpy arrays.  Images should be
            ``(H, W, 3)`` uint8; state should be ``(D,)`` float32.

        Returns
        -------
        np.ndarray
            Action of shape ``(action_dim,)`` in the original (un-normalized)
            joint-position space.
        """
        current_state = observation.get("observation.state")
        current_state_np = (
            np.asarray(current_state, dtype=np.float32).copy()
            if current_state is not None
            else None
        )

        if self.env_steps_until_next_policy_step > 0 and self.held_action is not None:
            self.env_steps_until_next_policy_step -= 1
            held_action_np = self.held_action.detach().cpu().numpy().copy()
            self._last_debug_snapshot = self._make_debug_snapshot(
                state=current_state_np,
                action=held_action_np,
                action_origin="held",
                fresh_rollout_generated=False,
                rollout_preview=None,
            )
            return held_action_np

        # Pre-process and append to queues
        processed = self._preprocess(observation)
        self._append_obs(processed)

        # Predict new trajectory when action queue is exhausted
        rollout_preview: Optional[list[np.ndarray]] = None
        fresh_rollout_generated = False
        if len(self.action_queue) == 0:
            stacked_obs = self._stack_obs_queue()
            with torch.inference_mode():
                action_seq = self.policy.rollout_actions(
                    obs_dict=stacked_obs,
                    mode=self.cfg.sample_selection_mode,
                    sample_k=self.cfg.sample_selection_k,
                )  # (1, action_horizon, action_dim)
            # Denormalize
            actions = action_seq[0]  # (action_horizon, action_dim)
            if self.action_stats is not None and self.cfg.normalize_actions:
                actions = min_max_denormalize(actions, self.action_stats)
            self.action_queue.extend(actions)
            rollout_preview = [
                actions[i].detach().cpu().numpy().copy()
                for i in range(min(3, actions.shape[0]))
            ]
            fresh_rollout_generated = True

        action = self.action_queue.popleft()
        if self.env_steps_per_policy_step > 1:
            self.held_action = action.detach().clone()
            self.env_steps_until_next_policy_step = self.env_steps_per_policy_step - 1
        action_np = action.detach().cpu().numpy().copy()
        self._last_debug_snapshot = self._make_debug_snapshot(
            state=current_state_np,
            action=action_np,
            action_origin="new",
            fresh_rollout_generated=fresh_rollout_generated,
            rollout_preview=rollout_preview,
        )
        return action_np

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, observation: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert raw numpy observation to device tensors."""
        result: Dict[str, torch.Tensor] = {}
        for key in self.cfg.rgb_keys:
            if key not in observation:
                continue
            img = observation[key]
            t = torch.from_numpy(img).float()
            if t.ndim == 3 and t.shape[-1] in (1, 3):
                t = t.permute(2, 0, 1)
            if t.max() > 1.0:
                t = t / 255.0
            t = F.interpolate(
                t.unsqueeze(0),
                size=(self.cfg.rgb_image_size, self.cfg.rgb_image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            result[key] = t.to(self.device)

        for key in self.cfg.depth_keys:
            if key not in observation:
                continue
            depth = observation[key]
            t = torch.from_numpy(np.asarray(depth)).float()
            # Convert uint16 mm to float meters, then normalize to [0, 1]
            if t.max() > 100.0:  # likely uint16 in mm
                t = t / 1000.0  # mm -> meters
            t = torch.clamp(t / 2.0, 0.0, 1.0)  # normalize to [0,1] assuming max 2m depth
            if t.ndim == 2:
                t = t.unsqueeze(0)  # (H, W) -> (1, H, W)
            t = F.interpolate(
                t.unsqueeze(0),
                size=(self.cfg.depth_image_size, self.cfg.depth_image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            result[key] = t.to(self.device)

        for key in self.cfg.low_dim_keys:
            if key not in observation:
                continue
            val = torch.as_tensor(observation[key], dtype=torch.float32)
            if self.state_stats is not None and self.cfg.normalize_state:
                val = min_max_normalize(val, self.state_stats)
            result[key] = val.to(self.device)

        return result

    def _append_obs(self, obs: Dict[str, torch.Tensor]) -> None:
        """Append processed observation to the queue.

        On the first call the queue is filled with copies of the current
        observation (mirrors ``LiquidMDNAlgo._append_obs``).
        """
        if self.obs_queue is None:
            self.reset()
        first_q = next(iter(self.obs_queue.values()))
        if len(first_q) == 0:
            for _ in range(self.cfg.observation_horizon):
                for key in self.obs_queue:
                    self.obs_queue[key].append(obs[key].clone())
        else:
            for key in self.obs_queue:
                self.obs_queue[key].append(obs[key].clone())

    def _stack_obs_queue(self) -> Dict[str, torch.Tensor]:
        """Stack queued observations into a batched dict with leading (1, T, ...) shape."""
        stacked: Dict[str, torch.Tensor] = {}
        for key, q in self.obs_queue.items():
            stacked[key] = torch.stack(list(q), dim=0).unsqueeze(0)  # (1, T, ...)
        return stacked

    def _make_debug_snapshot(
        self,
        state: Optional[np.ndarray],
        action: np.ndarray,
        action_origin: str,
        fresh_rollout_generated: bool,
        rollout_preview: Optional[list[np.ndarray]],
    ) -> Dict[str, object]:
        action_minus_state = None
        if state is not None and state.shape == action.shape:
            action_minus_state = action - state

        return {
            "state": state.copy() if state is not None else None,
            "action": action.copy(),
            "action_minus_state": action_minus_state.copy() if action_minus_state is not None else None,
            "action_origin": action_origin,
            "fresh_rollout_generated": fresh_rollout_generated,
            "remaining_hold_count": int(self.env_steps_until_next_policy_step),
            "action_queue_len": int(len(self.action_queue)),
            "rollout_preview": rollout_preview,
            "sample_selection_mode": self.cfg.sample_selection_mode,
            "control_hz": int(self.control_hz),
            "env_step_hz": int(self.env_step_hz or self.control_hz),
            "env_steps_per_policy_step": int(self.env_steps_per_policy_step),
        }


def _ensure_uint8_image(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    elif frame.ndim == 3 and frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=2)
    if frame.dtype == np.uint8:
        return frame
    if np.issubdtype(frame.dtype, np.floating):
        max_value = float(np.max(frame)) if frame.size else 0.0
        if max_value <= 1.0:
            frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def _make_video_grid(frames: list[np.ndarray]) -> np.ndarray:
    frames = [_ensure_uint8_image(frame) for frame in frames]
    if not frames:
        raise ValueError("At least one RGB frame is required to write a video.")
    if len(frames) == 1:
        return frames[0]

    cols = 2 if len(frames) > 2 else len(frames)
    rows = math.ceil(len(frames) / cols)
    tile_h = max(frame.shape[0] for frame in frames)
    tile_w = max(frame.shape[1] for frame in frames)

    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        h, w = frame.shape[:2]
        y0 = row * tile_h
        x0 = col * tile_w
        canvas[y0 : y0 + h, x0 : x0 + w] = frame
    return canvas


def export_dataset_video(
    policy: LiquidLeHomePolicy,
    dataset_root: str,
    video_path: str,
    episode_index: int = 0,
    max_steps: Optional[int] = None,
    video_fps: Optional[int] = None,
) -> Dict[str, float]:
    """Run offline evaluation over one dataset episode and save a tiled RGB video."""
    dataset = LeHomeSequenceDataset(
        dataset_root=dataset_root,
        obs_horizon=policy.cfg.observation_horizon,
        pred_horizon=policy.cfg.prediction_horizon,
        rgb_keys=policy.cfg.rgb_keys,
        low_dim_keys=policy.cfg.low_dim_keys,
        rgb_image_size=policy.cfg.rgb_image_size,
        episode_indices=[episode_index],
    )
    try:
        if not dataset._episodes:
            raise ValueError(f"Episode {episode_index} was not found in {dataset_root}")

        episode = dataset._episodes[0]
        rgb_keys = [key for key in policy.cfg.rgb_keys if key in episode.video_map]
        if not rgb_keys:
            raise ValueError("Configured rgb_keys are not present in the requested episode.")

        num_steps = episode.length if max_steps is None else min(episode.length, max_steps)
        if num_steps <= 0:
            raise ValueError("No frames available for the requested video export.")

        output_path = Path(video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = video_fps or dataset._fps

        policy.reset()
        mse_sum = 0.0
        with imageio.get_writer(str(output_path), fps=fps) as writer:
            for step in range(num_steps):
                row = dataset._df.iloc[episode.data_start + step]
                observation: Dict[str, np.ndarray] = {}
                video_frames = []

                for key in rgb_keys:
                    rgb_video_path, frame_start = episode.video_map[key]
                    frame = dataset._read_video_frame(rgb_video_path, frame_start + step)
                    observation[key] = frame
                    video_frames.append(frame)

                for key in policy.cfg.low_dim_keys:
                    if key in row.index:
                        observation[key] = np.asarray(row[key], dtype=np.float32)

                missing_keys = [
                    key
                    for key in list(policy.cfg.rgb_keys) + list(policy.cfg.low_dim_keys)
                    if key not in observation
                ]
                if missing_keys:
                    raise ValueError(
                        f"Episode {episode_index} is missing required observation keys: {missing_keys}"
                    )

                action = policy.select_action(observation)
                target_action = np.asarray(row["action"], dtype=np.float32)
                mse_sum += float(np.mean((action - target_action) ** 2))

                writer.append_data(_make_video_grid(video_frames))

        return {
            "episode_index": float(episode_index),
            "frames": float(num_steps),
            "fps": float(fps),
            "mean_action_mse": mse_sum / float(num_steps),
        }
    finally:
        dataset.close()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a Liquid+MDN LeHome checkpoint")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument("--config", type=str, default=None, help="Path to JSON config (optional)")
    p.add_argument("--dataset_root", type=str, default=None, help="Dataset root for stats")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--env_step_hz",
        type=int,
        default=None,
        help="Optional simulator step rate. When set, actions are held across simulator steps to match control_hz.",
    )
    p.add_argument("--video_path", type=str, default=None, help="Optional path for offline episode video export")
    p.add_argument("--episode_index", type=int, default=0, help="Dataset episode index to export when --video_path is set")
    p.add_argument("--max_steps", type=int, default=None, help="Optional limit on exported episode steps")
    p.add_argument("--video_fps", type=int, default=None, help="Optional FPS override for exported video")
    return p


def main() -> None:
    args = build_parser().parse_args()
    policy = LiquidLeHomePolicy(
        checkpoint_path=args.checkpoint,
        device=args.device,
        config_path=args.config,
        dataset_root=args.dataset_root,
        env_step_hz=args.env_step_hz,
    )
    print(f"Loaded policy from {args.checkpoint}")
    print(f"  action_dim={policy.cfg.action_dim}, obs_horizon={policy.cfg.observation_horizon}")
    print(f"  action_horizon={policy.cfg.action_horizon}, pred_horizon={policy.cfg.prediction_horizon}")
    print(
        f"  control_hz={policy.control_hz}, env_step_hz={policy.env_step_hz or policy.control_hz}, "
        f"env_steps_per_policy_step={policy.env_steps_per_policy_step}"
    )

    if args.video_path is not None:
        dataset_root = args.dataset_root or policy.cfg.dataset_root
        stats = export_dataset_video(
            policy=policy,
            dataset_root=dataset_root,
            video_path=args.video_path,
            episode_index=args.episode_index,
            max_steps=args.max_steps,
            video_fps=args.video_fps,
        )
        print(f"  saved video: {args.video_path}")
        print(
            f"  episode={int(stats['episode_index'])} frames={int(stats['frames'])} "
            f"fps={int(stats['fps'])} mean_action_mse={stats['mean_action_mse']:.6f}"
        )
        return

    # Quick smoke check with a random observation
    fake_obs: Dict[str, np.ndarray] = {}
    for key in policy.cfg.rgb_keys:
        fake_obs[key] = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    for key in policy.cfg.low_dim_keys:
        fake_obs[key] = np.random.randn(policy.cfg.action_dim).astype(np.float32)

    policy.reset()
    action = policy.select_action(fake_obs)
    print(f"  smoke-check action shape: {action.shape}, dtype: {action.dtype}")
    print("Eval entrypoint ready.")


if __name__ == "__main__":
    main()
