"""LeHome dataset adapter with sliding-window batching for Liquid+MDN training.

Reads LeRobot-format datasets directly using pandas (parquet) and imageio
(video decoding), with no dependency on the ``lerobot`` library.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class _EpisodeInfo:
    episode_index: int
    length: int
    data_start: int  # global index of the first frame in parquet
    # Per rgb_key: (video_path, first_video_frame_index)
    video_map: Dict[str, Tuple[str, int]]


class LeHomeSequenceDataset(Dataset):
    """Reads a LeRobot-format dataset and produces observation/action windows.

    Each item is a tuple ``(obs_dict, actions)`` where:

    * ``obs_dict`` maps observation keys to tensors with a leading time axis
      of length ``obs_horizon``.
    * ``actions`` is a ``(pred_horizon, action_dim)`` tensor.

    Images are resized to ``(rgb_image_size, rgb_image_size)`` and scaled to
    ``[0, 1]``. State and actions can optionally be normalized via a
    caller-supplied function.
    """

    def __init__(
        self,
        dataset_root: str,
        obs_horizon: int,
        pred_horizon: int,
        rgb_keys: Sequence[str],
        low_dim_keys: Sequence[str],
        rgb_image_size: int = 96,
        state_normalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        action_normalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        episode_indices: Optional[List[int]] = None,
        enable_image_augmentation: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.root = Path(dataset_root)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.rgb_keys = list(rgb_keys)
        self.low_dim_keys = list(low_dim_keys)
        self.rgb_image_size = rgb_image_size
        self.state_normalize_fn = state_normalize_fn
        self.action_normalize_fn = action_normalize_fn
        self.enable_image_augmentation = enable_image_augmentation

        # Load metadata
        with open(self.root / "meta" / "info.json", encoding="utf-8") as f:
            self._info = json.load(f)
        self._fps = int(self._info["fps"])

        # Load all parquet data chunks
        self._df = self._load_parquet()

        # Build episode info from metadata
        self._episodes: List[_EpisodeInfo] = []
        self._build_episode_info(episode_indices)

        # Build sliding-window index: list of (episode_info_idx, offset_within_episode)
        self._windows: List[Tuple[int, int]] = []
        self._episode_window_ranges: List[Tuple[int, int]] = []
        self._build_window_index()

        # Video reader cache - lazily initialized per-worker so each worker
        # owns its own ffmpeg subprocesses.
        self._reader_cache: Dict[str, imageio.core.format.Format.Reader] = {}
        self._cache_pid: Optional[int] = None

    def _load_parquet(self) -> pd.DataFrame:
        """Read and concatenate all parquet chunks."""
        data_dir = self.root / "data"
        chunks = sorted(data_dir.rglob("*.parquet"))
        if not chunks:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")
        dfs = [pd.read_parquet(p) for p in chunks]
        df = pd.concat(dfs, ignore_index=True)
        return df.sort_values("index").reset_index(drop=True)

    def _build_episode_info(self, episode_indices: Optional[List[int]]) -> None:
        """Parse episode metadata to build per-episode video mappings."""
        ep_meta_dir = self.root / "meta" / "episodes"
        ep_files = sorted(ep_meta_dir.rglob("*.parquet"))
        if not ep_files:
            raise FileNotFoundError(f"No episode metadata found in {ep_meta_dir}")
        ep_df = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True)
        ep_df = ep_df.sort_values("episode_index").reset_index(drop=True)

        allowed = set(episode_indices) if episode_indices is not None else None

        for _, row in ep_df.iterrows():
            ep_idx = int(row["episode_index"])
            if allowed is not None and ep_idx not in allowed:
                continue

            length = int(row["length"])
            data_start = int(row["dataset_from_index"])

            video_map: Dict[str, Tuple[str, int]] = {}
            for rgb_key in self.rgb_keys:
                chunk_col = f"videos/{rgb_key}/chunk_index"
                file_col = f"videos/{rgb_key}/file_index"
                ts_col = f"videos/{rgb_key}/from_timestamp"
                if chunk_col in row and file_col in row and ts_col in row:
                    chunk_idx = int(row[chunk_col])
                    file_idx = int(row[file_col])
                    from_ts = float(row[ts_col])
                    video_path = str(
                        self.root
                        / "videos"
                        / rgb_key
                        / f"chunk-{chunk_idx:03d}"
                        / f"file-{file_idx:03d}.mp4"
                    )
                    video_frame_start = round(from_ts * self._fps)
                    video_map[rgb_key] = (video_path, video_frame_start)

            self._episodes.append(
                _EpisodeInfo(
                    episode_index=ep_idx,
                    length=length,
                    data_start=data_start,
                    video_map=video_map,
                )
            )

    def _build_window_index(self) -> None:
        """Compute valid sliding-window start positions within each episode."""
        window_len = self.obs_horizon + self.pred_horizon
        for ep_i, ep in enumerate(self._episodes):
            range_start = len(self._windows)
            if ep.length < window_len:
                continue
            for offset in range(ep.length - window_len + 1):
                self._windows.append((ep_i, offset))
            range_end = len(self._windows)
            if range_end > range_start:
                self._episode_window_ranges.append((range_start, range_end))

    def __len__(self) -> int:
        return len(self._windows)

    @property
    def episode_window_ranges(self) -> List[Tuple[int, int]]:
        return list(self._episode_window_ranges)

    def _open_reader(self, video_path: str) -> imageio.core.format.Format.Reader:
        # AV1 videos can trigger ffmpeg thread initialization failures when
        # several workers open readers concurrently, so keep each reader
        # single-threaded.
        return imageio.get_reader(
            video_path,
            format="ffmpeg",
            input_params=["-threads", "1"],
            output_params=["-threads", "1"],
        )

    def _close_reader(self, video_path: str) -> None:
        reader = self._reader_cache.pop(video_path, None)
        if reader is not None:
            reader.close()

    def _get_reader(
        self, video_path: str, force_reopen: bool = False
    ) -> imageio.core.format.Format.Reader:
        # Invalidate cache if we're in a new process - each DataLoader worker
        # must open its own ffmpeg subprocesses.
        pid = os.getpid()
        if self._cache_pid is not None and self._cache_pid != pid:
            self.close()
        self._cache_pid = pid

        if force_reopen:
            self._close_reader(video_path)

        if video_path not in self._reader_cache:
            self._reader_cache[video_path] = self._open_reader(video_path)
        return self._reader_cache[video_path]

    def _read_video_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        """Read a single video frame as (H, W, 3) uint8."""
        reader = self._get_reader(video_path)
        try:
            return reader.get_data(frame_idx)
        except OSError:
            reader = self._get_reader(video_path, force_reopen=True)
            return reader.get_data(frame_idx)

    def _read_video_frames(
        self, frame_requests: Dict[str, Sequence[int]]
    ) -> Dict[Tuple[str, int], torch.Tensor]:
        """Decode a batch of requested frames with per-video deduplication."""
        frame_tensors: Dict[Tuple[str, int], torch.Tensor] = {}
        for video_path, frame_indices in frame_requests.items():
            for frame_idx in sorted(set(frame_indices)):
                frame = self._read_video_frame(video_path, frame_idx)
                frame_tensors[(video_path, frame_idx)] = self._to_image_tensor(frame)
        return frame_tensors

    def _build_sample(
        self,
        idx: int,
        frame_tensors: Optional[Dict[Tuple[str, int], torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        ep_i, offset = self._windows[idx]
        ep = self._episodes[ep_i]
        total_len = self.obs_horizon + self.pred_horizon

        # Parquet data: state and actions
        global_start = ep.data_start + offset
        rows = self._df.iloc[global_start : global_start + total_len]
        low_dim_columns = {key: rows[key] for key in self.low_dim_keys if key in rows}
        action_column = rows["action"]

        # Build obs_dict (first obs_horizon frames)
        obs_dict: Dict[str, torch.Tensor] = {}

        # RGB images from video
        for key in self.rgb_keys:
            if key not in ep.video_map:
                continue
            video_path, vid_frame_start = ep.video_map[key]
            images = []
            for t in range(self.obs_horizon):
                frame_in_video = vid_frame_start + offset + t
                if frame_tensors is not None and (video_path, frame_in_video) in frame_tensors:
                    img_tensor = frame_tensors[(video_path, frame_in_video)]
                else:
                    img = self._read_video_frame(video_path, frame_in_video)
                    img_tensor = self._to_image_tensor(img)
                images.append(img_tensor)
            obs_dict[key] = torch.stack(images, dim=0)

        # Low-dim state
        for key in self.low_dim_keys:
            parts = []
            for t in range(self.obs_horizon):
                val = low_dim_columns[key].iloc[t]
                parts.append(self._to_float_tensor(val))
            stacked = torch.stack(parts, dim=0)
            if self.state_normalize_fn is not None:
                stacked = self.state_normalize_fn(stacked)
            obs_dict[key] = stacked

        # Action target (pred_horizon frames, starting after observation window)
        actions = []
        for t in range(self.pred_horizon):
            act = action_column.iloc[self.obs_horizon + t]
            actions.append(self._to_float_tensor(act))
        action_tensor = torch.stack(actions, dim=0)
        if self.action_normalize_fn is not None:
            action_tensor = self.action_normalize_fn(action_tensor)

        return obs_dict, action_tensor

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return self._build_sample(idx)

    def __getitems__(self, indices: Sequence[int]) -> List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        frame_requests: Dict[str, List[int]] = {}
        for idx in indices:
            ep_i, offset = self._windows[idx]
            ep = self._episodes[ep_i]
            for key in self.rgb_keys:
                if key not in ep.video_map:
                    continue
                video_path, vid_frame_start = ep.video_map[key]
                requested_frames = frame_requests.setdefault(video_path, [])
                for t in range(self.obs_horizon):
                    requested_frames.append(vid_frame_start + offset + t)

        frame_tensors = self._read_video_frames(frame_requests)
        return [self._build_sample(idx, frame_tensors=frame_tensors) for idx in indices]

    def _to_image_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert a (H, W, 3) uint8 image to (3, H', W') float [0,1]."""
        if self.enable_image_augmentation:
            img = self._augment_image(img)
        t = torch.from_numpy(img).float() / 255.0
        t = t.permute(2, 0, 1)
        t = F.interpolate(
            t.unsqueeze(0),
            size=(self.rgb_image_size, self.rgb_image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return t

    @staticmethod
    def _augment_image(img: np.ndarray) -> np.ndarray:
        """Apply random color jitter and crop to a (H, W, 3) uint8 image."""
        # Random brightness/contrast jitter
        if np.random.rand() < 0.5:
            img = img.astype(np.float32)
            brightness = np.random.uniform(0.8, 1.2)
            img = img * brightness
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Random crop (keep 80-100% of the image)
        if np.random.rand() < 0.5:
            h, w = img.shape[:2]
            scale = np.random.uniform(0.80, 1.0)
            crop_h, crop_w = int(h * scale), int(w * scale)
            y = np.random.randint(0, h - crop_h + 1)
            x = np.random.randint(0, w - crop_w + 1)
            img = img[y : y + crop_h, x : x + crop_w]

        return img

    @staticmethod
    def _to_float_tensor(val) -> torch.Tensor:
        if isinstance(val, torch.Tensor):
            return val.float()
        arr = np.asarray(val, dtype=np.float32)
        if not arr.flags.writeable:
            arr = arr.copy()
        return torch.from_numpy(arr)

    def close(self) -> None:
        """Close any open video readers."""
        for reader in self._reader_cache.values():
            reader.close()
        self._reader_cache.clear()
