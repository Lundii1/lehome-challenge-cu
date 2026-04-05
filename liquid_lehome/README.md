# liquid_lehome — Liquid+MDN Policy for LeHome Data

Standalone training and evaluation of the Liquid+MDN imitation learning policy on
LeHome-format (LeRobot v3.0) datasets.  Reuses the model classes from
`liquid_robomimic/modeling.py` without depending on the RoboMimic framework.

## Expected Dataset Layout

```
dataset_root/
├── data/
│   └── chunk-000/
│       ├── file-000.parquet      # observation.state, action, episode_index, ...
│       └── file-001.parquet
├── videos/
│   ├── observation.images.top_rgb/chunk-000/file-000.mp4
│   ├── observation.images.left_rgb/chunk-000/file-000.mp4
│   └── observation.images.right_rgb/chunk-000/file-000.mp4
├── meta/
│   ├── info.json                 # fps, features, total_episodes, total_frames
│   ├── stats.json                # min/max/mean/std for normalization
│   └── episodes/chunk-000/file-000.parquet   # per-episode video mapping
```

**Default features:**

| Key | Shape | Description |
|-----|-------|-------------|
| `observation.state` | (12,) float32 | Dual-arm joint positions |
| `action` | (12,) float32 | Joint position targets |
| `observation.images.top_rgb` | (480, 640, 3) uint8 | Top camera RGB |
| `observation.images.left_rgb` | (480, 640, 3) uint8 | Left camera RGB |
| `observation.images.right_rgb` | (480, 640, 3) uint8 | Right camera RGB |

## Configurable Field Mapping

All observation keys are configurable via the JSON config:

```json
{
    "rgb_keys": ["observation.images.top_rgb", "observation.images.left_rgb"],
    "low_dim_keys": ["observation.state"],
    "action_dim": 12
}
```

- To use a single camera, set `rgb_keys` to just one key.
- To add depth: include the depth key in `low_dim_keys` (flattened) or add a
  custom image key to `rgb_keys` (single-channel depth images would need a
  minor encoder change).
- `action_dim` must match the dimension of the `action` column in parquet.

## Training

```bash
cd /path/to/liquidnets

# Full training (120 epochs)
python -m liquid_lehome.train \
    --config configs/lehome/liquid_mdn_lehome.json \
    --dataset_root /path/to/dataset

# Quick smoke test (1 epoch, CPU)
python -m liquid_lehome.train \
    --config configs/lehome/liquid_mdn_lehome.json \
    --dataset_root /path/to/dataset \
    --epochs 1 --device cpu
```

If AV1 video decoding is unstable on your machine, retry with fewer
DataLoader workers:

```bash
python -m liquid_lehome.train \
    --config configs/lehome/liquid_mdn_lehome.json \
    --dataset_root /path/to/dataset \
    --num_data_workers 0
```

Checkpoints are saved to `output_dir` (default: `outputs/liquid_lehome/`):
- `best.pt` — lowest validation loss
- `epoch_NNNN.pt` — periodic snapshots
- `last.pt` — final epoch

## Evaluation

```bash
# Smoke-check a trained checkpoint
python -m liquid_lehome.eval \
    --checkpoint outputs/liquid_lehome/best.pt \
    --dataset_root /path/to/dataset

# Export an offline episode video from the dataset
python -m liquid_lehome.eval \
    --checkpoint outputs/liquid_lehome/best.pt \
    --dataset_root /path/to/dataset \
    --video_path outputs/lehome_eval.mp4 \
    --episode_index 0

# With explicit config
python -m liquid_lehome.eval \
    --checkpoint outputs/liquid_lehome/best.pt \
    --config configs/lehome/liquid_mdn_lehome.json
```

When `--video_path` is set, the command replays one dataset episode offline, tiles
the configured RGB cameras into a single video, and prints mean action MSE against
the recorded actions for that episode. Use `--max_steps` to limit clip length and
`--video_fps` to override the dataset FPS.

### Using the policy in LeHome evaluation

```python
from liquid_lehome.eval import LiquidLeHomePolicy

policy = LiquidLeHomePolicy(
    checkpoint_path="outputs/liquid_lehome/best.pt",
    dataset_root="/path/to/dataset",  # for normalization stats
    device="cuda",
)
policy.reset()

# In your eval loop:
action = policy.select_action(observation)
# observation: dict with numpy arrays (images as HWC uint8, state as float32)
# action: numpy (12,) float32 in raw joint-position space
```

## Tests

```bash
python -m unittest tests.test_liquid_lehome -v
```

## Assumptions

- Actions are normalized to [-1, 1] using min-max from `meta/stats.json` during
  training, and denormalized back at inference.
- Images are resized to 96x96 (configurable via `rgb_image_size`).
- Video frames are decoded with `imageio`; parquet data via `pandas + pyarrow`.
- The model architecture (Liquid CfC + MDN head + shared backbone) is imported
  unchanged from `liquid_robomimic/modeling.py`.
- Teacher forcing schedule: ratio 1.0 → 0.15, free-running weight 0.2 → 0.75
  over the course of training (same as the original RoboMimic experiments).
