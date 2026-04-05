# liquid_lehome — Liquid+MDN Policy for LeHome Data

Standalone training and evaluation of the Liquid+MDN imitation learning policy on
LeHome-format (LeRobot v3.0) datasets.  Reuses the model classes from
`liquid_lehome/modeling.py`.

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

LeHome datasets store control/state/action streams at `30 Hz`, while the garment
simulator steps at `90 Hz`. The `liquid_lehome` config now treats `control_hz`
and `deployment_env_hz` as explicit settings:

- Training validates that `meta/info.json` matches `control_hz`
- The custom LeHome policy adapter holds each predicted action for
  `deployment_env_hz / control_hz` simulator steps during `scripts.eval`
- With the default LeHome settings, this means `30 Hz` control on a `90 Hz`
  simulator loop, i.e. `3` simulator steps per policy action

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

## Fast Diagnostics

Use the diagnostics entrypoint before committing to a long retrain. It can run
cheap offline checks on a checkpoint and/or a short training smoke on a handful
of episodes.

```bash
# Check whether a checkpoint reacts more to RGB swaps than to state swaps
python -m liquid_lehome --mode diagnose \
    --checkpoint outputs/liquid_lehome/best.pt \
    --dataset_root /path/to/dataset \
    --device cpu

# Run a 100-step training smoke on 4 episodes, then run the same sensitivity checks
python -m liquid_lehome --mode diagnose \
    --config configs/lehome/liquid_mdn_lehome.json \
    --dataset_root /path/to/dataset \
    --device cpu \
    --train_smoke_steps 100 \
    --train_smoke_episodes 4
```

The checkpoint sensitivity suite reports:

- `zero_rgb_delta`: how much predictions change when RGB is zeroed
- `rgb_swap_delta`: how much predictions change when RGB comes from another episode
- `state_swap_delta`: how much predictions change when state comes from another episode
- `rgb_state_ratio`: `rgb_swap_delta / state_swap_delta`; very small values usually mean RGB is being ignored
- `pairwise_chunk_delta`: how different the first predicted action chunks are across different dataset windows

The train smoke additionally reports gradient norms for coarse parameter groups
(`rgb`, `state`, `temporal`, `other`) so you can catch cases where the visual
path receives almost no learning signal.

### Using the policy in LeHome evaluation

```python
from liquid_lehome.eval import LiquidLeHomePolicy

policy = LiquidLeHomePolicy(
    checkpoint_path="outputs/liquid_lehome/best.pt",
    dataset_root="/path/to/dataset",  # for normalization stats
    device="cuda",
    env_step_hz=90,  # hold each control action for 3 sim steps by default
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
- The model architecture (Liquid/GRU observation core + MDN head + shared
  backbone) is implemented directly in `liquid_lehome/modeling.py`.
- Teacher forcing schedule: ratio 1.0 → 0.15, free-running weight 0.2 → 0.75
  over the course of training (same as the original RoboMimic experiments).
