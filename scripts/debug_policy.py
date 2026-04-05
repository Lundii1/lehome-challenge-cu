"""Diagnostic: check what the Liquid+MDN policy actually outputs."""
import sys
import numpy as np
import torch

from liquid_lehome.eval import LiquidLeHomePolicy
from liquid_lehome.normalize import load_stats

CHECKPOINT = "outputs/liquid_lehome/best.pt"
DATASET_ROOT = "Datasets/example/four_types_merged"

# 1. Load checkpoint and inspect
print("=== 1. Checkpoint inspection ===")
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
print(f"  Keys: {list(ckpt.keys())}")
if "epoch" in ckpt:
    print(f"  Epoch: {ckpt['epoch']}")
if "best_val_loss" in ckpt:
    print(f"  Best val loss: {ckpt['best_val_loss']:.6f}")
if "config" in ckpt:
    cfg_dict = ckpt["config"]
    print(f"  Config normalize_actions: {cfg_dict.get('normalize_actions')}")
    print(f"  Config normalize_state: {cfg_dict.get('normalize_state')}")
    print(f"  Config action_dim: {cfg_dict.get('action_dim')}")
    print(f"  Config observation_horizon: {cfg_dict.get('observation_horizon')}")
    print(f"  Config action_horizon: {cfg_dict.get('action_horizon')}")
    print(f"  Config prediction_horizon: {cfg_dict.get('prediction_horizon')}")

# 2. Load normalization stats
print("\n=== 2. Normalization stats ===")
norm_stats = load_stats(DATASET_ROOT)
action_stats = norm_stats.get("action")
state_stats = norm_stats.get("observation.state")
if action_stats:
    print(f"  Action min: {action_stats.min[:3].tolist()}... (first 3 dims)")
    print(f"  Action max: {action_stats.max[:3].tolist()}... (first 3 dims)")
    print(f"  Action mean: {action_stats.mean[:3].tolist()}... (first 3 dims)")
if state_stats:
    print(f"  State min: {state_stats.min[:3].tolist()}... (first 3 dims)")
    print(f"  State max: {state_stats.max[:3].tolist()}... (first 3 dims)")
    print(f"  State mean: {state_stats.mean[:3].tolist()}... (first 3 dims)")

# 3. Create policy and test inference
print("\n=== 3. Policy inference test ===")
policy = LiquidLeHomePolicy(
    checkpoint_path=CHECKPOINT,
    device="cpu",
    dataset_root=DATASET_ROOT,
)
print(f"  Policy device: {policy.device}")
print(f"  Action stats loaded: {policy.action_stats is not None}")
print(f"  State stats loaded: {policy.state_stats is not None}")

# Create realistic-ish observations
# Use the mean state as initial state
mean_state = np.array(action_stats.mean, dtype=np.float32) if action_stats else np.zeros(12, dtype=np.float32)
print(f"\n  Using mean state as observation.state: {mean_state}")

# Test with random images and mean state
fake_obs = {
    "observation.images.top_rgb": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
    "observation.images.left_rgb": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
    "observation.images.right_rgb": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
    "observation.state": mean_state.copy(),
}

policy.reset()
print("\n  Running 10 inference steps...")
for step in range(10):
    action = policy.select_action(fake_obs)
    print(f"  Step {step:2d}: action={action[:4]}... (first 4 dims), "
          f"min={action.min():.4f}, max={action.max():.4f}, "
          f"abs_mean={np.abs(action).mean():.4f}")

# 4. Check if actions change when state changes
print("\n=== 4. Sensitivity test: varying state ===")
policy.reset()
for trial_name, state_val in [
    ("zeros", np.zeros(12, dtype=np.float32)),
    ("ones", np.ones(12, dtype=np.float32)),
    ("neg_ones", -np.ones(12, dtype=np.float32)),
    ("mean_state", mean_state.copy()),
]:
    obs = {
        "observation.images.top_rgb": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
        "observation.images.left_rgb": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
        "observation.images.right_rgb": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
        "observation.state": state_val,
    }
    policy.reset()
    # Run a few steps to fill the queue
    a = policy.select_action(obs)
    print(f"  {trial_name:12s}: action={a[:4]}..., "
          f"min={a.min():.4f}, max={a.max():.4f}")

# 5. Check raw model output (before denormalization)
print("\n=== 5. Raw model output (normalized space) ===")
policy.reset()
obs = {
    "observation.images.top_rgb": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
    "observation.images.left_rgb": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
    "observation.images.right_rgb": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
    "observation.state": mean_state.copy(),
}
processed = policy._preprocess(obs)
policy._append_obs(processed)
stacked_obs = policy._stack_obs_queue()

with torch.inference_mode():
    raw_actions = policy.policy.rollout_actions(
        obs_dict=stacked_obs,
        mode=policy.cfg.sample_selection_mode,
        sample_k=policy.cfg.sample_selection_k,
    )
print(f"  Raw action shape: {raw_actions.shape}")
print(f"  Raw action range: [{raw_actions.min().item():.4f}, {raw_actions.max().item():.4f}]")
print(f"  Raw action mean:  {raw_actions.mean().item():.4f}")
print(f"  Raw action std:   {raw_actions.std().item():.4f}")
print(f"  Raw first step:   {raw_actions[0, 0, :4].tolist()} (first 4 dims)")

# 6. Compare with dataset action distribution
print("\n=== 6. Dataset action sample ===")
try:
    import pandas as pd
    from pathlib import Path
    data_dir = Path(DATASET_ROOT) / "data"
    chunks = sorted(data_dir.rglob("*.parquet"))
    if chunks:
        df = pd.read_parquet(chunks[0])
        actions = np.stack(df["action"].values[:10])
        print(f"  First 10 dataset actions (first 4 dims each):")
        for i, a in enumerate(actions):
            print(f"    [{i}]: {a[:4]}..., min={a.min():.4f}, max={a.max():.4f}")
except Exception as e:
    print(f"  Could not load dataset: {e}")

print("\n=== Done ===")
