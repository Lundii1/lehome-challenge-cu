"""
Custom policy for LeHome Challenge using Liquid+MDN.

Wraps ``LiquidLeHomePolicy`` from the ``liquid_lehome`` package so that the
LeHome eval pipeline (``scripts.eval --policy_type custom``) can use the
trained Liquid+MDN checkpoint directly.
"""

from typing import Dict, Optional

import numpy as np

from .base_policy import BasePolicy
from .registry import PolicyRegistry

from liquid_lehome.eval import LiquidLeHomePolicy


@PolicyRegistry.register("custom")
class CustomPolicy(BasePolicy):
    """Liquid+MDN policy adapter for the LeHome challenge eval loop."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        dataset_root: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if model_path is None:
            raise ValueError(
                "CustomPolicy requires --policy_path pointing to a Liquid+MDN checkpoint (.pt)"
            )
        self._policy = LiquidLeHomePolicy(
            checkpoint_path=model_path,
            device=device,
            dataset_root=dataset_root,
            env_step_hz=90,
        )
        print(
            f"[CustomPolicy] Loaded Liquid+MDN policy from {model_path} "
            f"(device={self._policy.device}, action_dim={self._policy.cfg.action_dim}, "
            f"control_hz={self._policy.control_hz}, env_step_hz={self._policy.env_step_hz}, "
            f"env_steps_per_policy_step={self._policy.env_steps_per_policy_step})"
        )

    def reset(self):
        self._policy.reset()
        print("[CustomPolicy] reset() called - new episode")

    def get_debug_snapshot(self):
        return self._policy.get_debug_snapshot()

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        return self._policy.select_action(observation)
