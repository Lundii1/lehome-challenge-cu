import os
import argparse
import gymnasium as gym
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg

from scripts.eval_policy import PolicyRegistry
from scripts.eval_policy.base_policy import BasePolicy

from scripts.utils.eval_utils import (
    convert_ee_pose_to_joints,
    save_videos_from_observations,
    calculate_and_print_metrics,
)

from lehome.utils.record import (
    RateLimiter,
    get_next_experiment_path_with_gap,
    append_episode_initial_pose,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from .common import stabilize_garment_after_reset
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


def _format_array(values: np.ndarray, precision: int = 4) -> str:
    return np.array2string(
        np.asarray(values, dtype=np.float32),
        precision=precision,
        suppress_small=True,
        max_line_width=200,
    )


def _should_print_full_diag(step_index: int) -> bool:
    return step_index < 10 or step_index % 25 == 0


def _should_print_compact_summary(step_index: int) -> bool:
    return step_index < 10 or step_index % 5 == 0


def _split_bimanual(values: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape[0] >= 12:
        return arr[:6], arr[6:12]
    if arr.shape[0] >= 6:
        return arr[:6], None
    return arr, None


def _build_joint_debug_metrics(
    pre_state: np.ndarray,
    commanded_action: np.ndarray,
    post_state: np.ndarray,
) -> Dict[str, Any]:
    pre_state = np.asarray(pre_state, dtype=np.float32)
    commanded_action = np.asarray(commanded_action, dtype=np.float32)
    post_state = np.asarray(post_state, dtype=np.float32)

    command_minus_pre = commanded_action - pre_state
    post_minus_pre = post_state - pre_state
    command_minus_post = commanded_action - post_state
    abs_tracking = np.abs(command_minus_post)

    left_tracking, right_tracking = _split_bimanual(abs_tracking)
    metrics: Dict[str, Any] = {
        "pre_state": pre_state,
        "commanded_action": commanded_action,
        "post_state": post_state,
        "command_minus_pre": command_minus_pre,
        "post_minus_pre": post_minus_pre,
        "command_minus_post": command_minus_post,
        "tracking_mean": float(abs_tracking.mean()),
        "tracking_max": float(abs_tracking.max()),
        "left_tracking_mean": float(np.abs(left_tracking).mean()) if left_tracking.size else 0.0,
        "right_tracking_mean": (
            float(np.abs(right_tracking).mean()) if right_tracking is not None and right_tracking.size else None
        ),
        "left_gripper_pre": float(pre_state[5]) if pre_state.shape[0] >= 6 else None,
        "left_gripper_cmd": float(commanded_action[5]) if commanded_action.shape[0] >= 6 else None,
        "left_gripper_post": float(post_state[5]) if post_state.shape[0] >= 6 else None,
        "left_gripper_err": float(abs_tracking[5]) if abs_tracking.shape[0] >= 6 else None,
        "right_gripper_pre": float(pre_state[11]) if pre_state.shape[0] >= 12 else None,
        "right_gripper_cmd": float(commanded_action[11]) if commanded_action.shape[0] >= 12 else None,
        "right_gripper_post": float(post_state[11]) if post_state.shape[0] >= 12 else None,
        "right_gripper_err": float(abs_tracking[11]) if abs_tracking.shape[0] >= 12 else None,
    }
    return metrics


class GripperWorldDiagnostics:
    """Optional world-space gripper diagnostics using forward kinematics."""

    def __init__(self, urdf_path: str):
        self.available = False
        self._warning_emitted = False
        self._error_message: Optional[str] = None
        self._solver = None
        self._compute_ee_pose_single_arm = None
        self._quat_to_mat = None

        try:
            from lehome.utils import RobotKinematics, compute_ee_pose_single_arm
            from lehome.utils.ee_pose_utils import quat_to_mat

            joint_names = [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ]
            self._solver = RobotKinematics(
                str(urdf_path),
                target_frame_name="gripper_frame_link",
                joint_names=joint_names,
            )
            self._compute_ee_pose_single_arm = compute_ee_pose_single_arm
            self._quat_to_mat = quat_to_mat
            self.available = True
        except Exception as exc:
            self._error_message = str(exc)

    def _warn_once(self, context: str) -> None:
        if not self._warning_emitted:
            logger.warning(f"{context}: {self._error_message}")
            self._warning_emitted = True

    def _commanded_world_position(self, arm, arm_action: np.ndarray) -> np.ndarray:
        ee_pose = self._compute_ee_pose_single_arm(
            self._solver,
            np.asarray(arm_action, dtype=np.float32),
            state_unit="rad",
        )
        base_pos = arm.data.root_pos_w[0].detach().cpu().numpy()
        base_quat_xyzw = arm.data.root_quat_w[0].detach().cpu().numpy()
        base_rot = self._quat_to_mat(base_quat_xyzw)
        return base_pos + base_rot @ ee_pose[:3]

    def compute(self, env: DirectRLEnv, commanded_action: np.ndarray) -> Optional[Dict[str, Any]]:
        if not self.available:
            self._warn_once(
                "[Diagnostics] World-space gripper diagnostics unavailable, falling back to joint-space only"
            )
            return None

        try:
            left_cmd, right_cmd = _split_bimanual(commanded_action)
            left_actual = env.left_arm.data.body_link_pos_w[0, -1].detach().cpu().numpy()
            left_commanded = self._commanded_world_position(env.left_arm, left_cmd)

            result: Dict[str, Any] = {
                "left_actual": left_actual.astype(np.float32),
                "left_commanded": left_commanded.astype(np.float32),
                "left_error_norm": float(np.linalg.norm(left_commanded - left_actual)),
            }

            if right_cmd is not None and hasattr(env, "right_arm"):
                right_actual = env.right_arm.data.body_link_pos_w[0, -1].detach().cpu().numpy()
                right_commanded = self._commanded_world_position(env.right_arm, right_cmd)
                result.update(
                    {
                        "right_actual": right_actual.astype(np.float32),
                        "right_commanded": right_commanded.astype(np.float32),
                        "right_error_norm": float(np.linalg.norm(right_commanded - right_actual)),
                    }
                )
            return result
        except Exception as exc:
            self._error_message = str(exc)
            self.available = False
            self._warn_once(
                "[Diagnostics] Failed to compute world-space gripper diagnostics, falling back to joint-space only"
            )
            return None


def _log_step_summary(
    step_index: int,
    reward: float,
    success: bool,
    model_snapshot: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    action_status = "held" if model_snapshot.get("action_origin") == "held" else "new"
    right_gripper = ""
    if metrics.get("right_gripper_cmd") is not None and metrics.get("right_gripper_post") is not None:
        right_gripper = (
            f" | R grip cmd={metrics['right_gripper_cmd']:.4f} real={metrics['right_gripper_post']:.4f}"
        )

    logger.info(
        "[Diag][Step %03d] reward=%.4f success=%s action=%s tracking_mean=%.4f "
        "L grip cmd=%.4f real=%.4f%s",
        step_index,
        reward,
        success,
        action_status,
        metrics["tracking_mean"],
        metrics["left_gripper_cmd"] if metrics["left_gripper_cmd"] is not None else float("nan"),
        metrics["left_gripper_post"] if metrics["left_gripper_post"] is not None else float("nan"),
        right_gripper,
    )


def _log_full_step_debug(
    step_index: int,
    reward: float,
    success: bool,
    model_snapshot: Dict[str, Any],
    metrics: Dict[str, Any],
    world_debug: Optional[Dict[str, Any]],
) -> None:
    logger.info(
        "[Diag][Step %03d] reward=%.4f success=%s source=%s fresh_rollout=%s hold_remaining=%d "
        "queue_len=%d mode=%s control_hz=%d env_step_hz=%d repeat=%d",
        step_index,
        reward,
        success,
        model_snapshot.get("action_origin"),
        model_snapshot.get("fresh_rollout_generated"),
        model_snapshot.get("remaining_hold_count", 0),
        model_snapshot.get("action_queue_len", 0),
        model_snapshot.get("sample_selection_mode"),
        model_snapshot.get("control_hz"),
        model_snapshot.get("env_step_hz"),
        model_snapshot.get("env_steps_per_policy_step"),
    )
    logger.info(
        "  model_action: %s",
        _format_array(np.asarray(model_snapshot["action"], dtype=np.float32)),
    )
    logger.info("  pre_state: %s", _format_array(metrics["pre_state"]))
    logger.info("  commanded_action: %s", _format_array(metrics["commanded_action"]))
    logger.info("  post_state: %s", _format_array(metrics["post_state"]))
    if model_snapshot.get("action_minus_state") is not None:
        logger.info(
            "  action_minus_state: %s",
            _format_array(np.asarray(model_snapshot["action_minus_state"], dtype=np.float32)),
        )
    logger.info("  command_minus_pre: %s", _format_array(metrics["command_minus_pre"]))
    logger.info("  post_minus_pre: %s", _format_array(metrics["post_minus_pre"]))
    logger.info("  command_minus_post: %s", _format_array(metrics["command_minus_post"]))
    logger.info(
        "  tracking_error mean/max: %.4f / %.4f rad",
        metrics["tracking_mean"],
        metrics["tracking_max"],
    )
    if metrics.get("right_tracking_mean") is None:
        logger.info("  arm_tracking mean: left=%.4f", metrics["left_tracking_mean"])
    else:
        logger.info(
            "  arm_tracking mean: left=%.4f right=%.4f",
            metrics["left_tracking_mean"],
            metrics["right_tracking_mean"],
        )
    logger.info(
        "  gripper_joint left: pre=%.4f cmd=%.4f post=%.4f err=%.4f",
        metrics["left_gripper_pre"],
        metrics["left_gripper_cmd"],
        metrics["left_gripper_post"],
        metrics["left_gripper_err"],
    )
    if metrics.get("right_gripper_pre") is not None:
        logger.info(
            "  gripper_joint right: pre=%.4f cmd=%.4f post=%.4f err=%.4f",
            metrics["right_gripper_pre"],
            metrics["right_gripper_cmd"],
            metrics["right_gripper_post"],
            metrics["right_gripper_err"],
        )

    rollout_preview = model_snapshot.get("rollout_preview")
    if rollout_preview:
        for preview_index, preview_action in enumerate(rollout_preview):
            logger.info(
                "  rollout_preview[%d]: %s",
                preview_index,
                _format_array(np.asarray(preview_action, dtype=np.float32)),
            )

    if world_debug is not None:
        logger.info(
            "  gripper_world left: actual=%s commanded=%s err=%.4f m",
            _format_array(world_debug["left_actual"]),
            _format_array(world_debug["left_commanded"]),
            world_debug["left_error_norm"],
        )
        if "right_actual" in world_debug:
            logger.info(
                "  gripper_world right: actual=%s commanded=%s err=%.4f m",
                _format_array(world_debug["right_actual"]),
                _format_array(world_debug["right_commanded"]),
                world_debug["right_error_norm"],
            )


def _log_episode_debug_summary(
    episode_index: int,
    episode_return: float,
    is_success: bool,
    summary: Dict[str, Any],
) -> None:
    logger.info(
        "[Diag][Episode %d] return=%.2f success=%s tracking_mean=%.4f tracking_max=%.4f "
        "left_tracking_mean=%.4f",
        episode_index + 1,
        episode_return,
        is_success,
        summary["tracking_mean"],
        summary["tracking_max"],
        summary["left_tracking_mean"],
    )
    if summary.get("right_tracking_mean") is not None:
        logger.info(
            "[Diag][Episode %d] right_tracking_mean=%.4f",
            episode_index + 1,
            summary["right_tracking_mean"],
        )
    logger.info(
        "[Diag][Episode %d] gripper_joint_error left=%.4f right=%s held_steps=%d fresh_steps=%d",
        episode_index + 1,
        summary["left_gripper_err_mean"],
        (
            f"{summary['right_gripper_err_mean']:.4f}"
            if summary.get("right_gripper_err_mean") is not None
            else "n/a"
        ),
        summary["held_steps"],
        summary["fresh_steps"],
    )
    if summary.get("left_world_err_mean") is not None:
        logger.info(
            "[Diag][Episode %d] gripper_world_error left=%.4f m right=%s",
            episode_index + 1,
            summary["left_world_err_mean"],
            (
                f"{summary['right_world_err_mean']:.4f} m"
                if summary.get("right_world_err_mean") is not None
                else "n/a"
            ),
        )


def run_evaluation_loop(
    env: DirectRLEnv,
    policy: BasePolicy,
    args: argparse.Namespace,
    ee_solver: Optional[Any] = None,
    is_bimanual: bool = False,
    garment_name: Optional[str] = None,
    gripper_world_diagnostics: Optional[GripperWorldDiagnostics] = None,
) -> List[Dict[str, Any]]:
    """
    Core evaluation loop.
    Refactored to be agnostic of specific model implementations.
    """

    # --- Dataset Recording Setup (Optional) ---
    eval_dataset = None
    json_path = None
    episode_index = 0
    if args.save_datasets:
        features = None
        if args.dataset_root and Path(args.dataset_root).exists():
            source_dataset = LeRobotDataset(repo_id="collected_dataset", root=Path(args.dataset_root))
            features = dict(source_dataset.meta.features)
            fps = source_dataset.fps
        else:
            fps = 30  # Default FPS if no source dataset is provided
            action_names = [
                "shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper",
            ]
            if is_bimanual:
                left_names = [f"left_{n}" for n in action_names]
                right_names = [f"right_{n}" for n in action_names]
                joint_names = left_names + right_names
            else:
                joint_names = action_names
            dim = len(joint_names)
            features = {
                "observation.state": {
                    "dtype": "float32",
                    "shape": (dim,),
                    "names": joint_names,
                },
                "action": {
                    "dtype": "float32",
                    "shape": (dim,),
                    "names": joint_names,
                },
            }
            image_keys = ["top_rgb", "left_rgb", "right_rgb"] if is_bimanual else ["top_rgb", "wrist_rgb"]
            for key in image_keys:
                features[f"observation.images.{key}"] = {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"],
                }
        root_path = Path(args.eval_dataset_path)
        eval_dataset = LeRobotDataset.create(
            repo_id="lehome_eval",
            fps=fps,
            root=get_next_experiment_path_with_gap(root_path),
            use_videos=True,
            image_writer_threads=8,
            image_writer_processes=0,
            features=features,
        )
        json_path = eval_dataset.root / "meta" / "garment_info.json"

    all_episode_metrics = []
    logger.info(f"Starting evaluation: {args.num_episodes} episodes")
    rate_limiter = RateLimiter(args.step_hz)

    for i in range(args.num_episodes):
        # 1. Reset Environment & Policy
        env.reset()
        policy.reset()
        stabilize_garment_after_reset(env, args)

        # 2. Initial Observation (Numpy)
        object_initial_pose = env.get_all_pose() if args.save_datasets else None
        observation_dict = env._get_observations()

        # Prepare for video recording
        episode_frames = (
            {k: [] for k in observation_dict.keys() if "images" in k}
            if args.save_video
            else {}
        )

        episode_return = 0.0
        episode_length = 0
        extra_steps = 0
        success_flag = False
        success = torch.tensor(False)
        last_success_result = False
        tracking_means: list[float] = []
        tracking_maxes: list[float] = []
        left_tracking_means: list[float] = []
        right_tracking_means: list[float] = []
        left_gripper_errs: list[float] = []
        right_gripper_errs: list[float] = []
        left_world_errs: list[float] = []
        right_world_errs: list[float] = []
        held_steps = 0
        fresh_steps = 0

        for st in range(args.max_steps):
            if rate_limiter:
                rate_limiter.sleep(env)

            pre_obs = observation_dict

            # 3. Policy Inference (The core abstraction)
            # Input: Numpy Dict -> Output: Numpy Array
            action_np = policy.select_action(pre_obs)
            debug_snapshot = policy.get_debug_snapshot()

            # 4. Prepare Action for Environment (Tensor)
            # Convert numpy action to tensor for Isaac Lab
            action = torch.from_numpy(action_np).float().to(args.device).unsqueeze(0)

            # 5. Inverse Kinematics (Optional Helper Logic)
            # If policy outputs EE pose but env needs joints
            if args.use_ee_pose and ee_solver is not None:
                current_joints = (
                    torch.from_numpy(observation_dict["observation.state"])
                    .float()
                    .to(args.device)
                )
                action = convert_ee_pose_to_joints(
                    ee_pose_action=action.squeeze(0),
                    current_joints=current_joints,
                    solver=ee_solver,
                    is_bimanual=is_bimanual,
                    state_unit="rad",
                    device=args.device,
                ).unsqueeze(0)
            commanded_action_np = action.squeeze(0).detach().cpu().numpy().copy()

            # 6. Step Environment
            env.step(action)

            # Check success first
            if not success_flag:
                success = env._get_success()
                if success.item():
                    success_flag = True
                    extra_steps = 50  # Run a bit longer after success to settle
            last_success_result = bool(success.item()) if isinstance(success, torch.Tensor) else bool(success)

            # Get reward from environment (Isaac Lab stores rewards internally)
            reward_value = env._get_rewards()
            if isinstance(reward_value, torch.Tensor):
                reward = reward_value.item()
            else:
                reward = float(reward_value)

            # Accumulate reward for all steps (including post-success steps)
            episode_return += reward
            # Only count length before success (for consistency with episode termination)
            if not success_flag:
                episode_length += 1

            # Update Observation
            observation_dict = env._get_observations()
            post_obs = observation_dict

            if debug_snapshot is not None and "observation.state" in pre_obs and "observation.state" in post_obs:
                metrics = _build_joint_debug_metrics(
                    pre_state=pre_obs["observation.state"],
                    commanded_action=commanded_action_np,
                    post_state=post_obs["observation.state"],
                )
                world_debug = None
                if gripper_world_diagnostics is not None:
                    world_debug = gripper_world_diagnostics.compute(env, commanded_action_np)

                tracking_means.append(metrics["tracking_mean"])
                tracking_maxes.append(metrics["tracking_max"])
                left_tracking_means.append(metrics["left_tracking_mean"])
                if metrics.get("right_tracking_mean") is not None:
                    right_tracking_means.append(metrics["right_tracking_mean"])
                if metrics.get("left_gripper_err") is not None:
                    left_gripper_errs.append(metrics["left_gripper_err"])
                if metrics.get("right_gripper_err") is not None:
                    right_gripper_errs.append(metrics["right_gripper_err"])
                if world_debug is not None:
                    left_world_errs.append(world_debug["left_error_norm"])
                    if "right_error_norm" in world_debug:
                        right_world_errs.append(world_debug["right_error_norm"])

                if debug_snapshot.get("action_origin") == "held":
                    held_steps += 1
                else:
                    fresh_steps += 1

                if _should_print_compact_summary(st):
                    _log_step_summary(
                        step_index=st,
                        reward=reward,
                        success=last_success_result,
                        model_snapshot=debug_snapshot,
                        metrics=metrics,
                    )
                if _should_print_full_diag(st):
                    _log_full_step_debug(
                        step_index=st,
                        reward=reward,
                        success=last_success_result,
                        model_snapshot=debug_snapshot,
                        metrics=metrics,
                        world_debug=world_debug,
                    )

            # Recording
            if args.save_datasets:
                frame = {
                    k: v
                    for k, v in observation_dict.items()
                    if k != "observation.top_depth"
                }
                frame["task"] = args.task_description
                eval_dataset.add_frame(frame)

            if args.save_video:
                for key, val in observation_dict.items():
                    if "images" in key:
                        episode_frames[key].append(val.copy())

            if success_flag:
                extra_steps -= 1
                if extra_steps <= 0:
                    break

        # --- End of Episode Handling ---
        is_success = success.item() if success_flag else False

        episode_debug_summary = {
            "tracking_mean": float(np.mean(tracking_means)) if tracking_means else 0.0,
            "tracking_max": float(np.max(tracking_maxes)) if tracking_maxes else 0.0,
            "left_tracking_mean": float(np.mean(left_tracking_means)) if left_tracking_means else 0.0,
            "right_tracking_mean": (
                float(np.mean(right_tracking_means)) if right_tracking_means else None
            ),
            "left_gripper_err_mean": (
                float(np.mean(left_gripper_errs)) if left_gripper_errs else 0.0
            ),
            "right_gripper_err_mean": (
                float(np.mean(right_gripper_errs)) if right_gripper_errs else None
            ),
            "left_world_err_mean": float(np.mean(left_world_errs)) if left_world_errs else None,
            "right_world_err_mean": float(np.mean(right_world_errs)) if right_world_errs else None,
            "held_steps": held_steps,
            "fresh_steps": fresh_steps,
            "last_success_result": last_success_result,
        }

        # Save Datasets
        if args.save_datasets:
            if success_flag:
                eval_dataset.save_episode()
                append_episode_initial_pose(
                    json_path,
                    episode_index,
                    object_initial_pose,
                    garment_name=garment_name,
                )
                episode_index += 1
            else:
                eval_dataset.clear_episode_buffer()

        # Save Videos (Using generic util)
        if args.save_video:
            save_videos_from_observations(
                episode_frames,
                success=success if success_flag else torch.tensor(False),
                save_dir=args.video_dir,
                episode_idx=i,
            )

        # Log Metrics
        all_episode_metrics.append(
            {"return": episode_return, "length": episode_length, "success": is_success}
        )
        logger.info(
            f"Episode {i + 1}/{args.num_episodes}: Return={episode_return:.2f}, Length={episode_length}, Success={is_success}"
        )
        _log_episode_debug_summary(
            episode_index=i,
            episode_return=episode_return,
            is_success=is_success,
            summary=episode_debug_summary,
        )
        logger.info(
            "[Diag][Episode %d] last_success_result=%s",
            i + 1,
            episode_debug_summary["last_success_result"],
        )

    return all_episode_metrics


def eval(args: argparse.Namespace, simulation_app: Any) -> None:
    """
    Main entry point for evaluation logic.
    """
    # 1. Environment Configuration
    env_cfg = parse_env_cfg(args.task, device=args.device)
    env_cfg.sim.use_fabric = False
    if args.use_random_seed:
        env_cfg.use_random_seed = True
    else:
        env_cfg.use_random_seed = False
        env_cfg.seed = args.seed
        # Propagate seed to sim config if structure exists
        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "seed"):
            env_cfg.sim.seed = args.seed

    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path

    # 2. Initialize Policy (Using the Policy Registry)
    # This replaces create_il_policy, make_pre_post_processors, etc.
    logger.info(f"Initializing Policy Type: {args.policy_type}")

    # Check if policy is registered
    if not PolicyRegistry.is_registered(args.policy_type):
        available_policies = PolicyRegistry.list_policies()
        raise ValueError(
            f"Policy type '{args.policy_type}' not found in registry. "
            f"Available policies: {', '.join(available_policies)}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_bimanual = "Bi" in args.task or "bi" in args.task.lower()

    # Create policy instance from registry with appropriate arguments
    # Different policies may require different initialization arguments
    policy_kwargs = {
        "device": device,
    }

    if args.policy_type == "lerobot":
        # LeRobot policy requires policy_path and dataset_root
        if not args.policy_path:
            raise ValueError("--policy_path is required for lerobot policy type")
        if not args.dataset_root:
            raise ValueError("--dataset_root is required for lerobot policy type")
        policy_kwargs.update(
            {
                "policy_path": args.policy_path,
                "dataset_root": args.dataset_root,
                "task_description": args.task_description,
            }
        )
    elif args.policy_type == "docker":
        # Docker policy connects to an external container
        policy_kwargs["docker_url"] = args.docker_url
    else:
        # For custom policies, pass policy_path as model_path if provided
        if args.policy_path:
            policy_kwargs["model_path"] = args.policy_path
        if args.dataset_root:
            policy_kwargs["dataset_root"] = args.dataset_root
        if getattr(args, "rollout_mode", None) is not None:
            policy_kwargs["rollout_mode"] = args.rollout_mode

    # Create policy from registry
    policy = PolicyRegistry.create(args.policy_type, **policy_kwargs)
    logger.info(f"Policy '{args.policy_type}' loaded successfully")

    gripper_world_diagnostics = None
    if args.policy_type == "custom":
        gripper_world_diagnostics = GripperWorldDiagnostics(args.ee_urdf_path)

    # 3. Initialize IK Solver (If needed)
    ee_solver = None
    if args.use_ee_pose:
        from lehome.utils import RobotKinematics

        urdf_path = args.ee_urdf_path  # Assuming path is handled or add check logic
        joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
        ee_solver = RobotKinematics(
            str(urdf_path),
            target_frame_name="gripper_frame_link",
            joint_names=joint_names,
        )
        logger.info(f"IK solver loaded.")

    # 4. Load Evaluation List
    # Only loads from 'Release' directory based on garment_type
    eval_list = []  # List of (name, stage)

    # Evaluate a specific category based on garment_type
    if args.garment_type == "custom":
        # For 'custom' type, we load from the root Release_test_list.txt
        eval_list_path = os.path.join(
            args.garment_cfg_base_path, "Release", "Release_test_list.txt"
        )
    else:
        # Map argument to specific sub-category directory
        type_map = {
            "top_long": "Top_Long",
            "top_short": "Top_Short",
            "pant_long": "Pant_Long",
            "pant_short": "Pant_Short",
        }
        file_prefix = type_map.get(args.garment_type, "Top_Long")
        # Path: Assets/objects/Challenge_Garment/Release/Top_Long/Top_Long.txt
        eval_list_path = os.path.join(
            args.garment_cfg_base_path, "Release", file_prefix, f"{file_prefix}.txt"
        )

    logger.info(
        f"Loading evaluation list for category '{args.garment_type}' from: {eval_list_path}"
    )

    if not os.path.exists(eval_list_path):
        raise FileNotFoundError(f"Evaluation list not found: {eval_list_path}")

    with open(eval_list_path, "r") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
        for name in names:
            eval_list.append((name, "Release"))

    logger.info(f"Loaded {len(eval_list)} garments for category: {args.garment_type}")

    if not eval_list:
        raise ValueError(
            f"No garments found to evaluate for category '{args.garment_type}'."
        )

    # 5. Main Evaluation Loops
    all_garment_metrics = []

    # Init Env with first garment
    first_name, first_stage = eval_list[0]
    env_cfg.garment_name = first_name
    env_cfg.garment_version = first_stage
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.initialize_obs()

    try:
        for garment_idx, (garment_name, garment_stage) in enumerate(eval_list):
            logger.info(
                f"Evaluating: {garment_name} ({garment_stage}) ({garment_idx+1}/{len(eval_list)})"
            )

            # Switch Garment Logic
            if garment_idx > 0:
                if hasattr(env, "switch_garment"):
                    env.switch_garment(garment_name, garment_stage)
                    env.reset()
                    policy.reset()
                else:
                    env.close()
                    env_cfg.garment_name = garment_name
                    env_cfg.garment_version = garment_stage
                    env = gym.make(args.task, cfg=env_cfg).unwrapped
                    env.initialize_obs()
                    policy.reset()

            # Run Loop
            metrics = run_evaluation_loop(
                env=env,
                policy=policy,
                args=args,
                ee_solver=ee_solver,
                is_bimanual=is_bimanual,
                garment_name=garment_name,
                gripper_world_diagnostics=gripper_world_diagnostics,
            )

            all_garment_metrics.append(
                {"garment_name": garment_name, "metrics": metrics}
            )

    finally:
        env.close()

    # Print summary across all garments
    logger.info("=" * 60)
    logger.info("Overall Summary")
    logger.info("=" * 60)

    if all_garment_metrics:
        # Aggregate all episode metrics
        all_episodes = []
        for garment_data in all_garment_metrics:
            for episode_metric in garment_data["metrics"]:
                episode_metric["garment_name"] = garment_data["garment_name"]
                all_episodes.append(episode_metric)

        # Print overall metrics
        calculate_and_print_metrics(all_episodes)

        # Print per-garment summary
        logger.info("=" * 60)
        logger.info("Per-Garment Summary")
        logger.info("=" * 60)
        for garment_data in all_garment_metrics:
            garment_name = garment_data["garment_name"]
            metrics = garment_data["metrics"]
            success_count = sum(1 for m in metrics if m["success"])
            success_rate = success_count / len(metrics) if metrics else 0.0
            avg_return = np.mean([m["return"] for m in metrics]) if metrics else 0.0
            logger.info(
                f"  {garment_name}: Success Rate = {success_rate:.2%}, Avg Return = {avg_return:.2f}"
            )
    else:
        logger.info("No metrics collected (all evaluations failed)")

    logger.info("=" * 60)
    logger.info("Evaluation completed successfully")
    logger.info("=" * 60)
