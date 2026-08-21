"""VR teleop for Sberchel: upper body via OpenXR, legs via locomotion policy (station-keeping).

The ONNX RL policy from green_challenge balances the robot while a P-controller
commands body-frame velocity toward the spawn pose (so drift is corrected);
XR hand tracking drives the arms and fingers through Pink IK.

Usage:
    python scripts/teleop_sberchel.py \\
        --config_dir_path generated_envs/isaac_sample \\
        --teleop_device handtracking \\
        --enable_pinocchio
"""

import argparse
import logging
import math
import os
import sys
from collections.abc import Callable

from isaaclab.app import AppLauncher

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from dsynth_isaac.green_challenge_path import ensure_green_challenge_importable  # noqa: E402
from dsynth_isaac.sim_defaults import apply_sberchel_cli_defaults  # noqa: E402

ensure_green_challenge_importable()

parser = argparse.ArgumentParser(description="Sberchel XR teleop with policy-balanced legs.")
parser.add_argument("--config_dir_path", type=str, required=True, help="Path to generated scene config directory.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Dsynth-Fetch-Any-Item-Sberchel-XR-v0", help="Gym task id.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="handtracking",
    help="Teleop device key from env teleop_devices (e.g. handtracking, manusvive).",
)
parser.add_argument("--target_item_name_regex", type=str, default=".*Nestle Honey.*", help="Fetch target regex.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio (required for Pink IK / hand tracking).",
)
parser.add_argument(
    "--hold_lin_gain",
    type=float,
    default=1.5,
    help="P-gain for XY station-keeping (m/s per meter of error).",
)
parser.add_argument(
    "--hold_ang_gain",
    type=float,
    default=2.0,
    help="P-gain for yaw station-keeping (rad/s per radian of error).",
)
parser.add_argument(
    "--hold_max_lin",
    type=float,
    default=0.8,
    help="Max commanded linear speed for station-keeping (m/s).",
)
parser.add_argument(
    "--hold_max_ang",
    type=float,
    default=1.0,
    help="Max commanded yaw rate for station-keeping (rad/s).",
)
parser.add_argument(
    "--hold_pos_deadzone",
    type=float,
    default=0.05,
    help="XY error (m) below which linear velocity command is zero.",
)
parser.add_argument(
    "--hold_yaw_deadzone",
    type=float,
    default=0.05,
    help="Yaw error (rad) below which angular velocity command is zero.",
)
AppLauncher.add_app_launcher_args(parser)
# Sberchel runs on green_challenge's CPU physics by default; --device overrides.
apply_sberchel_cli_defaults(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
_use_handtracking = "handtracking" in args_cli.teleop_device.lower()
if args_cli.enable_pinocchio or _use_handtracking:
    import pinocchio  # noqa: F401

if _use_handtracking:
    try:
        import dex_retargeting  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Sberchel hand-tracking teleop requires dex-retargeting. "
            "Install with: pip install 'dex-retargeting==0.4.6'"
        ) from exc

if _use_handtracking:
    app_launcher_args["xr"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from dsynth_isaac.robots.sberchel_camera_cfg import strip_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import parse_env_cfg

sys.path.append(".")
import dsynth_isaac.tasks.manager_based  # noqa: F401

from dsynth_isaac.tasks.manager_based.sberchel_base_env_cfg import (
    LEGS_ACTION_DIM,
    policy_leg_targets,
)
from green_challenge.policy.network import Network

logger = logging.getLogger(__name__)

_LEFT_HAND_BASE = "left_hand_base_link"
_RIGHT_HAND_BASE = "right_hand_base_link"


def _body_pose_7d(robot, body_name: str) -> list[float]:
    """World-frame pose (pos xyz + quat wxyz) for a named robot body."""
    body_ids, _ = robot.find_bodies(body_name)
    body_id = int(body_ids[0])
    pose = robot.data.body_link_state_w[0, body_id, :7].detach().cpu().tolist()
    return [float(v) for v in pose]


def _sync_retargeter_idle_from_robot(teleop_interface, robot) -> None:
    """Set wrist idle poses from the robot's current hand_base frames."""
    left_pose = _body_pose_7d(robot, _LEFT_HAND_BASE)
    right_pose = _body_pose_7d(robot, _RIGHT_HAND_BASE)
    for retargeter in getattr(teleop_interface, "_retargeters", []):
        if hasattr(retargeter, "_idle_left_wrist"):
            retargeter._idle_left_wrist = np.asarray(left_pose, dtype=np.float32)
            retargeter._idle_right_wrist = np.asarray(right_pose, dtype=np.float32)
        recalibrate = getattr(retargeter, "recalibrate_wrists", None)
        if callable(recalibrate):
            recalibrate()
    print(
        f">>> wrist idle left={left_pose[:3]} right={right_pose[:3]}",
        flush=True,
    )


def _yaw_from_quat_wxyz(quat) -> float:
    """Yaw (rad) from a wxyz quaternion (scalar or length-4 tensor/array)."""
    if hasattr(quat, "detach"):
        q = quat.detach().cpu().tolist()
    else:
        q = list(quat)
    w, x, y, z = (float(v) for v in q)
    # yaw from wxyz; matches Isaac Lab / robotics convention (Z-up).
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _capture_hold_pose(robot) -> tuple[float, float, float]:
    """Return (x, y, yaw) of env-0 root in world frame."""
    pos = robot.data.root_pos_w[0, :2].detach().cpu().tolist()
    yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w[0])
    return float(pos[0]), float(pos[1]), yaw


def _station_keep_command(
    robot,
    hold_xy_yaw: tuple[float, float, float],
    *,
    lin_gain: float,
    ang_gain: float,
    max_lin: float,
    max_ang: float,
    pos_deadzone: float,
    yaw_deadzone: float,
) -> tuple[float, float, float]:
    """Body-frame (vx, vy, wz) that drives the base back to the hold pose.

    Linear commands are proportional to the world-frame XY error, rotated into
    the robot body frame. Angular command restores yaw toward the hold yaw.
    """
    hold_x, hold_y, hold_yaw = hold_xy_yaw
    cur_x, cur_y = robot.data.root_pos_w[0, :2].detach().cpu().tolist()
    cur_yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w[0])

    err_x = hold_x - cur_x
    err_y = hold_y - cur_y
    err_yaw = _wrap_angle(hold_yaw - cur_yaw)

    # World XY error -> body frame (vx forward, vy left).
    c = math.cos(cur_yaw)
    s = math.sin(cur_yaw)
    err_bx = c * err_x + s * err_y
    err_by = -s * err_x + c * err_y

    dist = math.hypot(err_bx, err_by)
    if dist < pos_deadzone:
        vx = 0.0
        vy = 0.0
    else:
        vx = _clip(lin_gain * err_bx, max_lin)
        vy = _clip(lin_gain * err_by, max_lin)

    wz = 0.0 if abs(err_yaw) < yaw_deadzone else _clip(ang_gain * err_yaw, max_ang)
    return vx, vy, wz


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(f"Expected ManagerBasedRLEnvCfg, got {type(env_cfg).__name__}")

    env_cfg.seed = args_cli.seed
    env_cfg.terminations.time_out = None
    env_cfg.scene.scene_config_dir_path = args_cli.config_dir_path
    env_cfg.target_item_name_regex = args_cli.target_item_name_regex

    if args_cli.xr:
        env_cfg = strip_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset(seed=args_cli.seed)

    net = Network(env, type="zest", device="cpu", closed_kinematics=True)
    net.reset()
    robot = env.scene["robot"]
    hold_xy_yaw = _capture_hold_pose(robot)
    upper_body_dim = env.action_manager.total_action_dim - LEGS_ACTION_DIM
    print(
        f">>> action dim = {env.action_manager.total_action_dim} "
        f"(legs {LEGS_ACTION_DIM} + upper_body {upper_body_dim})",
        flush=True,
    )
    print(
        f">>> station-keep pose (x, y, yaw) = "
        f"({hold_xy_yaw[0]:.3f}, {hold_xy_yaw[1]:.3f}, {hold_xy_yaw[2]:.3f})",
        flush=True,
    )

    should_reset = False
    teleoperation_active = not args_cli.xr

    def reset_env() -> None:
        nonlocal should_reset
        should_reset = True
        print("Reset triggered - environment will reset on next step")

    def start_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = True
        _sync_retargeter_idle_from_robot(teleop_interface, robot)
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_env,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_env,
    }

    if not hasattr(env_cfg, "teleop_devices") or args_cli.teleop_device not in env_cfg.teleop_devices.devices:
        raise ValueError(
            f"Teleop device '{args_cli.teleop_device}' not found in env config. "
            f"Available: {list(env_cfg.teleop_devices.devices.keys())}"
        )

    teleop_interface = create_teleop_device(
        args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
    )
    teleop_interface.reset()
    _sync_retargeter_idle_from_robot(teleop_interface, robot)
    idle_upper = torch.tensor(
        [
            *_body_pose_7d(robot, _LEFT_HAND_BASE),
            *_body_pose_7d(robot, _RIGHT_HAND_BASE),
            *([0.0] * 12),
        ],
        dtype=torch.float32,
        device=env.device,
    )

    print(f"Using teleop device: {teleop_interface}")
    print("Legs: locomotion policy station-keeping toward spawn pose. Upper body: XR teleop.")
    if args_cli.xr:
        print("Press START in XR to begin teleoperation. Press R to reset.")

    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                vx, vy, wz = _station_keep_command(
                    robot,
                    hold_xy_yaw,
                    lin_gain=args_cli.hold_lin_gain,
                    ang_gain=args_cli.hold_ang_gain,
                    max_lin=args_cli.hold_max_lin,
                    max_ang=args_cli.hold_max_ang,
                    pos_deadzone=args_cli.hold_pos_deadzone,
                    yaw_deadzone=args_cli.hold_yaw_deadzone,
                )
                legs = policy_leg_targets(net, vx, vy, wz)
                legs_tensor = torch.as_tensor(legs, dtype=torch.float32, device=env.device)
                legs_tensor = legs_tensor.unsqueeze(0).repeat(env.num_envs, 1)

                if teleoperation_active:
                    upper_body = teleop_interface.advance()
                    if upper_body.ndim == 1:
                        upper_body = upper_body.unsqueeze(0)
                    upper_body = upper_body.to(env.device).repeat(env.num_envs, 1)
                    actions = torch.cat([legs_tensor, upper_body], dim=-1)
                    env.step(actions)
                else:
                    actions = torch.cat(
                        [legs_tensor, idle_upper.unsqueeze(0).repeat(env.num_envs, 1)],
                        dim=-1,
                    )
                    env.step(actions)

                if should_reset:
                    env.reset()
                    net.reset()
                    hold_xy_yaw = _capture_hold_pose(robot)
                    teleop_interface.reset()
                    _sync_retargeter_idle_from_robot(teleop_interface, robot)
                    idle_upper = torch.tensor(
                        [
                            *_body_pose_7d(robot, _LEFT_HAND_BASE),
                            *_body_pose_7d(robot, _RIGHT_HAND_BASE),
                            *([0.0] * 12),
                        ],
                        dtype=torch.float32,
                        device=env.device,
                    )
                    should_reset = False
                    print(
                        "Environment reset complete; "
                        f"station-keep pose = ({hold_xy_yaw[0]:.3f}, {hold_xy_yaw[1]:.3f}, {hold_xy_yaw[2]:.3f})"
                    )
        except Exception as exc:
            logger.error("Error during simulation step: %s", exc)
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
