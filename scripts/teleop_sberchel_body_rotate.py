"""VR teleop for Sberchel: upper body via OpenXR, in-place base rotation from torso intent.

The ONNX RL policy from green_challenge balances the robot. A P-controller
holds XY position (drift correction) while yaw is driven only when the user's
hand plane has rotated beyond a threshold (~20 deg) from START and the headset faces
the same direction as that plane. Head glances alone should not spin the base.

Usage:
    python scripts/teleop_sberchel_body_rotate.py \\
        --config_dir_path generated_envs/isaac_sample \\
        --teleop_device handtracking \\
        --enable_pinocchio
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from collections.abc import Callable

from isaaclab.app import AppLauncher

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from dsynth_isaac.green_challenge_path import ensure_green_challenge_importable  # noqa: E402

ensure_green_challenge_importable()

parser = argparse.ArgumentParser(
    description="Sberchel XR teleop with upper-body-triggered in-place rotation."
)
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
    help="P-gain for in-place body-yaw tracking (rad/s per radian of error).",
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
    help="Max commanded yaw rate for in-place rotation (rad/s).",
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
parser.add_argument(
    "--body_yaw_on_deg",
    type=float,
    default=20.0,
    help="Activate in-place rotation when |hand-plane delta| exceeds this (deg).",
)
parser.add_argument(
    "--body_yaw_off_deg",
    type=float,
    default=15.0,
    help="Deactivate in-place rotation when |hand-plane delta| drops below this (deg).",
)
parser.add_argument(
    "--body_yaw_recenter_deg",
    type=float,
    default=5.0,
    help="Recenter hand-plane reference when neutral and below this (deg).",
)
parser.add_argument(
    "--body_align_max_deg",
    type=float,
    default=25.0,
    help="Max |head_yaw - plane_yaw| (deg) to treat headset and hand plane as aligned.",
)
parser.add_argument(
    "--print_every_n",
    type=int,
    default=30,
    help="Print debug block every N simulation steps (0 to disable periodic prints).",
)
AppLauncher.add_app_launcher_args(parser)
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

from isaaclab.devices.device_base import DeviceBase
from dsynth_isaac.robots.sberchel_camera_cfg import strip_camera_configs
from isaaclab.devices.retargeter_base import RetargeterBase
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import parse_env_cfg

sys.path.append(".")
import dsynth_isaac.tasks.manager_based  # noqa: F401

from dsynth_isaac.robots.sberchel_cfg import apply_sberchel_xr_teleop_pd_gains
from dsynth_isaac.tasks.manager_based.sberchel_base_env_cfg import (
    LEGS_ACTION_DIM,
    policy_leg_targets,
)
from green_challenge.policy.network import Network

logger = logging.getLogger(__name__)

_LEFT_HAND_BASE = "left_hand_base_link"
_RIGHT_HAND_BASE = "right_hand_base_link"
_DEBUG_LOG_PATH = "/home/soshin/repo/RoboBenchMart-Isaac-Private/.cursor/debug-cc4d33.log"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": "cc4d33",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        pass
    # #endregion


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
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _circular_mean_yaw(angles: list[float]) -> float:
    """Mean heading on the circle (rad)."""
    if not angles:
        return 0.0
    sin_sum = sum(math.sin(a) for a in angles)
    cos_sum = sum(math.cos(a) for a in angles)
    return math.atan2(sin_sum, cos_sum)


def _estimate_plane_yaw(left_wrist_7d: np.ndarray, right_wrist_7d: np.ndarray) -> float:
    """Estimate hand-plane facing yaw from both wrists (no head quaternion)."""
    left_yaw = _yaw_from_quat_wxyz(left_wrist_7d[3:7])
    right_yaw = _yaw_from_quat_wxyz(right_wrist_7d[3:7])
    shoulder_yaw = math.atan2(
        float(right_wrist_7d[1]) - float(left_wrist_7d[1]),
        float(right_wrist_7d[0]) - float(left_wrist_7d[0]),
    )
    forward_yaw = _wrap_angle(shoulder_yaw - math.pi / 2.0)
    return _circular_mean_yaw([forward_yaw, left_yaw, right_yaw])


def _plane_yaw_forward_candidates(
    left_wrist_7d: np.ndarray, right_wrist_7d: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Return plane_yaw, forward_plus, forward_minus, left_yaw, right_yaw for debug."""
    left_yaw = _yaw_from_quat_wxyz(left_wrist_7d[3:7])
    right_yaw = _yaw_from_quat_wxyz(right_wrist_7d[3:7])
    shoulder_yaw = math.atan2(
        float(right_wrist_7d[1]) - float(left_wrist_7d[1]),
        float(right_wrist_7d[0]) - float(left_wrist_7d[0]),
    )
    forward_plus = _wrap_angle(shoulder_yaw + math.pi / 2.0)
    forward_minus = _wrap_angle(shoulder_yaw - math.pi / 2.0)
    plane_yaw = _circular_mean_yaw([forward_plus, left_yaw, right_yaw])
    plane_yaw_alt = _circular_mean_yaw([forward_minus, left_yaw, right_yaw])
    return plane_yaw, plane_yaw_alt, forward_plus, forward_minus, shoulder_yaw


def _head_plane_aligned(head_yaw: float, plane_yaw: float, *, align_max_rad: float) -> tuple[bool, float]:
    """True when headset facing matches hand-plane facing within tolerance."""
    align_err = abs(_wrap_angle(head_yaw - plane_yaw))
    return align_err <= align_max_rad, align_err


def _capture_hold_xy(robot) -> tuple[float, float]:
    """Return (x, y) of env-0 root in world frame."""
    pos = robot.data.root_pos_w[0, :2].detach().cpu().tolist()
    return float(pos[0]), float(pos[1])


def _robot_root_state(robot) -> tuple[tuple[float, float, float], float]:
    """Return ((x, y, z), yaw) of env-0 root in world frame."""
    pos = robot.data.root_pos_w[0, :3].detach().cpu().tolist()
    yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w[0])
    return (float(pos[0]), float(pos[1]), float(pos[2])), yaw


def _station_keep_xy(
    robot,
    hold_xy: tuple[float, float],
    *,
    lin_gain: float,
    max_lin: float,
    pos_deadzone: float,
) -> tuple[float, float, float, float, float]:
    """Body-frame (vx, vy) driving XY back to hold position.

    Returns:
        (vx, vy, err_x, err_y, dist_xy)
    """
    hold_x, hold_y = hold_xy
    cur_x, cur_y = robot.data.root_pos_w[0, :2].detach().cpu().tolist()
    cur_yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w[0])

    err_x = hold_x - cur_x
    err_y = hold_y - cur_y

    c = math.cos(cur_yaw)
    s = math.sin(cur_yaw)
    err_bx = c * err_x + s * err_y
    err_by = -s * err_x + c * err_y

    dist_xy = math.hypot(err_bx, err_by)
    if dist_xy < pos_deadzone:
        return 0.0, 0.0, err_x, err_y, dist_xy

    vx = _clip(lin_gain * err_bx, max_lin)
    vy = _clip(lin_gain * err_by, max_lin)
    return vx, vy, err_x, err_y, dist_xy


def _update_in_place_yaw_active(
    plane_delta: float, active: bool, *, on_rad: float, off_rad: float
) -> bool:
    """Hysteresis for hand-plane rotation intent."""
    if abs(plane_delta) >= on_rad:
        return True
    if abs(plane_delta) <= off_rad:
        return False
    return active


def _body_rotate_wz(
    robot,
    target_yaw: float,
    *,
    rotate_allowed: bool,
    ang_gain: float,
    max_ang: float,
    yaw_deadzone: float,
) -> tuple[float, float]:
    """In-place yaw command toward target facing when rotation is allowed."""
    if not rotate_allowed:
        return 0.0, 0.0
    cur_yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w[0])
    err_yaw = _wrap_angle(target_yaw - cur_yaw)
    if abs(err_yaw) < yaw_deadzone:
        return 0.0, err_yaw
    return _clip(ang_gain * err_yaw, max_ang), err_yaw


def _print_debug_block(
    step: int,
    *,
    active: bool,
    robot_pos: tuple[float, float, float],
    robot_yaw: float,
    plane_yaw: float | None,
    plane_delta: float | None,
    head_yaw: float | None,
    align_err: float | None,
    aligned: bool,
    in_place_active: bool,
    rotate_allowed: bool,
    err_x: float,
    err_y: float,
    dist_xy: float,
    err_yaw: float,
    vx: float,
    vy: float,
    wz: float,
) -> None:
    plane_yaw_str = f"{plane_yaw:.3f}" if plane_yaw is not None else "n/a"
    plane_delta_str = f"{plane_delta:.3f}" if plane_delta is not None else "n/a"
    head_yaw_str = f"{head_yaw:.3f}" if head_yaw is not None else "n/a"
    align_err_str = f"{align_err:.3f}" if align_err is not None else "n/a"
    lines = [
        f"[body_rotate] step={step} active={active}",
        f"  robot   pos=({robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}) yaw={robot_yaw:.3f}",
        f"  plane   yaw={plane_yaw_str} delta={plane_delta_str}",
        f"  head    yaw={head_yaw_str} align_err={align_err_str} aligned={aligned}",
        f"  gate    in_place={in_place_active} rotate_allowed={rotate_allowed}",
        f"  hold    dxy=({err_x:.3f}, {err_y:.3f}) dist={dist_xy:.3f} dyaw={err_yaw:.3f}",
        f"  cmd     vx={vx:.3f} vy={vy:.3f} wz={wz:.3f}",
    ]
    print("\n".join(lines), flush=True)


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

    net = Network(env, type="zest", device="cpu", closed_kinematics=False)
    net.reset()
    apply_sberchel_xr_teleop_pd_gains(env.scene["robot"])
    robot = env.scene["robot"]
    hold_xy = _capture_hold_xy(robot)
    upper_body_dim = env.action_manager.total_action_dim - LEGS_ACTION_DIM
    print(
        f">>> action dim = {env.action_manager.total_action_dim} "
        f"(legs {LEGS_ACTION_DIM} + upper_body {upper_body_dim})",
        flush=True,
    )
    print(f">>> station-keep XY = ({hold_xy[0]:.3f}, {hold_xy[1]:.3f})", flush=True)

    plane_yaw_on_rad = math.radians(args_cli.body_yaw_on_deg)
    plane_yaw_off_rad = math.radians(args_cli.body_yaw_off_deg)
    plane_yaw_recenter_rad = math.radians(args_cli.body_yaw_recenter_deg)
    body_align_max_rad = math.radians(args_cli.body_align_max_deg)

    should_reset = False
    teleoperation_active = not args_cli.xr
    step_count = 0
    first_active_logged = False
    need_body_calib = False
    ref_plane_yaw: float | None = None
    ref_head_yaw: float | None = None
    ref_robot_yaw: float | None = None
    in_place_yaw_active = False
    rotate_allowed = False

    def reset_env() -> None:
        nonlocal should_reset
        should_reset = True
        print("Reset triggered - environment will reset on next step")

    def start_teleoperation() -> None:
        nonlocal teleoperation_active, first_active_logged, need_body_calib
        nonlocal ref_plane_yaw, ref_head_yaw, ref_robot_yaw, in_place_yaw_active, rotate_allowed, hold_xy
        teleoperation_active = True
        first_active_logged = False
        need_body_calib = True
        ref_plane_yaw = None
        ref_head_yaw = None
        ref_robot_yaw = None
        in_place_yaw_active = False
        rotate_allowed = False
        hold_xy = _capture_hold_xy(robot)
        _sync_retargeter_idle_from_robot(teleop_interface, robot)
        print("Teleoperation activated (upper body + in-place body-yaw rotation)")

    def stop_teleoperation() -> None:
        nonlocal teleoperation_active, in_place_yaw_active, rotate_allowed
        teleoperation_active = False
        in_place_yaw_active = False
        rotate_allowed = False
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
    teleop_interface._required_features.add(RetargeterBase.Requirement.HEAD_TRACKING)
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
    print(
        "Legs: XY station-keep + in-place rotation when hand plane turns "
        f">= {args_cli.body_yaw_on_deg:.0f} deg and headset aligns (<= "
        f"{args_cli.body_align_max_deg:.0f} deg). Upper body: XR teleop."
    )
    if args_cli.xr:
        print("Press START in XR to begin teleoperation. Press R to reset.")

    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                step_count += 1
                robot_pos, robot_yaw = _robot_root_state(robot)
                plane_yaw: float | None = None
                plane_delta: float | None = None
                head_yaw: float | None = None
                head_delta: float | None = None
                target_robot_yaw: float | None = None
                align_err: float | None = None
                aligned = False
                rotate_allowed = False
                err_yaw = 0.0

                vx, vy, err_x, err_y, dist_xy = _station_keep_xy(
                    robot,
                    hold_xy,
                    lin_gain=args_cli.hold_lin_gain,
                    max_lin=args_cli.hold_max_lin,
                    pos_deadzone=args_cli.hold_pos_deadzone,
                )
                wz = 0.0

                if teleoperation_active:
                    raw = teleop_interface._get_raw_data()
                    head_pose = raw.get(DeviceBase.TrackingTarget.HEAD)
                    left_wrist = raw.get(DeviceBase.TrackingTarget.HAND_LEFT, {}).get("wrist")
                    right_wrist = raw.get(DeviceBase.TrackingTarget.HAND_RIGHT, {}).get("wrist")

                    if head_pose is not None and left_wrist is not None and right_wrist is not None:
                        plane_yaw = _estimate_plane_yaw(left_wrist, right_wrist)
                        head_yaw = _yaw_from_quat_wxyz(head_pose[3:7])
                        aligned, align_err = _head_plane_aligned(
                            head_yaw, plane_yaw, align_max_rad=body_align_max_rad
                        )
                        if (
                            need_body_calib
                            or ref_plane_yaw is None
                            or ref_head_yaw is None
                            or ref_robot_yaw is None
                        ):
                            ref_plane_yaw = plane_yaw
                            ref_head_yaw = head_yaw
                            ref_robot_yaw = robot_yaw
                            in_place_yaw_active = False
                            rotate_allowed = False
                            need_body_calib = False
                            print(
                                f">>> rotation reference calibrated: plane={ref_plane_yaw:.3f} rad, "
                                f"head={ref_head_yaw:.3f} rad, robot={ref_robot_yaw:.3f} rad",
                                flush=True,
                            )
                        else:
                            plane_delta = _wrap_angle(plane_yaw - ref_plane_yaw)
                            head_delta = _wrap_angle(head_yaw - ref_head_yaw)
                            in_place_yaw_active = _update_in_place_yaw_active(
                                abs(plane_delta),
                                in_place_yaw_active,
                                on_rad=plane_yaw_on_rad,
                                off_rad=plane_yaw_off_rad,
                            )
                            if not in_place_yaw_active and abs(plane_delta) < plane_yaw_recenter_rad:
                                ref_plane_yaw = plane_yaw
                                ref_head_yaw = head_yaw
                                ref_robot_yaw = robot_yaw
                                plane_delta = 0.0
                                head_delta = 0.0
                            rotate_allowed = in_place_yaw_active and aligned
                            target_robot_yaw = _wrap_angle(ref_robot_yaw + plane_delta)
                            wz, err_yaw = _body_rotate_wz(
                                robot,
                                target_robot_yaw,
                                rotate_allowed=rotate_allowed,
                                ang_gain=args_cli.hold_ang_gain,
                                max_ang=args_cli.hold_max_ang,
                                yaw_deadzone=args_cli.hold_yaw_deadzone,
                            )
                            if rotate_allowed or in_place_yaw_active or step_count % max(
                                args_cli.print_every_n, 1
                            ) == 0:
                                (
                                    _plane_yaw,
                                    plane_yaw_alt,
                                    forward_plus,
                                    forward_minus,
                                    shoulder_yaw,
                                ) = _plane_yaw_forward_candidates(left_wrist, right_wrist)
                                _debug_log(
                                    "B",
                                    "teleop_sberchel_body_rotate.py:gate",
                                    "rotation_gate",
                                    {
                                        "step": step_count,
                                        "plane_delta": plane_delta,
                                        "head_delta": head_delta,
                                        "target_robot_yaw": target_robot_yaw,
                                        "ref_robot_yaw": ref_robot_yaw,
                                        "aligned": aligned,
                                        "align_err": align_err,
                                        "in_place": in_place_yaw_active,
                                        "rotate_allowed": rotate_allowed,
                                        "robot_yaw": robot_yaw,
                                        "plane_yaw": plane_yaw,
                                        "head_yaw": head_yaw,
                                        "err_yaw": err_yaw,
                                        "wz": wz,
                                        "shoulder_yaw": shoulder_yaw,
                                    },
                                )

                    upper_body = torch.cat(
                        [r.retarget(raw) for r in teleop_interface._retargeters],
                        dim=-1,
                    )
                    if upper_body.ndim == 1:
                        upper_body = upper_body.unsqueeze(0)
                    upper_body = upper_body.to(env.device).repeat(env.num_envs, 1)
                else:
                    upper_body = idle_upper.unsqueeze(0).repeat(env.num_envs, 1)

                legs = policy_leg_targets(net, vx, vy, wz)
                legs_tensor = torch.as_tensor(legs, dtype=torch.float32, device=env.device)
                legs_tensor = legs_tensor.unsqueeze(0).repeat(env.num_envs, 1)
                actions = torch.cat([legs_tensor, upper_body], dim=-1)
                env.step(actions)

                should_print = False
                if args_cli.print_every_n > 0:
                    if teleoperation_active and not first_active_logged:
                        should_print = True
                        first_active_logged = True
                    elif step_count % args_cli.print_every_n == 0:
                        should_print = True
                if should_print:
                    _print_debug_block(
                        step_count,
                        active=teleoperation_active,
                        robot_pos=robot_pos,
                        robot_yaw=robot_yaw,
                        plane_yaw=plane_yaw,
                        plane_delta=plane_delta,
                        head_yaw=head_yaw,
                        align_err=align_err,
                        aligned=aligned,
                        in_place_active=in_place_yaw_active,
                        rotate_allowed=rotate_allowed,
                        err_x=err_x,
                        err_y=err_y,
                        dist_xy=dist_xy,
                        err_yaw=err_yaw,
                        vx=vx,
                        vy=vy,
                        wz=wz,
                    )

                if should_reset:
                    env.reset()
                    net.reset()
                    hold_xy = _capture_hold_xy(robot)
                    apply_sberchel_xr_teleop_pd_gains(robot)
                    teleop_interface.reset()
                    ref_plane_yaw = None
                    ref_head_yaw = None
                    ref_robot_yaw = None
                    in_place_yaw_active = False
                    rotate_allowed = False
                    need_body_calib = teleoperation_active
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
                    first_active_logged = False
                    print(
                        "Environment reset complete; "
                        f"station-keep XY = ({hold_xy[0]:.3f}, {hold_xy[1]:.3f})"
                    )
        except Exception as exc:
            logger.error("Error during simulation step: %s", exc)
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
