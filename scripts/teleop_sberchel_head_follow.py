"""VR teleop for Sberchel: upper body via OpenXR, legs follow headset pose.

The ONNX RL policy from green_challenge balances the robot while a P-controller
commands body-frame velocity toward the OpenXR headset world XY position and yaw;
XR hand tracking drives the arms and fingers through Pink IK.

Head-follow locomotion activates only after pressing START in XR (same gate as
upper-body teleop).

Usage:
    python scripts/teleop_sberchel_head_follow.py \\
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
from dataclasses import dataclass
from typing import Literal

from isaaclab.app import AppLauncher

_DEFAULT_GC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "green_challenge"))
_GC = os.environ.get("GREEN_CHALLENGE_PATH", _DEFAULT_GC)
_gc_parent = os.path.dirname(os.path.abspath(_GC))
if _gc_parent not in sys.path:
    sys.path.insert(0, _gc_parent)

parser = argparse.ArgumentParser(description="Sberchel XR teleop with headset-follow locomotion.")
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
    "--follow_lin_gain",
    type=float,
    default=1.5,
    help="P-gain for XY head-follow (m/s per meter of error).",
)
parser.add_argument(
    "--follow_ang_gain",
    type=float,
    default=2.5,
    help="P-gain for yaw head-follow (rad/s per radian of error).",
)
parser.add_argument(
    "--follow_max_lin",
    type=float,
    default=0.8,
    help="Max commanded linear speed for head-follow (m/s).",
)
parser.add_argument(
    "--follow_max_ang",
    type=float,
    default=1.5,
    help="Max commanded yaw rate for head-follow (rad/s).",
)
parser.add_argument(
    "--follow_pos_deadzone",
    type=float,
    default=0.05,
    help="XY error (m) below which linear velocity command is zero.",
)
parser.add_argument(
    "--follow_yaw_deadzone",
    type=float,
    default=0.05,
    help="Yaw error (rad) below which angular velocity command is zero.",
)
parser.add_argument(
    "--follow_yaw_on_gate",
    type=float,
    default=0.35,
    help="Use bearing-yaw tracking when XY distance to headset exceeds this (m).",
)
parser.add_argument(
    "--follow_yaw_off_gate",
    type=float,
    default=0.25,
    help="Switch to headset-yaw tracking when XY distance drops below this (m). "
    "Hysteresis with --follow_yaw_on_gate avoids mode chatter at the boundary.",
)
parser.add_argument(
    "--follow_yaw_target",
    type=str,
    default="bearing_then_headset",
    choices=("bearing_then_headset", "bearing", "headset"),
    help="Yaw target: bearing while far then headset yaw (default), bearing only, or headset only.",
)
parser.add_argument(
    "--follow_head_yaw_offset_deg",
    type=float,
    default=0.0,
    help="Extra heading offset applied to headset view yaw (deg). Tune if still misaligned.",
)
parser.add_argument(
    "--teleop_while_pursuing",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Allow XR hand teleop before robot matches headset XY+yaw (default: freeze until arrived).",
)
parser.add_argument(
    "--arrive_xy_tol",
    type=float,
    default=0.20,
    help="Enter arrived (enable XR teleop) when XY distance to headset is at or below this (m).",
)
parser.add_argument(
    "--arrive_xy_leave",
    type=float,
    default=0.45,
    help="Leave arrived when XY distance exceeds this (m). Must be >= --arrive_xy_tol.",
)
parser.add_argument(
    "--arrive_yaw_tol",
    type=float,
    default=0.25,
    help="Enter arrived when |headset yaw - robot yaw| is at or below this (rad).",
)
parser.add_argument(
    "--arrive_yaw_leave",
    type=float,
    default=0.50,
    help="Leave arrived when |headset yaw - robot yaw| exceeds this (rad). "
    "Must be >= --arrive_yaw_tol.",
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
import isaaclab.utils.math as math_utils
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
from dsynth_isaac.tasks.manager_based.sberchel_upperbody_xr_env_cfg import (
    SBERCHEL_HAND_JOINT_NAMES,
)
from green_challenge.policy.network import Network

logger = logging.getLogger(__name__)

_LEFT_HAND_BASE = "left_hand_base_link"
_RIGHT_HAND_BASE = "right_hand_base_link"

# Isaac robot body forward is +X (Z-up). OpenXR HMD view direction is local -Z.
_ROBOT_FORWARD_LOCAL = (1.0, 0.0, 0.0)
_OPENXR_VIEW_FORWARD_LOCAL = (0.0, 0.0, -1.0)

YawMode = Literal["bearing", "headset"]
YawTargetMode = Literal["bearing_then_headset", "bearing", "headset"]
UpperBodyMode = Literal["teleop", "frozen", "idle"]


@dataclass
class _FrozenUpperSnapshot:
    """Pelvis-relative wrists and finger joints captured when pursuit begins."""

    left_wrist_in_pelvis: np.ndarray
    right_wrist_in_pelvis: np.ndarray
    fingers: np.ndarray


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


def _build_idle_upper_from_robot(robot, device: str) -> torch.Tensor:
    """26D upper body: left/right hand_base world poses + neutral fingers."""
    return torch.tensor(
        [
            *_body_pose_7d(robot, _LEFT_HAND_BASE),
            *_body_pose_7d(robot, _RIGHT_HAND_BASE),
            *([0.0] * 12),
        ],
        dtype=torch.float32,
        device=device,
    )


def _pelvis_pose_world(robot) -> tuple[torch.Tensor, torch.Tensor]:
    """Pelvis (root) pose in world frame as (t, q) with batch dim N=1."""
    return robot.data.root_pos_w[0:1, :3], robot.data.root_quat_w[0:1, :]


def _pose7_to_tensors(pose7) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.tensor(pose7[:3], dtype=torch.float32).unsqueeze(0)
    q = torch.tensor(pose7[3:7], dtype=torch.float32).unsqueeze(0)
    return t, q


def _tensors_to_pose7(pos: torch.Tensor, quat: torch.Tensor) -> np.ndarray:
    return np.concatenate(
        [pos[0].detach().cpu().numpy(), quat[0].detach().cpu().numpy()],
        dtype=np.float32,
    )


def _world_to_pelvis(
    pelvis_t: torch.Tensor,
    pelvis_q: torch.Tensor,
    world_t: torch.Tensor,
    world_q: torch.Tensor,
) -> np.ndarray:
    local_t, local_q = math_utils.subtract_frame_transforms(
        pelvis_t, pelvis_q, world_t, world_q
    )
    return _tensors_to_pose7(local_t, local_q)


def _pelvis_to_world(
    pelvis_t: torch.Tensor,
    pelvis_q: torch.Tensor,
    local_t: torch.Tensor,
    local_q: torch.Tensor,
) -> np.ndarray:
    world_t, world_q = math_utils.combine_frame_transforms(
        pelvis_t, pelvis_q, local_t, local_q
    )
    return _tensors_to_pose7(world_t, world_q)


def _finger_positions_from_robot(robot) -> np.ndarray:
    joint_pos = robot.data.joint_pos[0]
    name_to_id = {name: idx for idx, name in enumerate(robot.joint_names)}
    return np.array(
        [
            float(joint_pos[name_to_id[name]].item()) if name in name_to_id else 0.0
            for name in SBERCHEL_HAND_JOINT_NAMES
        ],
        dtype=np.float32,
    )


def _capture_frozen_upper_from_robot(robot) -> _FrozenUpperSnapshot:
    """Snapshot wrists in pelvis frame and finger joint positions from the robot."""
    pelvis_t, pelvis_q = _pelvis_pose_world(robot)
    left_t, left_q = _pose7_to_tensors(_body_pose_7d(robot, _LEFT_HAND_BASE))
    right_t, right_q = _pose7_to_tensors(_body_pose_7d(robot, _RIGHT_HAND_BASE))
    return _FrozenUpperSnapshot(
        left_wrist_in_pelvis=_world_to_pelvis(pelvis_t, pelvis_q, left_t, left_q),
        right_wrist_in_pelvis=_world_to_pelvis(pelvis_t, pelvis_q, right_t, right_q),
        fingers=_finger_positions_from_robot(robot),
    )


def _capture_frozen_upper_from_teleop(
    robot, teleop_upper: torch.Tensor
) -> _FrozenUpperSnapshot:
    """Snapshot pelvis-relative wrists and finger targets from the last teleop command."""
    upper = teleop_upper.detach().cpu().numpy().reshape(-1)
    left_pose7 = upper[0:7]
    right_pose7 = upper[7:14]
    fingers = np.asarray(upper[14:26], dtype=np.float32)
    pelvis_t, pelvis_q = _pelvis_pose_world(robot)
    left_t, left_q = _pose7_to_tensors(left_pose7)
    right_t, right_q = _pose7_to_tensors(right_pose7)
    return _FrozenUpperSnapshot(
        left_wrist_in_pelvis=_world_to_pelvis(pelvis_t, pelvis_q, left_t, left_q),
        right_wrist_in_pelvis=_world_to_pelvis(pelvis_t, pelvis_q, right_t, right_q),
        fingers=fingers,
    )


def _capture_frozen_upper(
    robot, teleop_upper: torch.Tensor | None = None
) -> _FrozenUpperSnapshot:
    """Prefer last teleop command (grasp intent); fall back to measured robot state."""
    if teleop_upper is not None:
        return _capture_frozen_upper_from_teleop(robot, teleop_upper)
    return _capture_frozen_upper_from_robot(robot)


def _build_upper_from_frozen_snapshot(
    robot, snapshot: _FrozenUpperSnapshot, device: str
) -> torch.Tensor:
    """Replay pelvis-relative wrists at the current pelvis pose + saved fingers."""
    pelvis_t, pelvis_q = _pelvis_pose_world(robot)
    left_t, left_q = _pose7_to_tensors(snapshot.left_wrist_in_pelvis)
    right_t, right_q = _pose7_to_tensors(snapshot.right_wrist_in_pelvis)
    left_world = _pelvis_to_world(pelvis_t, pelvis_q, left_t, left_q)
    right_world = _pelvis_to_world(pelvis_t, pelvis_q, right_t, right_q)
    return torch.tensor(
        [*left_world, *right_world, *snapshot.fingers.tolist()],
        dtype=torch.float32,
        device=device,
    )


def _update_loco_arrived(
    dist_xy: float,
    yaw_err: float,
    loco_arrived: bool,
    *,
    dist_arrive: float,
    dist_leave: float,
    yaw_arrive: float,
    yaw_leave: float,
) -> bool:
    """Hysteresis: arrived when XY and yaw match headset; leave when either drifts."""
    abs_yaw = abs(yaw_err)
    if not loco_arrived:
        return dist_xy <= dist_arrive and abs_yaw <= yaw_arrive
    if dist_xy >= dist_leave or abs_yaw > yaw_leave:
        return False
    return True


def _quat_rotate_vec_wxyz(
    quat, vec: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Rotate a vector by a wxyz quaternion."""
    if hasattr(quat, "detach"):
        q = quat.detach().cpu().tolist()
    else:
        q = list(quat)
    w, x, y, z = (float(v) for v in q)
    vx, vy, vz = vec
    qxv_x = y * vz - z * vy
    qxv_y = z * vx - x * vz
    qxv_z = x * vy - y * vx
    qxqv_x = y * qxv_z - z * qxv_y
    qxqv_y = z * qxv_x - x * qxv_z
    qxqv_z = x * qxv_y - y * qxv_x
    return (
        vx + 2.0 * (w * qxv_x + qxqv_x),
        vy + 2.0 * (w * qxv_y + qxqv_y),
        vz + 2.0 * (w * qxv_z + qxqv_z),
    )


def _heading_from_quat_wxyz(
    quat,
    forward_local: tuple[float, float, float] = _ROBOT_FORWARD_LOCAL,
) -> float:
    """World XY heading from quaternion by projecting a body/view forward axis."""
    fx, fy, _fz = _quat_rotate_vec_wxyz(quat, forward_local)
    return math.atan2(fy, fx)


def _yaw_from_quat_wxyz(quat) -> float:
    """Robot root yaw (rad): heading of body +X in world XY."""
    return _heading_from_quat_wxyz(quat, _ROBOT_FORWARD_LOCAL)


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _head_yaw_from_pose(head_pose_7d: np.ndarray, *, offset_rad: float = 0.0) -> float:
    """Headset view heading (rad): OpenXR look direction (-Z) projected to world XY."""
    return _wrap_angle(
        _heading_from_quat_wxyz(head_pose_7d[3:7], _OPENXR_VIEW_FORWARD_LOCAL) + offset_rad
    )


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _robot_root_state(robot) -> tuple[tuple[float, float, float], float]:
    """Return ((x, y, z), yaw) of env-0 root in world frame."""
    pos = robot.data.root_pos_w[0, :3].detach().cpu().tolist()
    yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w[0])
    return (float(pos[0]), float(pos[1]), float(pos[2])), yaw


def _update_yaw_mode(
    dist_xy: float, yaw_mode: YawMode, *, yaw_on_gate: float, yaw_off_gate: float
) -> YawMode:
    """Hysteresis: bearing while far, headset yaw when close."""
    if dist_xy >= yaw_on_gate:
        return "bearing"
    if dist_xy <= yaw_off_gate:
        return "headset"
    return yaw_mode


def _update_bearing_yaw_active(
    dist_xy: float, bearing_yaw_active: bool, *, yaw_on_gate: float, yaw_off_gate: float
) -> bool:
    """Hysteresis for bearing-only mode: yaw active only while far from the headset."""
    if dist_xy >= yaw_on_gate:
        return True
    if dist_xy <= yaw_off_gate:
        return False
    return bearing_yaw_active


def _resolve_yaw_target(
    yaw_target_mode: YawTargetMode,
    dist_xy: float,
    yaw_mode: YawMode,
    bearing_yaw_active: bool,
    bearing_yaw: float,
    head_yaw: float,
    *,
    yaw_on_gate: float,
    yaw_off_gate: float,
) -> tuple[YawMode, bool, float, bool]:
    """Return (updated_yaw_mode, bearing_yaw_active, target_yaw, yaw_active)."""
    if yaw_target_mode == "headset":
        return "headset", bearing_yaw_active, head_yaw, True
    if yaw_target_mode == "bearing":
        bearing_yaw_active = _update_bearing_yaw_active(
            dist_xy, bearing_yaw_active, yaw_on_gate=yaw_on_gate, yaw_off_gate=yaw_off_gate
        )
        return "bearing", bearing_yaw_active, bearing_yaw, bearing_yaw_active
    yaw_mode = _update_yaw_mode(
        dist_xy, yaw_mode, yaw_on_gate=yaw_on_gate, yaw_off_gate=yaw_off_gate
    )
    target_yaw = bearing_yaw if yaw_mode == "bearing" else head_yaw
    return yaw_mode, bearing_yaw_active, target_yaw, True


def _headset_follow_command(
    robot,
    head_pose_7d: np.ndarray,
    *,
    yaw_mode: YawMode,
    bearing_yaw_active: bool,
    yaw_target_mode: YawTargetMode,
    yaw_on_gate: float,
    yaw_off_gate: float,
    lin_gain: float,
    ang_gain: float,
    max_lin: float,
    max_ang: float,
    pos_deadzone: float,
    yaw_deadzone: float,
    head_yaw_offset_rad: float,
) -> tuple[float, float, float, float, float, float, float, float, float, YawMode, float, bool, bool]:
    """Body-frame (vx, vy, wz) driving the base toward the headset world XY.

    Yaw tracks bearing toward headset XY while far, then headset orientation when
    close (``bearing_then_headset``). ``yaw_mode`` uses distance hysteresis.

    Returns:
        (vx, vy, wz, err_x, err_y, err_yaw, dist_xy, head_yaw, bearing_yaw,
         yaw_mode, target_yaw, yaw_active, bearing_yaw_active)
    """
    target_x = float(head_pose_7d[0])
    target_y = float(head_pose_7d[1])
    head_yaw = _head_yaw_from_pose(head_pose_7d, offset_rad=head_yaw_offset_rad)

    cur_x, cur_y = robot.data.root_pos_w[0, :2].detach().cpu().tolist()
    cur_yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w[0])

    err_x = target_x - cur_x
    err_y = target_y - cur_y

    c = math.cos(cur_yaw)
    s = math.sin(cur_yaw)
    err_bx = c * err_x + s * err_y
    err_by = -s * err_x + c * err_y

    dist_xy = math.hypot(err_bx, err_by)

    if dist_xy < pos_deadzone:
        vx = 0.0
        vy = 0.0
    else:
        vx = _clip(lin_gain * err_bx, max_lin)
        vy = _clip(lin_gain * err_by, max_lin)

    bearing_yaw = math.atan2(err_y, err_x)
    yaw_mode, bearing_yaw_active, target_yaw, yaw_active = _resolve_yaw_target(
        yaw_target_mode,
        dist_xy,
        yaw_mode,
        bearing_yaw_active,
        bearing_yaw,
        head_yaw,
        yaw_on_gate=yaw_on_gate,
        yaw_off_gate=yaw_off_gate,
    )
    if not yaw_active:
        err_yaw = 0.0
        wz = 0.0
    else:
        err_yaw = _wrap_angle(target_yaw - cur_yaw)
        wz = 0.0 if abs(err_yaw) < yaw_deadzone else _clip(ang_gain * err_yaw, max_ang)
    return (
        vx,
        vy,
        wz,
        err_x,
        err_y,
        err_yaw,
        dist_xy,
        head_yaw,
        bearing_yaw,
        yaw_mode,
        target_yaw,
        yaw_active,
        bearing_yaw_active,
    )


def _print_debug_block(
    step: int,
    *,
    active: bool,
    robot_pos: tuple[float, float, float],
    robot_yaw: float,
    head_pose: np.ndarray | None,
    err_x: float,
    err_y: float,
    err_yaw: float,
    dist_xy: float,
    vx: float,
    vy: float,
    wz: float,
    bearing_yaw: float | None = None,
    head_yaw: float | None = None,
    target_yaw: float | None = None,
    yaw_mode: YawMode | None = None,
    yaw_active: bool | None = None,
    upper_body_mode: UpperBodyMode | None = None,
    loco_arrived: bool | None = None,
) -> None:
    lines = [
        f"[head_follow] step={step} active={active}",
        f"  robot   pos=({robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}) yaw={robot_yaw:.3f}",
    ]
    if head_pose is not None:
        if head_yaw is None:
            head_yaw = _head_yaw_from_pose(head_pose)
        lines.append(
            f"  headset pos=({float(head_pose[0]):.3f}, {float(head_pose[1]):.3f}, "
            f"{float(head_pose[2]):.3f}) yaw={head_yaw:.3f}"
        )
    else:
        lines.append("  headset unavailable")
    bearing_str = f"{bearing_yaw:.3f}" if bearing_yaw is not None else "n/a"
    head_yaw_str = f"{head_yaw:.3f}" if head_yaw is not None else "n/a"
    target_yaw_str = f"{target_yaw:.3f}" if target_yaw is not None else "n/a"
    yaw_mode_str = yaw_mode if yaw_mode is not None else "n/a"
    yaw_gate_str = str(yaw_active) if yaw_active is not None else "n/a"
    upper_body_str = upper_body_mode if upper_body_mode is not None else "n/a"
    loco_str = str(loco_arrived) if loco_arrived is not None else "n/a"
    lines.extend([
        f"  yaw     mode={yaw_mode_str} target={target_yaw_str} bearing={bearing_str} "
        f"head={head_yaw_str} yaw_active={yaw_gate_str}",
        f"  upper   body={upper_body_str} loco_arrived={loco_str}",
        f"  error   dxy=({err_x:.3f}, {err_y:.3f}) dist={dist_xy:.3f} dyaw={err_yaw:.3f}",
        f"  cmd     vx={vx:.3f} vy={vy:.3f} wz={wz:.3f}",
    ])
    print("\n".join(lines), flush=True)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(f"Expected ManagerBasedRLEnvCfg, got {type(env_cfg).__name__}")

    env_cfg.seed = args_cli.seed
    env_cfg.terminations.time_out = None
    # env_cfg.scene.scene_config_dir_path = args_cli.config_dir_path
    # env_cfg.target_item_name_regex = args_cli.target_item_name_regex

    if args_cli.xr:
        env_cfg = strip_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset(seed=args_cli.seed)

    net = Network(env, type="zest", device="cpu", closed_kinematics=False)
    net.reset()
    apply_sberchel_xr_teleop_pd_gains(env.scene["robot"])
    robot = env.scene["robot"]
    upper_body_dim = env.action_manager.total_action_dim - LEGS_ACTION_DIM
    print(
        f">>> action dim = {env.action_manager.total_action_dim} "
        f"(legs {LEGS_ACTION_DIM} + upper_body {upper_body_dim})",
        flush=True,
    )

    should_reset = False
    teleoperation_active = not args_cli.xr
    step_count = 0
    first_active_logged = False
    yaw_mode: YawMode = "bearing"
    bearing_yaw_active = False
    loco_arrived = False
    prev_loco_arrived = False
    frozen_snapshot: _FrozenUpperSnapshot | None = None
    last_teleop_upper: torch.Tensor | None = None

    def reset_env() -> None:
        nonlocal should_reset
        should_reset = True
        print("Reset triggered - environment will reset on next step")

    def start_teleoperation() -> None:
        nonlocal teleoperation_active, first_active_logged, yaw_mode, bearing_yaw_active
        nonlocal loco_arrived, prev_loco_arrived, frozen_snapshot, last_teleop_upper
        teleoperation_active = True
        first_active_logged = False
        yaw_mode = "bearing"
        bearing_yaw_active = False
        loco_arrived = False
        prev_loco_arrived = False
        frozen_snapshot = None
        last_teleop_upper = None
        _sync_retargeter_idle_from_robot(teleop_interface, robot)
        print("Teleoperation activated (upper body + head-follow locomotion)")

    def stop_teleoperation() -> None:
        nonlocal teleoperation_active, yaw_mode, bearing_yaw_active
        nonlocal loco_arrived, prev_loco_arrived, frozen_snapshot, last_teleop_upper
        teleoperation_active = False
        yaw_mode = "bearing"
        bearing_yaw_active = False
        loco_arrived = False
        prev_loco_arrived = False
        frozen_snapshot = None
        last_teleop_upper = None
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
    idle_upper = _build_idle_upper_from_robot(robot, env.device)

    print(f"Using teleop device: {teleop_interface}")
    print(
        "Legs: locomotion policy follows headset after START. "
        "Upper body: XR teleop when robot matches headset XY+yaw; "
        "arms/fingers frozen body-relative while pursuing."
    )
    head_yaw_offset_rad = math.radians(args_cli.follow_head_yaw_offset_deg)
    if args_cli.xr:
        print("Press START in XR to begin teleoperation and head-follow. Press R to reset.")

    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                step_count += 1
                robot_pos, robot_yaw = _robot_root_state(robot)
                head_pose = None
                err_x = err_y = err_yaw = dist_xy = 0.0
                vx = vy = wz = 0.0
                bearing_yaw: float | None = None
                head_yaw: float | None = None
                target_yaw: float | None = None
                yaw_mode_dbg: YawMode | None = None
                yaw_active: bool | None = None
                upper_body_mode: UpperBodyMode | None = None
                loco_arrived_dbg: bool | None = None

                if teleoperation_active:
                    raw = teleop_interface._get_raw_data()
                    head_pose = raw.get(DeviceBase.TrackingTarget.HEAD)
                    if head_pose is not None:
                        (
                            vx,
                            vy,
                            wz,
                            err_x,
                            err_y,
                            err_yaw,
                            dist_xy,
                            head_yaw,
                            bearing_yaw,
                            yaw_mode,
                            target_yaw,
                            yaw_active,
                            bearing_yaw_active,
                        ) = _headset_follow_command(
                            robot,
                            head_pose,
                            yaw_mode=yaw_mode,
                            bearing_yaw_active=bearing_yaw_active,
                            yaw_target_mode=args_cli.follow_yaw_target,
                            yaw_on_gate=args_cli.follow_yaw_on_gate,
                            yaw_off_gate=args_cli.follow_yaw_off_gate,
                            lin_gain=args_cli.follow_lin_gain,
                            ang_gain=args_cli.follow_ang_gain,
                            max_lin=args_cli.follow_max_lin,
                            max_ang=args_cli.follow_max_ang,
                            pos_deadzone=args_cli.follow_pos_deadzone,
                            yaw_deadzone=args_cli.follow_yaw_deadzone,
                            head_yaw_offset_rad=head_yaw_offset_rad,
                        )
                        yaw_mode_dbg = yaw_mode
                        alignment_yaw_err = _wrap_angle(head_yaw - robot_yaw)
                        new_loco_arrived = _update_loco_arrived(
                            dist_xy,
                            alignment_yaw_err,
                            loco_arrived,
                            dist_arrive=args_cli.arrive_xy_tol,
                            dist_leave=args_cli.arrive_xy_leave,
                            yaw_arrive=args_cli.arrive_yaw_tol,
                            yaw_leave=args_cli.arrive_yaw_leave,
                        )
                        if prev_loco_arrived and not new_loco_arrived:
                            frozen_snapshot = _capture_frozen_upper(robot, last_teleop_upper)
                        elif not prev_loco_arrived and new_loco_arrived:
                            _sync_retargeter_idle_from_robot(teleop_interface, robot)
                            frozen_snapshot = None
                        loco_arrived = new_loco_arrived
                        prev_loco_arrived = loco_arrived
                        loco_arrived_dbg = loco_arrived

                    use_teleop = args_cli.teleop_while_pursuing or loco_arrived
                    if use_teleop:
                        upper_body = torch.cat(
                            [r.retarget(raw) for r in teleop_interface._retargeters],
                            dim=-1,
                        )
                        last_teleop_upper = (
                            upper_body.squeeze(0).clone()
                            if upper_body.ndim > 1
                            else upper_body.clone()
                        )
                        upper_body_mode = "teleop"
                    else:
                        if frozen_snapshot is None:
                            frozen_snapshot = _capture_frozen_upper(robot, last_teleop_upper)
                        upper_body = _build_upper_from_frozen_snapshot(
                            robot, frozen_snapshot, env.device
                        )
                        upper_body_mode = "frozen"
                    if upper_body.ndim == 1:
                        upper_body = upper_body.unsqueeze(0)
                    upper_body = upper_body.to(env.device).repeat(env.num_envs, 1)
                else:
                    upper_body = idle_upper.unsqueeze(0).repeat(env.num_envs, 1)
                    upper_body_mode = "idle"

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
                        head_pose=head_pose,
                        err_x=err_x,
                        err_y=err_y,
                        err_yaw=err_yaw,
                        dist_xy=dist_xy,
                        vx=vx,
                        vy=vy,
                        wz=wz,
                        bearing_yaw=bearing_yaw,
                        head_yaw=head_yaw,
                        target_yaw=target_yaw,
                        yaw_mode=yaw_mode_dbg,
                        yaw_active=yaw_active,
                        upper_body_mode=upper_body_mode,
                        loco_arrived=loco_arrived_dbg,
                    )

                if should_reset:
                    env.reset()
                    net.reset()
                    apply_sberchel_xr_teleop_pd_gains(robot)
                    teleop_interface.reset()
                    yaw_mode = "bearing"
                    bearing_yaw_active = False
                    loco_arrived = False
                    prev_loco_arrived = False
                    frozen_snapshot = None
                    last_teleop_upper = None
                    _sync_retargeter_idle_from_robot(teleop_interface, robot)
                    idle_upper = _build_idle_upper_from_robot(robot, env.device)
                    should_reset = False
                    first_active_logged = False
                    print("Environment reset complete")
        except Exception as exc:
            logger.error("Error during simulation step: %s", exc)
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
