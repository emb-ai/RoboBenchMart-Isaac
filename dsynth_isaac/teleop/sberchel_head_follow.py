"""Shared head-follow locomotion + XR upper-body helpers for Sberchel teleop scripts."""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

LEFT_HAND_BASE = "left_hand_base_link"
RIGHT_HAND_BASE = "right_hand_base_link"

# Isaac robot body forward is +X (Z-up). OpenXR HMD view direction is local -Z.
ROBOT_FORWARD_LOCAL = (1.0, 0.0, 0.0)
OPENXR_VIEW_FORWARD_LOCAL = (0.0, 0.0, -1.0)

YawMode = Literal["bearing", "headset"]
YawTargetMode = Literal["bearing_then_headset", "bearing", "headset"]
UpperBodyMode = Literal["teleop", "frozen", "idle"]


@dataclass
class FrozenUpperSnapshot:
    """Pelvis-relative wrists and finger joints captured when pursuit begins."""

    left_wrist_in_pelvis: np.ndarray
    right_wrist_in_pelvis: np.ndarray
    fingers: np.ndarray


@dataclass(frozen=True)
class HeadFollowParams:
    """Runtime head-follow and arrival tuning from CLI."""

    follow_lin_gain: float = 1.5
    follow_ang_gain: float = 2.5
    follow_max_lin: float = 0.8
    follow_max_ang: float = 1.5
    follow_pos_deadzone: float = 0.05
    follow_yaw_deadzone: float = 0.05
    follow_yaw_on_gate: float = 0.35
    follow_yaw_off_gate: float = 0.25
    follow_yaw_target: YawTargetMode = "bearing_then_headset"
    follow_head_yaw_offset_deg: float = 0.0
    teleop_while_pursuing: bool = False
    arrive_xy_tol: float = 0.20
    arrive_xy_leave: float = 0.45
    arrive_yaw_tol: float = 0.25
    arrive_yaw_leave: float = 0.50
    print_every_n: int = 30

    @property
    def head_yaw_offset_rad(self) -> float:
        return math.radians(self.follow_head_yaw_offset_deg)

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> HeadFollowParams:
        return cls(
            follow_lin_gain=args.follow_lin_gain,
            follow_ang_gain=args.follow_ang_gain,
            follow_max_lin=args.follow_max_lin,
            follow_max_ang=args.follow_max_ang,
            follow_pos_deadzone=args.follow_pos_deadzone,
            follow_yaw_deadzone=args.follow_yaw_deadzone,
            follow_yaw_on_gate=args.follow_yaw_on_gate,
            follow_yaw_off_gate=args.follow_yaw_off_gate,
            follow_yaw_target=args.follow_yaw_target,
            follow_head_yaw_offset_deg=args.follow_head_yaw_offset_deg,
            teleop_while_pursuing=args.teleop_while_pursuing,
            arrive_xy_tol=args.arrive_xy_tol,
            arrive_xy_leave=args.arrive_xy_leave,
            arrive_yaw_tol=args.arrive_yaw_tol,
            arrive_yaw_leave=args.arrive_yaw_leave,
            print_every_n=args.print_every_n,
        )


def add_head_follow_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register head-follow / arrival CLI flags shared by teleop and record scripts."""
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


def create_sberchel_xr_teleop_interface(
    env_cfg: Any,
    teleop_device_name: str,
    callbacks: dict[str, Callable[[], None]],
    *,
    require_head_tracking: bool = True,
):
    """Create and configure the OpenXR teleop device from env teleop_devices."""
    from isaaclab.devices.retargeter_base import RetargeterBase
    from isaaclab.devices.teleop_device_factory import create_teleop_device

    if not hasattr(env_cfg, "teleop_devices") or teleop_device_name not in env_cfg.teleop_devices.devices:
        raise ValueError(
            f"Teleop device '{teleop_device_name}' not found in env config. "
            f"Available: {list(env_cfg.teleop_devices.devices.keys())}"
        )
    teleop_interface = create_teleop_device(
        teleop_device_name, env_cfg.teleop_devices.devices, callbacks
    )
    if require_head_tracking:
        teleop_interface._required_features.add(RetargeterBase.Requirement.HEAD_TRACKING)
    return teleop_interface


def body_pose_7d(robot, body_name: str) -> list[float]:
    """World-frame pose (pos xyz + quat wxyz) for a named robot body."""
    body_ids, _ = robot.find_bodies(body_name)
    body_id = int(body_ids[0])
    pose = robot.data.body_link_state_w[0, body_id, :7].detach().cpu().tolist()
    return [float(v) for v in pose]


def sync_retargeter_idle_from_robot(teleop_interface, robot) -> None:
    """Set wrist idle poses from the robot's current hand_base frames."""
    left_pose = body_pose_7d(robot, LEFT_HAND_BASE)
    right_pose = body_pose_7d(robot, RIGHT_HAND_BASE)
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


def build_idle_upper_from_robot(robot, device: str) -> torch.Tensor:
    """26D upper body: left/right hand_base world poses + neutral fingers."""
    return torch.tensor(
        [
            *body_pose_7d(robot, LEFT_HAND_BASE),
            *body_pose_7d(robot, RIGHT_HAND_BASE),
            *([0.0] * 12),
        ],
        dtype=torch.float32,
        device=device,
    )


def pelvis_pose_world(robot) -> tuple[torch.Tensor, torch.Tensor]:
    """Pelvis (root) pose in world frame as (t, q) with batch dim N=1."""
    return robot.data.root_pos_w[0:1, :3], robot.data.root_quat_w[0:1, :]


def pose7_to_tensors(pose7) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.tensor(pose7[:3], dtype=torch.float32).unsqueeze(0)
    q = torch.tensor(pose7[3:7], dtype=torch.float32).unsqueeze(0)
    return t, q


def tensors_to_pose7(pos: torch.Tensor, quat: torch.Tensor) -> np.ndarray:
    return np.concatenate(
        [pos[0].detach().cpu().numpy(), quat[0].detach().cpu().numpy()],
        dtype=np.float32,
    )


def world_to_pelvis(
    pelvis_t: torch.Tensor,
    pelvis_q: torch.Tensor,
    world_t: torch.Tensor,
    world_q: torch.Tensor,
) -> np.ndarray:
    import isaaclab.utils.math as math_utils

    local_t, local_q = math_utils.subtract_frame_transforms(
        pelvis_t, pelvis_q, world_t, world_q
    )
    return tensors_to_pose7(local_t, local_q)


def pelvis_to_world(
    pelvis_t: torch.Tensor,
    pelvis_q: torch.Tensor,
    local_t: torch.Tensor,
    local_q: torch.Tensor,
) -> np.ndarray:
    import isaaclab.utils.math as math_utils

    world_t, world_q = math_utils.combine_frame_transforms(
        pelvis_t, pelvis_q, local_t, local_q
    )
    return tensors_to_pose7(world_t, world_q)


def finger_positions_from_robot(robot) -> np.ndarray:
    from dsynth_isaac.tasks.manager_based.sberchel_upperbody_xr_env_cfg import (
        SBERCHEL_HAND_JOINT_NAMES,
    )

    joint_pos = robot.data.joint_pos[0]
    name_to_id = {name: idx for idx, name in enumerate(robot.joint_names)}
    return np.array(
        [
            float(joint_pos[name_to_id[name]].item()) if name in name_to_id else 0.0
            for name in SBERCHEL_HAND_JOINT_NAMES
        ],
        dtype=np.float32,
    )


def capture_frozen_upper_from_robot(robot) -> FrozenUpperSnapshot:
    """Snapshot wrists in pelvis frame and finger joint positions from the robot."""
    pelvis_t, pelvis_q = pelvis_pose_world(robot)
    left_t, left_q = pose7_to_tensors(body_pose_7d(robot, LEFT_HAND_BASE))
    right_t, right_q = pose7_to_tensors(body_pose_7d(robot, RIGHT_HAND_BASE))
    return FrozenUpperSnapshot(
        left_wrist_in_pelvis=world_to_pelvis(pelvis_t, pelvis_q, left_t, left_q),
        right_wrist_in_pelvis=world_to_pelvis(pelvis_t, pelvis_q, right_t, right_q),
        fingers=finger_positions_from_robot(robot),
    )


def capture_frozen_upper_from_teleop(robot, teleop_upper: torch.Tensor) -> FrozenUpperSnapshot:
    """Snapshot pelvis-relative wrists and finger targets from the last teleop command."""
    upper = teleop_upper.detach().cpu().numpy().reshape(-1)
    left_pose7 = upper[0:7]
    right_pose7 = upper[7:14]
    fingers = np.asarray(upper[14:26], dtype=np.float32)
    pelvis_t, pelvis_q = pelvis_pose_world(robot)
    left_t, left_q = pose7_to_tensors(left_pose7)
    right_t, right_q = pose7_to_tensors(right_pose7)
    return FrozenUpperSnapshot(
        left_wrist_in_pelvis=world_to_pelvis(pelvis_t, pelvis_q, left_t, left_q),
        right_wrist_in_pelvis=world_to_pelvis(pelvis_t, pelvis_q, right_t, right_q),
        fingers=fingers,
    )


def capture_frozen_upper(robot, teleop_upper: torch.Tensor | None = None) -> FrozenUpperSnapshot:
    """Prefer last teleop command (grasp intent); fall back to measured robot state."""
    if teleop_upper is not None:
        return capture_frozen_upper_from_teleop(robot, teleop_upper)
    return capture_frozen_upper_from_robot(robot)


def build_upper_from_frozen_snapshot(
    robot, snapshot: FrozenUpperSnapshot, device: str
) -> torch.Tensor:
    """Replay pelvis-relative wrists at the current pelvis pose + saved fingers."""
    pelvis_t, pelvis_q = pelvis_pose_world(robot)
    left_t, left_q = pose7_to_tensors(snapshot.left_wrist_in_pelvis)
    right_t, right_q = pose7_to_tensors(snapshot.right_wrist_in_pelvis)
    left_world = pelvis_to_world(pelvis_t, pelvis_q, left_t, left_q)
    right_world = pelvis_to_world(pelvis_t, pelvis_q, right_t, right_q)
    return torch.tensor(
        [*left_world, *right_world, *snapshot.fingers.tolist()],
        dtype=torch.float32,
        device=device,
    )


def update_loco_arrived(
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


def quat_rotate_vec_wxyz(
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


def heading_from_quat_wxyz(
    quat,
    forward_local: tuple[float, float, float] = ROBOT_FORWARD_LOCAL,
) -> float:
    """World XY heading from quaternion by projecting a body/view forward axis."""
    fx, fy, _fz = quat_rotate_vec_wxyz(quat, forward_local)
    return math.atan2(fy, fx)


def yaw_from_quat_wxyz(quat) -> float:
    """Robot root yaw (rad): heading of body +X in world XY."""
    return heading_from_quat_wxyz(quat, ROBOT_FORWARD_LOCAL)


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def head_yaw_from_pose(head_pose_7d: np.ndarray, *, offset_rad: float = 0.0) -> float:
    """Headset view heading (rad): OpenXR look direction (-Z) projected to world XY."""
    return wrap_angle(
        heading_from_quat_wxyz(head_pose_7d[3:7], OPENXR_VIEW_FORWARD_LOCAL) + offset_rad
    )


def clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def robot_root_state(robot) -> tuple[tuple[float, float, float], float]:
    """Return ((x, y, z), yaw) of env-0 root in world frame."""
    pos = robot.data.root_pos_w[0, :3].detach().cpu().tolist()
    yaw = yaw_from_quat_wxyz(robot.data.root_quat_w[0])
    return (float(pos[0]), float(pos[1]), float(pos[2])), yaw


def update_yaw_mode(
    dist_xy: float, yaw_mode: YawMode, *, yaw_on_gate: float, yaw_off_gate: float
) -> YawMode:
    """Hysteresis: bearing while far, headset yaw when close."""
    if dist_xy >= yaw_on_gate:
        return "bearing"
    if dist_xy <= yaw_off_gate:
        return "headset"
    return yaw_mode


def update_bearing_yaw_active(
    dist_xy: float, bearing_yaw_active: bool, *, yaw_on_gate: float, yaw_off_gate: float
) -> bool:
    """Hysteresis for bearing-only mode: yaw active only while far from the headset."""
    if dist_xy >= yaw_on_gate:
        return True
    if dist_xy <= yaw_off_gate:
        return False
    return bearing_yaw_active


def resolve_yaw_target(
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
        bearing_yaw_active = update_bearing_yaw_active(
            dist_xy, bearing_yaw_active, yaw_on_gate=yaw_on_gate, yaw_off_gate=yaw_off_gate
        )
        return "bearing", bearing_yaw_active, bearing_yaw, bearing_yaw_active
    yaw_mode = update_yaw_mode(
        dist_xy, yaw_mode, yaw_on_gate=yaw_on_gate, yaw_off_gate=yaw_off_gate
    )
    target_yaw = bearing_yaw if yaw_mode == "bearing" else head_yaw
    return yaw_mode, bearing_yaw_active, target_yaw, True


def headset_follow_command(
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
    """Body-frame (vx, vy, wz) driving the base toward the headset world XY."""
    target_x = float(head_pose_7d[0])
    target_y = float(head_pose_7d[1])
    head_yaw = head_yaw_from_pose(head_pose_7d, offset_rad=head_yaw_offset_rad)

    cur_x, cur_y = robot.data.root_pos_w[0, :2].detach().cpu().tolist()
    cur_yaw = yaw_from_quat_wxyz(robot.data.root_quat_w[0])

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
        vx = clip(lin_gain * err_bx, max_lin)
        vy = clip(lin_gain * err_by, max_lin)

    bearing_yaw = math.atan2(err_y, err_x)
    yaw_mode, bearing_yaw_active, target_yaw, yaw_active = resolve_yaw_target(
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
        err_yaw = wrap_angle(target_yaw - cur_yaw)
        wz = 0.0 if abs(err_yaw) < yaw_deadzone else clip(ang_gain * err_yaw, max_ang)
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


def print_debug_block(
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
            head_yaw = head_yaw_from_pose(head_pose)
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
