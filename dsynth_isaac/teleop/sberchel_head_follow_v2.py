"""Standing / Pursuing head-follow locomotion for Sberchel XR teleop (v2).

Standing: station-keep to a latched robot pose; live XR upper-body teleop.
Pursuing: P-control toward filtered headset XY; wz always tracks filtered headset view yaw.
Upper body freezes during Pursuing (pelvis-relative snapshot from last teleop command).

No bearing yaw, short/long pursuit, or arrival-gated teleop (see v1 module for those).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from dsynth_isaac.teleop.sberchel_head_follow import (
    FrozenUpperSnapshot,
    UpperBodyMode,
    clip,
    head_yaw_from_pose,
    station_keep_command,
    wrap_angle,
    yaw_from_quat_wxyz,
)

LocoMode = Literal["standing", "pursuing"]


@dataclass(frozen=True)
class HeadFollowV2Params:
    """Runtime v2 head-follow tuning from CLI."""

    head_filter_pos_alpha: float = 0.05
    head_filter_quat_alpha: float = 0.05
    head_filter_max_pos_step: float = 0.012
    head_filter_max_yaw_step: float = 0.035
    pursue_pos_on: float = 0.15
    pursue_pos_off: float = 0.08
    pursue_yaw_on: float = 0.20
    pursue_yaw_off: float = 0.12
    arrive_robot_head_xy: float = 0.25
    arrive_robot_head_yaw: float = 0.35
    follow_lin_gain: float = 1.5
    follow_ang_gain: float = 2.5
    follow_max_lin: float = 0.8
    follow_max_ang: float = 1.5
    follow_pos_deadzone: float = 0.05
    follow_yaw_deadzone: float = 0.05
    follow_head_yaw_offset_deg: float = 0.0
    print_every_n: int = 30

    @property
    def head_yaw_offset_rad(self) -> float:
        return math.radians(self.follow_head_yaw_offset_deg)

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> HeadFollowV2Params:
        return cls(
            head_filter_pos_alpha=args.head_filter_pos_alpha,
            head_filter_quat_alpha=args.head_filter_quat_alpha,
            head_filter_max_pos_step=args.head_filter_max_pos_step,
            head_filter_max_yaw_step=args.head_filter_max_yaw_step,
            pursue_pos_on=args.pursue_pos_on,
            pursue_pos_off=args.pursue_pos_off,
            pursue_yaw_on=args.pursue_yaw_on,
            pursue_yaw_off=args.pursue_yaw_off,
            arrive_robot_head_xy=args.arrive_robot_head_xy,
            arrive_robot_head_yaw=args.arrive_robot_head_yaw,
            follow_lin_gain=args.follow_lin_gain,
            follow_ang_gain=args.follow_ang_gain,
            follow_max_lin=args.follow_max_lin,
            follow_max_ang=args.follow_max_ang,
            follow_pos_deadzone=args.follow_pos_deadzone,
            follow_yaw_deadzone=args.follow_yaw_deadzone,
            follow_head_yaw_offset_deg=args.follow_head_yaw_offset_deg,
            print_every_n=args.print_every_n,
        )


def add_head_follow_v2_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register v2 Standing/Pursuing head-follow CLI flags."""
    parser.add_argument(
        "--head_filter_pos_alpha",
        type=float,
        default=0.05,
        help="EMA weight for filtered headset XY (higher = more responsive).",
    )
    parser.add_argument(
        "--head_filter_quat_alpha",
        type=float,
        default=0.05,
        help="Slerp step toward raw headset orientation each step.",
    )
    parser.add_argument(
        "--head_filter_max_pos_step",
        type=float,
        default=0.012,
        help="Max filtered headset XY displacement per sim step (m). "
        "Caps response to fast head translations.",
    )
    parser.add_argument(
        "--head_filter_max_yaw_step",
        type=float,
        default=0.035,
        help="Max filtered headset yaw change per sim step (rad). "
        "Caps response to fast head rotations.",
    )
    parser.add_argument(
        "--pursue_pos_on",
        type=float,
        default=0.15,
        help="Enter Pursuing when headset XY drift from head_ref exceeds this (m). "
        "Standing entry uses raw headset; exit uses filtered.",
    )
    parser.add_argument(
        "--pursue_pos_off",
        type=float,
        default=0.08,
        help="Return to Standing when filtered XY drift drops below this (m). "
        "Must be < --pursue_pos_on.",
    )
    parser.add_argument(
        "--pursue_yaw_on",
        type=float,
        default=0.20,
        help="Enter Pursuing when headset yaw drift from head_ref exceeds this (rad). "
        "Standing entry uses raw headset; exit uses filtered.",
    )
    parser.add_argument(
        "--pursue_yaw_off",
        type=float,
        default=0.12,
        help="Return to Standing when filtered yaw drift drops below this (rad). "
        "Must be < --pursue_yaw_on.",
    )
    parser.add_argument(
        "--arrive_robot_head_xy",
        type=float,
        default=0.25,
        help="Return to Standing when robot XY is within this distance of filtered headset (m).",
    )
    parser.add_argument(
        "--arrive_robot_head_yaw",
        type=float,
        default=0.35,
        help="Return to Standing when |headset yaw - robot yaw| is below this (rad).",
    )
    parser.add_argument(
        "--follow_lin_gain",
        type=float,
        default=1.5,
        help="P-gain for XY locomotion (m/s per meter of error).",
    )
    parser.add_argument(
        "--follow_ang_gain",
        type=float,
        default=2.5,
        help="P-gain for yaw locomotion (rad/s per radian of error).",
    )
    parser.add_argument(
        "--follow_max_lin",
        type=float,
        default=0.8,
        help="Max commanded linear speed (m/s).",
    )
    parser.add_argument(
        "--follow_max_ang",
        type=float,
        default=1.5,
        help="Max commanded yaw rate (rad/s).",
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
        "--follow_head_yaw_offset_deg",
        type=float,
        default=0.0,
        help="Extra heading offset applied to headset view yaw (deg).",
    )
    parser.add_argument(
        "--print_every_n",
        type=int,
        default=30,
        help="Print debug block every N simulation steps (0 to disable periodic prints).",
    )


def _quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / norm


def _quat_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation between unit quaternions (wxyz)."""
    q0 = _quat_normalize_wxyz(q0.astype(np.float64))
    q1 = _quat_normalize_wxyz(q1.astype(np.float64))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return _quat_normalize_wxyz(out.astype(np.float32))
    theta_0 = math.acos(min(1.0, max(-1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    out = s0 * q0 + s1 * q1
    return _quat_normalize_wxyz(out.astype(np.float32))


def _quat_angle_wxyz(q0: np.ndarray, q1: np.ndarray) -> float:
    """Geodesic angle (rad) between unit quaternions (wxyz)."""
    q0 = _quat_normalize_wxyz(q0.astype(np.float64))
    q1 = _quat_normalize_wxyz(q1.astype(np.float64))
    dot = abs(float(np.dot(q0, q1)))
    return 2.0 * math.acos(min(1.0, dot))


def _quat_step_limit_wxyz(
    q0: np.ndarray, q1: np.ndarray, max_angle: float
) -> np.ndarray:
    """Slerp from q0 toward q1, limiting rotation to max_angle (rad)."""
    if max_angle <= 0.0:
        return _quat_normalize_wxyz(q0.astype(np.float32))
    angle = _quat_angle_wxyz(q0, q1)
    if angle <= max_angle:
        return _quat_normalize_wxyz(q1.astype(np.float32))
    return _quat_slerp_wxyz(q0, q1, max_angle / angle)


class HeadsetPoseFilter:
    """EMA position + slerp orientation filter for OpenXR HEAD pose (7D wxyz).

    Per-step XY and yaw rate limits suppress pursuit triggers from fast head flicks.
    """

    def __init__(
        self,
        pos_alpha: float,
        quat_alpha: float,
        max_pos_step: float = 0.012,
        max_yaw_step: float = 0.035,
    ) -> None:
        self._pos_alpha = pos_alpha
        self._quat_alpha = quat_alpha
        self._max_pos_step = max_pos_step
        self._max_yaw_step = max_yaw_step
        self._pose: np.ndarray | None = None

    def reset(self, pose_7d: np.ndarray) -> np.ndarray:
        self._pose = np.asarray(pose_7d, dtype=np.float32).reshape(7).copy()
        self._pose[3:7] = _quat_normalize_wxyz(self._pose[3:7])
        return self._pose.copy()

    def _apply_xy_step_limit(
        self, target_xy: tuple[float, float]
    ) -> tuple[float, float]:
        if self._pose is None or self._max_pos_step <= 0.0:
            return target_xy
        dx = target_xy[0] - float(self._pose[0])
        dy = target_xy[1] - float(self._pose[1])
        dist_xy = math.hypot(dx, dy)
        if dist_xy <= self._max_pos_step:
            return target_xy
        scale = self._max_pos_step / dist_xy
        return (
            float(self._pose[0]) + dx * scale,
            float(self._pose[1]) + dy * scale,
        )

    def update(self, raw_pose_7d: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_pose_7d, dtype=np.float32).reshape(7)
        if self._pose is None:
            return self.reset(raw)
        pa = self._pos_alpha
        target_x = (1.0 - pa) * self._pose[0] + pa * raw[0]
        target_y = (1.0 - pa) * self._pose[1] + pa * raw[1]
        target_x, target_y = self._apply_xy_step_limit((target_x, target_y))
        self._pose[0] = target_x
        self._pose[1] = target_y
        self._pose[2] = (1.0 - pa) * self._pose[2] + pa * raw[2]
        quat_target = _quat_slerp_wxyz(
            self._pose[3:7], raw[3:7], self._quat_alpha
        )
        self._pose[3:7] = _quat_step_limit_wxyz(
            self._pose[3:7], quat_target, self._max_yaw_step
        )
        return self._pose.copy()

    @property
    def pose(self) -> np.ndarray | None:
        if self._pose is None:
            return None
        return self._pose.copy()


def capture_standing_latch(robot) -> tuple[float, float, float]:
    """Latch robot root (x, y, yaw) for Standing station-keep."""
    pos = robot.data.root_pos_w[0, :2].detach().cpu().tolist()
    yaw = yaw_from_quat_wxyz(robot.data.root_quat_w[0])
    return float(pos[0]), float(pos[1]), yaw


def capture_head_ref_from_pose(
    pose_7d: np.ndarray, *, offset_rad: float = 0.0
) -> tuple[float, float, float]:
    """Reference headset (x, y, yaw) for pursuit threshold comparison."""
    return (
        float(pose_7d[0]),
        float(pose_7d[1]),
        head_yaw_from_pose(pose_7d, offset_rad=offset_rad),
    )


def head_ref_delta(
    filtered_pose_7d: np.ndarray,
    head_ref_xy_yaw: tuple[float, float, float],
    *,
    offset_rad: float = 0.0,
) -> tuple[float, float]:
    """Return (xy_distance, yaw_error) between filtered pose and head reference."""
    ref_x, ref_y, ref_yaw = head_ref_xy_yaw
    dx = float(filtered_pose_7d[0]) - ref_x
    dy = float(filtered_pose_7d[1]) - ref_y
    dist_xy = math.hypot(dx, dy)
    head_yaw = head_yaw_from_pose(filtered_pose_7d, offset_rad=offset_rad)
    yaw_err = wrap_angle(head_yaw - ref_yaw)
    return dist_xy, yaw_err


def update_loco_mode(
    mode: LocoMode,
    filtered_pose_7d: np.ndarray,
    head_ref_xy_yaw: tuple[float, float, float],
    *,
    pos_on: float,
    pos_off: float,
    yaw_on: float,
    yaw_off: float,
    offset_rad: float = 0.0,
    entry_pose_7d: np.ndarray | None = None,
    robot_xy: tuple[float, float] | None = None,
    robot_yaw: float | None = None,
    arrive_robot_head_xy: float = 0.25,
    arrive_robot_head_yaw: float = 0.35,
) -> LocoMode:
    """Hysteresis: Standing <-> Pursuing based on headset vs head_ref.

    Standing -> Pursuing uses ``entry_pose_7d`` when provided (typically raw
    headset) so upper-body freeze happens before filter lag. Pursuing exit and
    arrive checks still use the filtered pose for stability.

    Pursuing also exits when the robot reaches the filtered headset pose
    (XY distance and yaw alignment), even if the head did not return to head_ref.
    """
    entry_pose = filtered_pose_7d if entry_pose_7d is None else entry_pose_7d
    entry_dist_xy, entry_yaw_err = head_ref_delta(
        entry_pose, head_ref_xy_yaw, offset_rad=offset_rad
    )
    filt_dist_xy, filt_yaw_err = head_ref_delta(
        filtered_pose_7d, head_ref_xy_yaw, offset_rad=offset_rad
    )
    if mode == "standing":
        if entry_dist_xy >= pos_on or abs(entry_yaw_err) >= yaw_on:
            return "pursuing"
        return "standing"

    if filt_dist_xy < pos_off and abs(filt_yaw_err) < yaw_off:
        return "standing"

    if robot_xy is not None and robot_yaw is not None:
        robot_head_dist = math.hypot(
            float(filtered_pose_7d[0]) - robot_xy[0],
            float(filtered_pose_7d[1]) - robot_xy[1],
        )
        head_yaw = head_yaw_from_pose(filtered_pose_7d, offset_rad=offset_rad)
        align_yaw_err = wrap_angle(head_yaw - robot_yaw)
        if (
            robot_head_dist < arrive_robot_head_xy
            and abs(align_yaw_err) < arrive_robot_head_yaw
        ):
            return "standing"

    return "pursuing"


def pursuing_command(
    robot,
    filtered_pose_7d: np.ndarray,
    *,
    lin_gain: float,
    ang_gain: float,
    max_lin: float,
    max_ang: float,
    pos_deadzone: float,
    yaw_deadzone: float,
    head_yaw_offset_rad: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Body-frame (vx, vy, wz) toward filtered headset XY and headset view yaw.

    Returns:
        (vx, vy, wz, err_x, err_y, err_yaw, dist_xy, head_yaw)
    """
    target_x = float(filtered_pose_7d[0])
    target_y = float(filtered_pose_7d[1])
    head_yaw = head_yaw_from_pose(filtered_pose_7d, offset_rad=head_yaw_offset_rad)

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

    err_yaw = wrap_angle(head_yaw - cur_yaw)
    wz = 0.0 if abs(err_yaw) < yaw_deadzone else clip(ang_gain * err_yaw, max_ang)

    return vx, vy, wz, err_x, err_y, err_yaw, dist_xy, head_yaw


@dataclass
class HeadFollowV2Runtime:
    """Mutable runtime state for v2 Standing/Pursuing locomotion."""

    filter: HeadsetPoseFilter
    mode: LocoMode = "standing"
    standing_latch_xy_yaw: tuple[float, float, float] | None = None
    head_ref_xy_yaw: tuple[float, float, float] | None = None
    frozen_snapshot: FrozenUpperSnapshot | None = None
    last_teleop_upper: torch.Tensor | None = None

    @classmethod
    def from_params(cls, params: HeadFollowV2Params) -> HeadFollowV2Runtime:
        return cls(
            filter=HeadsetPoseFilter(
                params.head_filter_pos_alpha,
                params.head_filter_quat_alpha,
                params.head_filter_max_pos_step,
                params.head_filter_max_yaw_step,
            ),
        )

    def clear_session(self) -> None:
        """Clear mode and snapshots (STOP); keep filter coefficients."""
        self.mode = "standing"
        self.standing_latch_xy_yaw = None
        self.head_ref_xy_yaw = None
        self.frozen_snapshot = None
        self.last_teleop_upper = None
        self.filter._pose = None

    def begin_standing(
        self,
        robot,
        filtered_pose_7d: np.ndarray,
        *,
        offset_rad: float,
        latch_robot: bool = True,
    ) -> None:
        """Enter Standing: latch robot pose (optional) and set head reference."""
        if latch_robot or self.standing_latch_xy_yaw is None:
            self.standing_latch_xy_yaw = capture_standing_latch(robot)
        self.head_ref_xy_yaw = capture_head_ref_from_pose(
            filtered_pose_7d, offset_rad=offset_rad
        )
        self.mode = "standing"
        self.frozen_snapshot = None

    def on_enter_standing_from_pursuing(
        self,
        robot,
        filtered_pose_7d: np.ndarray,
        *,
        offset_rad: float,
    ) -> None:
        """Pursuing -> Standing: latch robot pose at arrival and refresh head_ref."""
        self.standing_latch_xy_yaw = capture_standing_latch(robot)
        self.head_ref_xy_yaw = capture_head_ref_from_pose(
            filtered_pose_7d, offset_rad=offset_rad
        )
        self.mode = "standing"
        self.frozen_snapshot = None


def print_debug_block_v2(
    step: int,
    *,
    active: bool,
    robot_pos: tuple[float, float, float],
    robot_yaw: float,
    raw_head_pose: np.ndarray | None,
    filtered_head_pose: np.ndarray | None,
    head_ref_xy_yaw: tuple[float, float, float] | None,
    standing_latch_xy_yaw: tuple[float, float, float] | None,
    loco_mode: LocoMode | None,
    err_x: float,
    err_y: float,
    err_yaw: float,
    dist_xy: float,
    head_yaw: float | None,
    ref_dist_xy: float | None,
    ref_yaw_err: float | None,
    vx: float,
    vy: float,
    wz: float,
    upper_body_mode: UpperBodyMode | None,
) -> None:
    lines = [
        f"[head_follow_v2] step={step} active={active}",
        f"  robot   pos=({robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}) "
        f"yaw={robot_yaw:.3f}",
    ]
    if raw_head_pose is not None:
        raw_yaw = head_yaw_from_pose(raw_head_pose)
        lines.append(
            f"  head_raw pos=({float(raw_head_pose[0]):.3f}, {float(raw_head_pose[1]):.3f}, "
            f"{float(raw_head_pose[2]):.3f}) yaw={raw_yaw:.3f}"
        )
    else:
        lines.append("  head_raw unavailable")
    if filtered_head_pose is not None:
        filt_yaw = head_yaw if head_yaw is not None else head_yaw_from_pose(filtered_head_pose)
        lines.append(
            f"  head_filt pos=({float(filtered_head_pose[0]):.3f}, "
            f"{float(filtered_head_pose[1]):.3f}, {float(filtered_head_pose[2]):.3f}) "
            f"yaw={filt_yaw:.3f}"
        )
    else:
        lines.append("  head_filt unavailable")
    if head_ref_xy_yaw is not None:
        lines.append(
            f"  head_ref xy=({head_ref_xy_yaw[0]:.3f}, {head_ref_xy_yaw[1]:.3f}) "
            f"yaw={head_ref_xy_yaw[2]:.3f}"
        )
    if standing_latch_xy_yaw is not None:
        lines.append(
            f"  stand_latch xy=({standing_latch_xy_yaw[0]:.3f}, "
            f"{standing_latch_xy_yaw[1]:.3f}) yaw={standing_latch_xy_yaw[2]:.3f}"
        )
    mode_str = loco_mode if loco_mode is not None else "n/a"
    upper_str = upper_body_mode if upper_body_mode is not None else "n/a"
    ref_dist_str = f"{ref_dist_xy:.3f}" if ref_dist_xy is not None else "n/a"
    ref_yaw_str = f"{ref_yaw_err:.3f}" if ref_yaw_err is not None else "n/a"
    lines.extend([
        f"  loco    mode={mode_str} upper_body={upper_str}",
        f"  vs_ref  dxy={ref_dist_str} dyaw={ref_yaw_str}",
        f"  cmd_err dxy=({err_x:.3f}, {err_y:.3f}) dist={dist_xy:.3f} dyaw={err_yaw:.3f}",
        f"  cmd     vx={vx:.3f} vy={vy:.3f} wz={wz:.3f}",
    ])
    print("\n".join(lines), flush=True)


def validate_v2_params(params: HeadFollowV2Params) -> None:
    """Raise ValueError if hysteresis thresholds are inconsistent."""
    if params.pursue_pos_off >= params.pursue_pos_on:
        raise ValueError(
            f"--pursue_pos_off ({params.pursue_pos_off}) must be < "
            f"--pursue_pos_on ({params.pursue_pos_on})"
        )
    if params.pursue_yaw_off >= params.pursue_yaw_on:
        raise ValueError(
            f"--pursue_yaw_off ({params.pursue_yaw_off}) must be < "
            f"--pursue_yaw_on ({params.pursue_yaw_on})"
        )


def standing_leg_command(
    robot,
    standing_latch_xy_yaw: tuple[float, float, float],
    params: HeadFollowV2Params,
) -> tuple[float, float, float]:
    """Station-keep legs toward latched Standing pose."""
    return station_keep_command(
        robot,
        standing_latch_xy_yaw,
        lin_gain=params.follow_lin_gain,
        ang_gain=params.follow_ang_gain,
        max_lin=params.follow_max_lin,
        max_ang=params.follow_max_ang,
        pos_deadzone=params.follow_pos_deadzone,
        yaw_deadzone=params.follow_yaw_deadzone,
    )
