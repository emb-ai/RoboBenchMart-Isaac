"""VRD-style head-follow locomotion for Sberchel XR teleop (v3).

Outer loop ports Vision Robot Demo locomotion (breadcrumbs, headset-velocity
feed-forward, cruise/precision/settled, deadband-aware command mapping,
turn-then-walk). No root kinematic teleport / yaw-assist.

Stages gate both legs and upper body (1B):
  - cruise / precision: ZEST chase; upper body frozen (last teleop fingers)
  - settled (2C): zero linear command; yaw-only soft-keep if needed; live XR
    upper-body teleop. Leave settled on XY translation only (not head yaw).
    XY soft-keep is disabled — it floored to 0.2 m/s and walked past reentry.
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch

from dsynth_isaac.teleop.sberchel_head_follow import (
    FrozenUpperSnapshot,
    UpperBodyMode,
    build_upper_from_frozen_snapshot,
    capture_frozen_upper,
    clip,
    head_yaw_from_pose,
    station_keep_command,
    sync_retargeter_idle_from_robot,
    wrap_angle,
    yaw_from_quat_wxyz,
)
from dsynth_isaac.teleop.sberchel_head_follow_v2 import HeadsetPoseFilter

LocoStage = Literal["cruise", "precision", "settled"]


@dataclass(frozen=True)
class HeadFollowV3Params:
    """Runtime v3 head-follow tuning from CLI (VRD-like defaults)."""

    # Headset filter (reuse v2 filter)
    head_filter_pos_alpha: float = 0.05
    head_filter_quat_alpha: float = 0.05
    head_filter_max_pos_step: float = 0.04
    head_filter_max_yaw_step: float = 0.08
    follow_head_yaw_offset_deg: float = 0.0

    # Breadcrumbs / pure-pursuit
    breadcrumb_min_spacing: float = 0.22
    breadcrumb_min_speed: float = 0.15
    breadcrumb_min_interval: float = 0.05
    waypoint_capture_radius: float = 0.18
    waypoint_shortcut_radius: float = 0.35
    waypoint_cruise_speed: float = 0.95
    pursuit_lookahead: float = 0.45
    final_goal_yaw_blend: float = 1.0
    waypoint_travel_heading_min: float = 0.35
    breadcrumb_capacity: int = 300

    # Stage distances
    precision_entry_distance: float = 0.50
    precision_position_tol: float = 0.10
    precision_yaw_tol: float = 0.07
    precision_gain: float = 0.50
    precision_heading_align: float = 0.60
    # Leave settled on XY only — head look/reach yaw must not freeze teleop.
    settled_reentry_position: float = 0.22
    settled_reentry_yaw: float = 0.10  # unused for leave; kept for CLI compat
    settled_reentry_dwell: float = 0.45
    # Soft station-keep engages when residual is outside this band but still
    # inside reentry (2C). Defaults equal reentry so soft-keep covers that gap.
    soft_keep_position: float = 0.16
    soft_keep_yaw: float = 0.07

    # Cruise feedback / feed-forward
    follow_gain: float = 0.42
    headset_velocity_ff: float = 0.85
    yaw_follow_gain: float = 0.85
    headset_yaw_ff: float = 0.10

    # ZEST deadband / tracking-curve inverses (from VRD characterization)
    policy_forward_slope: float = 1.26
    policy_forward_intercept: float = 0.09
    policy_forward_min_command: float = 0.20
    policy_backward_ratio: float = 0.55
    policy_backward_min_command: float = 0.30
    policy_yaw_slope: float = 0.60
    policy_yaw_intercept: float = 0.15
    forward_engage_speed: float = 0.10
    yaw_engage_rate: float = 0.04
    max_forward_speed: float = 1.0
    max_backward_speed: float = 0.40
    max_yaw_rate: float = 1.5
    max_linear_command_rate: float = 1.2
    max_yaw_command_rate: float = 2.4

    # Soft station-keep in settled (2C)
    soft_lin_gain: float = 0.8
    soft_ang_gain: float = 1.2
    soft_max_lin: float = 0.25
    soft_max_ang: float = 0.40
    soft_pos_deadzone: float = 0.03
    soft_yaw_deadzone: float = 0.03

    print_every_n: int = 30

    @property
    def head_yaw_offset_rad(self) -> float:
        return math.radians(self.follow_head_yaw_offset_deg)

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> HeadFollowV3Params:
        return cls(
            head_filter_pos_alpha=args.head_filter_pos_alpha,
            head_filter_quat_alpha=args.head_filter_quat_alpha,
            head_filter_max_pos_step=args.head_filter_max_pos_step,
            head_filter_max_yaw_step=args.head_filter_max_yaw_step,
            follow_head_yaw_offset_deg=args.follow_head_yaw_offset_deg,
            breadcrumb_min_spacing=args.breadcrumb_min_spacing,
            breadcrumb_min_speed=args.breadcrumb_min_speed,
            breadcrumb_min_interval=args.breadcrumb_min_interval,
            waypoint_capture_radius=args.waypoint_capture_radius,
            waypoint_shortcut_radius=args.waypoint_shortcut_radius,
            waypoint_cruise_speed=args.waypoint_cruise_speed,
            pursuit_lookahead=args.pursuit_lookahead,
            final_goal_yaw_blend=args.final_goal_yaw_blend,
            precision_entry_distance=args.precision_entry_distance,
            precision_position_tol=args.precision_position_tol,
            precision_yaw_tol=args.precision_yaw_tol,
            settled_reentry_position=args.settled_reentry_position,
            settled_reentry_yaw=args.settled_reentry_yaw,
            settled_reentry_dwell=args.settled_reentry_dwell,
            soft_keep_position=args.soft_keep_position,
            soft_keep_yaw=args.soft_keep_yaw,
            follow_gain=args.follow_gain,
            headset_velocity_ff=args.headset_velocity_ff,
            yaw_follow_gain=args.yaw_follow_gain,
            max_forward_speed=args.max_forward_speed,
            max_backward_speed=args.max_backward_speed,
            max_yaw_rate=args.max_yaw_rate,
            soft_lin_gain=args.soft_lin_gain,
            soft_ang_gain=args.soft_ang_gain,
            soft_max_lin=args.soft_max_lin,
            soft_max_ang=args.soft_max_ang,
            print_every_n=args.print_every_n,
        )


def add_head_follow_v3_cli_args(parser: argparse.ArgumentParser) -> None:
    """CLI for VRD-style v3 locomotion."""
    parser.add_argument("--head_filter_pos_alpha", type=float, default=0.05)
    parser.add_argument("--head_filter_quat_alpha", type=float, default=0.05)
    parser.add_argument("--head_filter_max_pos_step", type=float, default=0.04)
    parser.add_argument("--head_filter_max_yaw_step", type=float, default=0.08)
    parser.add_argument("--follow_head_yaw_offset_deg", type=float, default=0.0)

    parser.add_argument("--breadcrumb_min_spacing", type=float, default=0.22)
    parser.add_argument("--breadcrumb_min_speed", type=float, default=0.15)
    parser.add_argument("--breadcrumb_min_interval", type=float, default=0.05)
    parser.add_argument("--waypoint_capture_radius", type=float, default=0.18)
    parser.add_argument("--waypoint_shortcut_radius", type=float, default=0.35)
    parser.add_argument("--waypoint_cruise_speed", type=float, default=0.95)
    parser.add_argument("--pursuit_lookahead", type=float, default=0.45)
    parser.add_argument("--final_goal_yaw_blend", type=float, default=1.0)

    parser.add_argument("--precision_entry_distance", type=float, default=0.50)
    parser.add_argument("--precision_position_tol", type=float, default=0.10)
    parser.add_argument("--precision_yaw_tol", type=float, default=0.07)
    parser.add_argument("--settled_reentry_position", type=float, default=0.22)
    parser.add_argument("--settled_reentry_yaw", type=float, default=0.10)
    parser.add_argument("--settled_reentry_dwell", type=float, default=0.45)
    parser.add_argument(
        "--soft_keep_position",
        type=float,
        default=0.16,
        help="Settled soft station-keep engages above this XY residual (m).",
    )
    parser.add_argument(
        "--soft_keep_yaw",
        type=float,
        default=0.07,
        help="Settled soft station-keep engages above this |yaw| residual (rad).",
    )

    parser.add_argument("--follow_gain", type=float, default=0.42)
    parser.add_argument("--headset_velocity_ff", type=float, default=0.85)
    parser.add_argument("--yaw_follow_gain", type=float, default=0.85)
    parser.add_argument("--max_forward_speed", type=float, default=1.0)
    parser.add_argument("--max_backward_speed", type=float, default=0.40)
    parser.add_argument("--max_yaw_rate", type=float, default=1.5)

    parser.add_argument("--soft_lin_gain", type=float, default=0.8)
    parser.add_argument("--soft_ang_gain", type=float, default=1.2)
    parser.add_argument("--soft_max_lin", type=float, default=0.25)
    parser.add_argument("--soft_max_ang", type=float, default=0.40)

    parser.add_argument(
        "--print_every_n",
        type=int,
        default=30,
        help="Print debug block every N simulation steps (0 to disable).",
    )


def validate_v3_params(params: HeadFollowV3Params) -> None:
    if params.precision_position_tol <= 0:
        raise ValueError("precision_position_tol must be > 0")
    if params.settled_reentry_position < params.precision_position_tol:
        raise ValueError(
            "settled_reentry_position must be >= precision_position_tol"
        )
    if params.settled_reentry_yaw < params.precision_yaw_tol:
        raise ValueError("settled_reentry_yaw must be >= precision_yaw_tol")
    if params.soft_keep_position > params.settled_reentry_position:
        raise ValueError(
            "soft_keep_position must be <= settled_reentry_position"
        )
    if params.waypoint_capture_radius >= params.breadcrumb_min_spacing:
        raise ValueError(
            "waypoint_capture_radius must be < breadcrumb_min_spacing"
        )


@dataclass
class BreadcrumbWaypoint:
    x: float
    y: float
    yaw: float
    timestamp: float


@dataclass
class LocoDebug:
    """Per-step locomotion debug fields for printing / HUD."""

    stage: LocoStage = "settled"
    breadcrumb_count: int = 0
    err_x: float = 0.0
    err_y: float = 0.0
    err_yaw: float = 0.0
    dist_xy: float = 0.0
    head_yaw: float = 0.0
    headset_vx: float = 0.0
    headset_vy: float = 0.0
    headset_wz: float = 0.0
    desired_forward: float = 0.0
    desired_yaw_rate: float = 0.0
    cmd_vx: float = 0.0
    cmd_vy: float = 0.0
    cmd_wz: float = 0.0
    soft_station_keep: bool = False


@dataclass
class HeadFollowV3Runtime:
    """Mutable runtime state for v3 cruise/precision/settled locomotion."""

    filter: HeadsetPoseFilter
    stage: LocoStage = "settled"
    breadcrumbs: deque[BreadcrumbWaypoint] = field(default_factory=deque)
    frozen_snapshot: FrozenUpperSnapshot | None = None
    last_teleop_upper: torch.Tensor | None = None
    last_command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    last_command_time: float | None = None
    settled_breach_started_at: float | None = None
    _head_hist: list[tuple[float, float, float, float]] = field(
        default_factory=list
    )  # (t, x, y, yaw)
    _session_time: float = 0.0

    @classmethod
    def from_params(cls, params: HeadFollowV3Params) -> HeadFollowV3Runtime:
        return cls(
            filter=HeadsetPoseFilter(
                params.head_filter_pos_alpha,
                params.head_filter_quat_alpha,
                params.head_filter_max_pos_step,
                params.head_filter_max_yaw_step,
            ),
        )

    def clear_session(self) -> None:
        self.stage = "settled"
        self.breadcrumbs.clear()
        self.frozen_snapshot = None
        self.last_teleop_upper = None
        self.last_command = (0.0, 0.0, 0.0)
        self.last_command_time = None
        self.settled_breach_started_at = None
        self._head_hist.clear()
        self._session_time = 0.0
        self.filter._pose = None

    def begin_session(self, filtered_pose_7d: np.ndarray) -> None:
        """First frame after START: settle at current pose, clear path memory."""
        self.clear_session()
        self.filter.reset(filtered_pose_7d)
        self.stage = "settled"


def _blend_angles(from_yaw: float, to_yaw: float, blend: float) -> float:
    blend = max(0.0, min(1.0, blend))
    return wrap_angle(from_yaw + blend * wrap_angle(to_yaw - from_yaw))


def _body_axes(yaw: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Forward and lateral unit axes in world XY for robot yaw (Isaac: +X forward)."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    forward = (c, s)
    lateral = (-s, c)
    return forward, lateral


def _dot2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def forward_command_units(speed: float, params: HeadFollowV3Params) -> float:
    """Map desired forward speed (m/s) to Network lin_vel_x (m/s).

    Isaac ``Network`` expects physical m/s. Sub-engage speeds become exact 0.
    Engaged speeds are floored to ``policy_forward_min_command`` so soft-keep /
    precision creep still produces a gait step.
    """
    if speed >= params.forward_engage_speed:
        capped = min(speed, params.max_forward_speed)
        return max(capped, params.policy_forward_min_command)
    if speed <= -params.forward_engage_speed:
        capped = max(speed, -params.max_backward_speed)
        return min(capped, -params.policy_backward_min_command)
    return 0.0


def yaw_command_units(rate: float, params: HeadFollowV3Params) -> float:
    """Map desired yaw rate (rad/s) to Network ang_vel_z (rad/s).

    Do **not** floor small yaw rates: a constant yaw min-floor can trap the
    robot in precision near the goal (limit-cycle, no settle).
    """
    if abs(rate) < params.yaw_engage_rate:
        return 0.0
    return clip(rate, params.max_yaw_rate)


def policy_command(
    forward: float,
    yaw_rate: float,
    params: HeadFollowV3Params,
    *,
    lateral: float = 0.0,
) -> tuple[float, float, float]:
    """Physical desired velocities -> Network command units."""
    # Lateral is weak on this ZEST checkpoint but still helps when the user
    # steps sideways; v3 previously forced 0 and then only yawed in place.
    max_y = 0.4
    return (
        forward_command_units(forward, params),
        clip(lateral, max_y),
        yaw_command_units(yaw_rate, params),
    )


def _slew_command(
    runtime: HeadFollowV3Runtime,
    target: tuple[float, float, float],
    params: HeadFollowV3Params,
    timestamp: float,
) -> tuple[float, float, float]:
    if runtime.last_command_time is None:
        runtime.last_command = target
        runtime.last_command_time = timestamp
        return target
    dt = max(0.0, min(timestamp - runtime.last_command_time, 0.08))
    runtime.last_command_time = timestamp
    max_lin = params.max_linear_command_rate * dt
    max_yaw = params.max_yaw_command_rate * dt
    cx, cy, cz = runtime.last_command
    tx, ty, tz = target
    out = (
        cx + clip(tx - cx, max_lin),
        cy + clip(ty - cy, max_lin),
        cz + clip(tz - cz, max_yaw),
    )
    runtime.last_command = out
    return out


def _update_headset_motion(
    runtime: HeadFollowV3Runtime,
    x: float,
    y: float,
    yaw: float,
    timestamp: float,
) -> tuple[float, float, float]:
    """Return (vx, vy, yaw_rate) from short history; store sample."""
    hist = runtime._head_hist
    hist.append((timestamp, x, y, yaw))
    # Keep ~0.35 s of samples
    while len(hist) > 1 and timestamp - hist[0][0] > 0.35:
        hist.pop(0)
    if len(hist) < 2:
        return 0.0, 0.0, 0.0
    t0, x0, y0, yaw0 = hist[0]
    t1, x1, y1, yaw1 = hist[-1]
    dt = t1 - t0
    if dt < 1e-3:
        return 0.0, 0.0, 0.0
    return (x1 - x0) / dt, (y1 - y0) / dt, wrap_angle(yaw1 - yaw0) / dt


def _append_breadcrumb(
    runtime: HeadFollowV3Runtime,
    params: HeadFollowV3Params,
    x: float,
    y: float,
    yaw: float,
    speed: float,
    timestamp: float,
) -> None:
    # Do not drop crumbs while settled — residual reach/look motion must not
    # force cruise and freeze upper-body teleop.
    if runtime.stage == "settled":
        return
    if speed <= params.breadcrumb_min_speed:
        return
    if runtime.breadcrumbs:
        last = runtime.breadcrumbs[-1]
        if timestamp - last.timestamp < params.breadcrumb_min_interval:
            return
        if math.hypot(x - last.x, y - last.y) <= params.breadcrumb_min_spacing:
            return
    runtime.breadcrumbs.append(
        BreadcrumbWaypoint(x=x, y=y, yaw=yaw, timestamp=timestamp)
    )
    while len(runtime.breadcrumbs) > params.breadcrumb_capacity:
        runtime.breadcrumbs.popleft()


def _pop_satisfied_waypoints(
    runtime: HeadFollowV3Runtime,
    params: HeadFollowV3Params,
    robot_x: float,
    robot_y: float,
) -> None:
    while runtime.breadcrumbs:
        front = runtime.breadcrumbs[0]
        if math.hypot(front.x - robot_x, front.y - robot_y) > params.waypoint_capture_radius:
            break
        runtime.breadcrumbs.popleft()


def _prune_queue_for_shortcut(
    runtime: HeadFollowV3Runtime,
    params: HeadFollowV3Params,
    robot_x: float,
    robot_y: float,
) -> None:
    newest_match = None
    for i in range(len(runtime.breadcrumbs) - 1, -1, -1):
        wp = runtime.breadcrumbs[i]
        if math.hypot(wp.x - robot_x, wp.y - robot_y) <= params.waypoint_shortcut_radius:
            newest_match = i
            break
    if newest_match is None:
        return
    for _ in range(newest_match + 1):
        runtime.breadcrumbs.popleft()


def _pursuit_target_point(
    runtime: HeadFollowV3Runtime,
    params: HeadFollowV3Params,
    robot_x: float,
    robot_y: float,
) -> tuple[float, float] | None:
    if not runtime.breadcrumbs:
        return None
    previous = (runtime.breadcrumbs[0].x, runtime.breadcrumbs[0].y)
    remaining = params.pursuit_lookahead - math.hypot(
        previous[0] - robot_x, previous[1] - robot_y
    )
    if remaining <= 0.0:
        return previous
    for wp in list(runtime.breadcrumbs)[1:]:
        seg = (wp.x - previous[0], wp.y - previous[1])
        seg_len = math.hypot(seg[0], seg[1])
        if seg_len > 1e-4 and remaining < seg_len:
            return (
                previous[0] + seg[0] * (remaining / seg_len),
                previous[1] + seg[1] * (remaining / seg_len),
            )
        remaining -= seg_len
        previous = (wp.x, wp.y)
    return previous


def _waypoint_pursuit_command(
    runtime: HeadFollowV3Runtime,
    params: HeadFollowV3Params,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    headset_yaw: float,
    dist_to_headset: float,
) -> tuple[float, float, float, float, float]:
    """Return (cmd_vx, cmd_vy, cmd_wz, desired_forward, desired_yaw_rate)."""
    first = runtime.breadcrumbs[0]
    pursuit = _pursuit_target_point(runtime, params, robot_x, robot_y)
    if pursuit is None:
        pursuit = (first.x, first.y)
    to_p = (pursuit[0] - robot_x, pursuit[1] - robot_y)
    pursuit_dist = math.hypot(to_p[0], to_p[1])
    if pursuit_dist > 1e-3:
        direction = (to_p[0] / pursuit_dist, to_p[1] / pursuit_dist)
        desired_world = (
            direction[0] * params.waypoint_cruise_speed,
            direction[1] * params.waypoint_cruise_speed,
        )
    else:
        desired_world = (0.0, 0.0)

    forward, lateral_axis = _body_axes(robot_yaw)
    desired_forward = _dot2(desired_world, forward)
    desired_lateral = _dot2(desired_world, lateral_axis)

    if pursuit_dist > 1e-3:
        travel_heading = math.atan2(to_p[1], to_p[0])
    else:
        travel_heading = robot_yaw

    if dist_to_headset < params.final_goal_yaw_blend:
        blend = 1.0 - dist_to_headset / params.final_goal_yaw_blend
        target_yaw = _blend_angles(travel_heading, headset_yaw, blend)
    elif pursuit_dist > params.waypoint_travel_heading_min:
        target_yaw = travel_heading
    else:
        target_yaw = first.yaw

    yaw_err = wrap_angle(target_yaw - robot_yaw)
    desired_yaw_rate = yaw_err * params.yaw_follow_gain
    align = abs(yaw_err)
    if align > params.precision_heading_align:
        scale = max(0.35, math.cos(min(align, math.pi * 0.5)))
        desired_forward *= scale
        desired_yaw_rate = clip(desired_yaw_rate * 1.25, params.max_yaw_rate)
    cmd = policy_command(
        desired_forward, desired_yaw_rate, params, lateral=desired_lateral
    )
    return (*cmd, desired_forward, desired_yaw_rate)


def _precision_approach_command(
    params: HeadFollowV3Params,
    err_x: float,
    err_y: float,
    yaw_err: float,
    robot_yaw: float,
    distance: float,
) -> tuple[float, float, float, float, float]:
    desired_forward = 0.0
    desired_lateral = 0.0
    desired_yaw_rate = 0.0

    if distance > params.precision_position_tol:
        error_heading = math.atan2(err_y, err_x)
        forward_heading_err = wrap_angle(error_heading - robot_yaw)
        moving_forward = abs(forward_heading_err) <= math.pi * 0.5
        aim_err = (
            forward_heading_err
            if moving_forward
            else wrap_angle(forward_heading_err - math.pi)
        )
        desired_yaw_rate = aim_err * params.yaw_follow_gain
        creep = max(
            distance * params.precision_gain,
            params.forward_engage_speed + 0.02,
        )
        # Always allow some creep (scaled); hard-zero while aligning left the
        # robot stuck near the goal with only the yaw floor (see logs).
        scale = max(0.35, math.cos(min(abs(aim_err), math.pi * 0.5)))
        step = creep * scale
        desired_forward = step if moving_forward else -step
        forward, lateral_axis = _body_axes(robot_yaw)
        world = (err_x / distance * creep, err_y / distance * creep)
        desired_lateral = _dot2(world, lateral_axis) * scale
    elif abs(yaw_err) > params.precision_yaw_tol:
        desired_yaw_rate = yaw_err * params.yaw_follow_gain

    cmd = policy_command(
        desired_forward, desired_yaw_rate, params, lateral=desired_lateral
    )
    return (*cmd, desired_forward, desired_yaw_rate)


def _cruise_live_command(
    params: HeadFollowV3Params,
    err_x: float,
    err_y: float,
    distance: float,
    robot_yaw: float,
    yaw_err: float,
    headset_vx: float,
    headset_vy: float,
    headset_wz: float,
) -> tuple[float, float, float, float, float]:
    desired_world = (
        err_x * params.follow_gain + headset_vx * params.headset_velocity_ff,
        err_y * params.follow_gain + headset_vy * params.headset_velocity_ff,
    )
    forward, lateral_axis = _body_axes(robot_yaw)
    desired_forward = _dot2(desired_world, forward)
    desired_lateral = _dot2(desired_world, lateral_axis)

    error_heading = (
        math.atan2(err_y, err_x) if distance > 1e-3 else robot_yaw
    )
    user_facing = wrap_angle(robot_yaw + yaw_err)
    if distance < params.final_goal_yaw_blend:
        blend = 1.0 - distance / params.final_goal_yaw_blend
        target_yaw = _blend_angles(error_heading, user_facing, blend)
    else:
        target_yaw = error_heading
    yaw_err_to_target = wrap_angle(target_yaw - robot_yaw)
    desired_yaw_rate = (
        yaw_err_to_target * params.yaw_follow_gain
        + headset_wz * params.headset_yaw_ff
    )
    align = abs(yaw_err_to_target)
    if align > params.precision_heading_align:
        scale = max(0.35, math.cos(min(align, math.pi * 0.5)))
        desired_forward *= scale
        desired_yaw_rate = clip(desired_yaw_rate * 1.25, params.max_yaw_rate)
    cmd = policy_command(
        desired_forward, desired_yaw_rate, params, lateral=desired_lateral
    )
    return (*cmd, desired_forward, desired_yaw_rate)


def _soft_station_keep_mapped(
    robot,
    hold_xy_yaw: tuple[float, float, float],
    params: HeadFollowV3Params,
    *,
    dist_xy: float,
) -> tuple[float, float, float, float, float]:
    """Yaw-only soft keep while settled (no XY locomotion commands)."""
    _ = dist_xy
    _vx_phys, _vy_phys, wz_phys = station_keep_command(
        robot,
        hold_xy_yaw,
        lin_gain=params.soft_lin_gain,
        ang_gain=params.soft_ang_gain,
        max_lin=params.soft_max_lin,
        max_ang=params.soft_max_ang,
        pos_deadzone=params.soft_pos_deadzone,
        yaw_deadzone=params.soft_yaw_deadzone,
    )
    vx_phys = 0.0
    cmd = policy_command(vx_phys, wz_phys, params, lateral=0.0)
    return (*cmd, vx_phys, wz_phys)


def advance_locomotion(
    runtime: HeadFollowV3Runtime,
    robot,
    filtered_pose_7d: np.ndarray,
    params: HeadFollowV3Params,
    dt: float,
) -> tuple[float, float, float, LocoStage, LocoDebug]:
    """Update breadcrumbs/stage and return slew-limited ZEST command units."""
    runtime._session_time += max(dt, 1e-4)
    timestamp = runtime._session_time

    head_x = float(filtered_pose_7d[0])
    head_y = float(filtered_pose_7d[1])
    head_yaw = head_yaw_from_pose(
        filtered_pose_7d, offset_rad=params.head_yaw_offset_rad
    )

    robot_xy = robot.data.root_pos_w[0, :2].detach().cpu().tolist()
    robot_x, robot_y = float(robot_xy[0]), float(robot_xy[1])
    robot_yaw = yaw_from_quat_wxyz(robot.data.root_quat_w[0])

    err_x = head_x - robot_x
    err_y = head_y - robot_y
    dist_xy = math.hypot(err_x, err_y)
    err_yaw = wrap_angle(head_yaw - robot_yaw)

    headset_vx, headset_vy, headset_wz = _update_headset_motion(
        runtime, head_x, head_y, head_yaw, timestamp
    )
    speed = math.hypot(headset_vx, headset_vy)

    _append_breadcrumb(
        runtime, params, head_x, head_y, head_yaw, speed, timestamp
    )
    _pop_satisfied_waypoints(runtime, params, robot_x, robot_y)
    _prune_queue_for_shortcut(runtime, params, robot_x, robot_y)

    debug = LocoDebug(
        err_x=err_x,
        err_y=err_y,
        err_yaw=err_yaw,
        dist_xy=dist_xy,
        head_yaw=head_yaw,
        headset_vx=headset_vx,
        headset_vy=headset_vy,
        headset_wz=headset_wz,
        breadcrumb_count=len(runtime.breadcrumbs),
    )

    soft_keep = False
    if runtime.breadcrumbs:
        runtime.stage = "cruise"
        runtime.settled_breach_started_at = None
        cmd_vx, cmd_vy, cmd_wz, des_f, des_w = _waypoint_pursuit_command(
            runtime,
            params,
            robot_x,
            robot_y,
            robot_yaw,
            head_yaw,
            dist_xy,
        )
        debug.desired_forward = des_f
        debug.desired_yaw_rate = des_w
    else:
        # Live-goal stage machine
        if runtime.stage == "settled":
            # Leave settled on XY only (not head yaw). walking_away uses the
            # reentry distance so residual reach/look motion does not freeze
            # teleop; soft-keep corrects yaw while staying settled.
            walking_away = (
                speed > params.breadcrumb_min_speed
                and dist_xy > params.settled_reentry_position
            )
            breached = dist_xy > params.settled_reentry_position
            if walking_away:
                runtime.settled_breach_started_at = None
                runtime.stage = (
                    "cruise"
                    if dist_xy > params.precision_entry_distance
                    else "precision"
                )
            elif breached:
                if runtime.settled_breach_started_at is None:
                    runtime.settled_breach_started_at = timestamp
                elif (
                    timestamp - runtime.settled_breach_started_at
                    >= params.settled_reentry_dwell
                ):
                    runtime.settled_breach_started_at = None
                    runtime.stage = (
                        "cruise"
                        if dist_xy > params.precision_entry_distance
                        else "precision"
                    )
            else:
                runtime.settled_breach_started_at = None
        elif runtime.stage == "precision":
            runtime.settled_breach_started_at = None
            if dist_xy > params.precision_entry_distance:
                runtime.stage = "cruise"
        else:  # cruise
            runtime.settled_breach_started_at = None
            if dist_xy <= params.precision_entry_distance:
                runtime.stage = "precision"

        if runtime.stage == "precision":
            if (
                dist_xy <= params.precision_position_tol
                and abs(err_yaw) <= params.precision_yaw_tol
            ):
                runtime.stage = "settled"
                runtime.breadcrumbs.clear()
                runtime.last_command = (0.0, 0.0, 0.0)
                cmd_vx = cmd_vy = cmd_wz = 0.0
                debug.desired_forward = 0.0
                debug.desired_yaw_rate = 0.0
            else:
                cmd_vx, cmd_vy, cmd_wz, des_f, des_w = _precision_approach_command(
                    params, err_x, err_y, err_yaw, robot_yaw, dist_xy
                )
                debug.desired_forward = des_f
                debug.desired_yaw_rate = des_w
        elif runtime.stage == "cruise":
            cmd_vx, cmd_vy, cmd_wz, des_f, des_w = _cruise_live_command(
                params,
                err_x,
                err_y,
                dist_xy,
                robot_yaw,
                err_yaw,
                headset_vx,
                headset_vy,
                headset_wz,
            )
            debug.desired_forward = des_f
            debug.desired_yaw_rate = des_w
        else:
            # settled 2C: zero linear; yaw soft-keep only.
            in_tight = (
                dist_xy <= params.precision_position_tol
                and abs(err_yaw) <= params.precision_yaw_tol
            )
            need_soft = abs(err_yaw) > params.soft_keep_yaw
            if in_tight or not need_soft:
                cmd_vx = cmd_vy = cmd_wz = 0.0
                debug.desired_forward = 0.0
                debug.desired_yaw_rate = 0.0
            else:
                soft_keep = True
                hold = (head_x, head_y, head_yaw)
                cmd_vx, cmd_vy, cmd_wz, des_f, des_w = _soft_station_keep_mapped(
                    robot, hold, params, dist_xy=dist_xy
                )
                debug.desired_forward = des_f
                debug.desired_yaw_rate = des_w

    cmd_vx, cmd_vy, cmd_wz = _slew_command(
        runtime, (cmd_vx, cmd_vy, cmd_wz), params, timestamp
    )
    debug.stage = runtime.stage
    debug.cmd_vx = cmd_vx
    debug.cmd_vy = cmd_vy
    debug.cmd_wz = cmd_wz
    debug.soft_station_keep = soft_keep
    debug.breadcrumb_count = len(runtime.breadcrumbs)
    return cmd_vx, cmd_vy, cmd_wz, runtime.stage, debug


def on_stage_transition(
    runtime: HeadFollowV3Runtime,
    robot,
    teleop_interface: Any,
    prev_stage: LocoStage,
    new_stage: LocoStage,
) -> None:
    """Handle upper-body freeze / unfreeze on stage edges."""
    was_settled = prev_stage == "settled"
    now_settled = new_stage == "settled"
    if was_settled and not now_settled:
        runtime.frozen_snapshot = capture_frozen_upper(
            robot, runtime.last_teleop_upper
        )
    elif not was_settled and now_settled:
        runtime.breadcrumbs.clear()
        runtime.last_command = (0.0, 0.0, 0.0)
        sync_retargeter_idle_from_robot(teleop_interface, robot)
        runtime.frozen_snapshot = None


def upper_body_for_stage(
    runtime: HeadFollowV3Runtime,
    robot,
    stage: LocoStage,
    *,
    raw_xr_data: dict,
    teleop_interface: Any,
    device: str,
    idle_upper: torch.Tensor,
) -> tuple[torch.Tensor, UpperBodyMode]:
    """Live retarget in settled; frozen wrists+last fingers otherwise."""
    if stage == "settled":
        upper_body = torch.cat(
            [r.retarget(raw_xr_data) for r in teleop_interface._retargeters],
            dim=-1,
        )
        runtime.last_teleop_upper = (
            upper_body.squeeze(0).clone()
            if upper_body.ndim > 1
            else upper_body.clone()
        )
        mode: UpperBodyMode = "teleop"
    else:
        if runtime.frozen_snapshot is None:
            runtime.frozen_snapshot = capture_frozen_upper(
                robot, runtime.last_teleop_upper
            )
        upper_body = build_upper_from_frozen_snapshot(
            robot, runtime.frozen_snapshot, device
        )
        mode = "frozen"

    if upper_body.ndim == 1:
        upper_body = upper_body.unsqueeze(0)
    return upper_body.to(device), mode


def print_debug_block_v3(
    step: int,
    *,
    active: bool,
    robot_pos: tuple[float, float, float],
    robot_yaw: float,
    raw_head_pose: np.ndarray | None,
    filtered_head_pose: np.ndarray | None,
    debug: LocoDebug | None,
    upper_body_mode: UpperBodyMode | None,
) -> None:
    """Periodic stdout summary for v3 recording."""
    active_str = "yes" if active else "no"
    upper_str = upper_body_mode if upper_body_mode is not None else "n/a"
    print(f"[step {step}] active={active_str} upper={upper_str}")
    print(
        f"  robot   xy=({robot_pos[0]:.3f}, {robot_pos[1]:.3f}) "
        f"yaw={robot_yaw:.3f}"
    )
    if raw_head_pose is not None:
        print(
            f"  head_raw xy=({float(raw_head_pose[0]):.3f}, "
            f"{float(raw_head_pose[1]):.3f})"
        )
    if filtered_head_pose is not None:
        print(
            f"  head_filt xy=({float(filtered_head_pose[0]):.3f}, "
            f"{float(filtered_head_pose[1]):.3f})"
        )
    if debug is not None:
        print(
            f"  loco    stage={debug.stage} crumbs={debug.breadcrumb_count} "
            f"soft_keep={debug.soft_station_keep}"
        )
        print(
            f"  err     xy=({debug.err_x:.3f}, {debug.err_y:.3f}) "
            f"dist={debug.dist_xy:.3f} yaw={debug.err_yaw:.3f} "
            f"head_yaw={debug.head_yaw:.3f}"
        )
        print(
            f"  hmd_vel xy=({debug.headset_vx:.3f}, {debug.headset_vy:.3f}) "
            f"wz={debug.headset_wz:.3f}"
        )
        print(
            f"  desired fwd={debug.desired_forward:.3f} "
            f"yaw_rate={debug.desired_yaw_rate:.3f}"
        )
        print(
            f"  cmd     vx={debug.cmd_vx:.3f} vy={debug.cmd_vy:.3f} "
            f"wz={debug.cmd_wz:.3f}"
        )
