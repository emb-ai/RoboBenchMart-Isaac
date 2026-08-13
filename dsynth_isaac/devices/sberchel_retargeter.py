"""Sberchel OpenXR retargeters.

Two finger mappings share one wrist-tracking path:

* :class:`SberchelRetargeter` -- VRD-style per-digit curl (default).
* :class:`SberchelDexRetargeter` -- DexPilot optimizer, matching the approach
  IsaacLab ships for GR1T2.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1t2_retargeter import (
    GR1T2Retargeter,
    GR1T2RetargeterCfg,
)
from isaaclab.devices.retargeter_base import RetargeterBase

logger = logging.getLogger(__name__)

# OpenXR wrist -> Sberchel hand_base_link (wxyz).
_LEFT_WRIST_FRAME_OFFSET_WXYZ = (0.5, 0.5, 0.5, 0.5)
_RIGHT_WRIST_FRAME_OFFSET_WXYZ = (0.5, 0.5, 0.5, 0.5)

# Proximal joint spans (MJCF range: 0 = open/extended, upper = closed).
_DIGIT_PROXIMAL_LIMIT = (0.0, 1.7)
_THUMB_PITCH_LIMIT = (0.0, 0.5)
_THUMB_YAW_LIMIT = (-0.1, 1.3)

# The thumb pitch drives two mimic-coupled slaves (intermediate = 1.334x,
# range 0..0.8; distal = 0.898x, range 0..0.4). The distal slave hits its
# limit at 0.4/0.898 = 0.445 rad of master travel, so pitch commands beyond
# that are unreachable: they just park the joint against the 8.5 Nm effort
# limit (observed live as a permanent saturated 0.4+ rad error). Cap just
# below the blocking point.
_THUMB_PITCH_MAX_REACHABLE = 0.44

# VisionRobotDemo ARKitTrackingManager.swift
_DIGIT_CURL_NORMALIZER = 2.6
_DIGIT_CURL_DEADZONE = 0.05
_THUMB_CURL_NORMALIZER = 1.8
_THUMB_CURL_DEADZONE = 0.05
_DIGIT_FALLBACK_NORMALIZER = 0.92
_THUMB_FALLBACK_NORMALIZER = 1.04

# VisionRobotDemo MuJoCoBridge.mm closeFractionAtFullCurl.
_CLOSE_FRACTION = {
    "thumb_yaw": 1.0,
    "thumb_pitch": 1.0,
    "index": 1.45 / 1.70,
    "middle": 1.48 / 1.70,
    "ring": 1.43 / 1.70,
    "pinky": 1.36 / 1.70,
}

_DIGIT_STEMS = (
    ("index", "index"),
    ("middle", "middle"),
    ("ring", "ring"),
    ("little", "pinky"),
)

# VRD finger blend (MuJoCoBridge.mm VRDUpdateFingerTargets).
_FINGER_BLEND = 0.28

# Palm-plane thumb angle (deg): rest/abduct often near -35..-45° in OpenXR.
_THUMB_ANGLE_OPEN_DEG = -50.0
_THUMB_ANGLE_FULL_DEG = 12.0


def _pose_matrix_from_7d(pose_7d: np.ndarray) -> torch.Tensor:
    position = torch.tensor(pose_7d[:3], dtype=torch.float32)
    quaternion = torch.tensor(pose_7d[3:], dtype=torch.float32)
    return PoseUtils.make_pose(position, PoseUtils.matrix_from_quat(quaternion))


def _pose_7d_from_matrix(pose_matrix: torch.Tensor) -> np.ndarray:
    position, rotation_matrix = PoseUtils.unmake_pose(pose_matrix)
    quaternion = PoseUtils.quat_from_matrix(rotation_matrix)
    return np.concatenate([position.numpy(), quaternion.numpy()])


def _apply_hand_base_frame_offset(
    wrist_7d: np.ndarray, offset_wxyz: tuple[float, float, float, float]
) -> np.ndarray:
    """Map OpenXR wrist pose in world to Sberchel hand_base_link control frame."""
    openxr_wrist_in_world = _pose_matrix_from_7d(wrist_7d)
    offset_quat = torch.tensor(offset_wxyz, dtype=torch.float32)
    hand_base_in_openxr = PoseUtils.make_pose(
        torch.zeros(3, dtype=torch.float32),
        PoseUtils.matrix_from_quat(offset_quat),
    )
    hand_base_in_world = PoseUtils.pose_in_A_to_pose_in_B(hand_base_in_openxr, openxr_wrist_in_world)
    return _pose_7d_from_matrix(hand_base_in_world)


def _joint_xyz(hand_poses: dict[str, np.ndarray], name: str) -> np.ndarray | None:
    pose = hand_poses.get(name)
    if pose is None or len(pose) < 3:
        return None
    return np.asarray(pose[:3], dtype=np.float64)


def _segment_flexion(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    first = b - a
    second = c - b
    n1 = np.linalg.norm(first)
    n2 = np.linalg.norm(second)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_ang = float(np.dot(first / n1, second / n2))
    return float(np.arccos(np.clip(cos_ang, -1.0, 1.0)))


def _normalized_curl(flexion_sum: float, normalizer: float, deadzone: float) -> float:
    raw = float(np.clip(flexion_sum / normalizer, 0.0, 1.0))
    if raw <= deadzone:
        return 0.0
    return float(np.clip((raw - deadzone) / (1.0 - deadzone), 0.0, 1.0))


@dataclass
class _CurlMeasurement:
    value: float
    measured_segments: int


def _digit_curl(hand_poses: dict[str, np.ndarray], openxr_digit: str) -> _CurlMeasurement:
    """VisionRobotDemo ``digitCurl`` on OpenXR bone names."""
    wrist = _joint_xyz(hand_poses, "wrist")
    metacarpal = _joint_xyz(hand_poses, f"{openxr_digit}_metacarpal")
    knuckle = _joint_xyz(hand_poses, f"{openxr_digit}_proximal")
    intermediate_base = _joint_xyz(hand_poses, f"{openxr_digit}_intermediate")
    intermediate_tip = _joint_xyz(hand_poses, f"{openxr_digit}_distal")
    tip = _joint_xyz(hand_poses, f"{openxr_digit}_tip")

    proximal = metacarpal if metacarpal is not None else wrist
    flexion_sum = 0.0
    measured = 0

    if proximal is not None and knuckle is not None and intermediate_base is not None:
        angle = _segment_flexion(proximal, knuckle, intermediate_base)
        if angle is not None:
            flexion_sum += angle
            measured += 1
    if knuckle is not None and intermediate_base is not None and intermediate_tip is not None:
        angle = _segment_flexion(knuckle, intermediate_base, intermediate_tip)
        if angle is not None:
            flexion_sum += angle
            measured += 1
    if intermediate_base is not None and intermediate_tip is not None and tip is not None:
        angle = _segment_flexion(intermediate_base, intermediate_tip, tip)
        if angle is not None:
            flexion_sum += angle
            measured += 1

    if measured > 0:
        normalizer = _DIGIT_CURL_NORMALIZER * measured / 3.0
        return _CurlMeasurement(
            _normalized_curl(flexion_sum, normalizer, _DIGIT_CURL_DEADZONE), measured
        )

    if knuckle is not None and intermediate_base is not None and tip is not None:
        angle = _segment_flexion(knuckle, intermediate_base, tip)
        if angle is not None:
            return _CurlMeasurement(
                _normalized_curl(angle, _DIGIT_FALLBACK_NORMALIZER, _DIGIT_CURL_DEADZONE), 1
            )

    return _CurlMeasurement(0.0, 0)


def _thumb_flexion_curl(hand_poses: dict[str, np.ndarray]) -> _CurlMeasurement:
    """VisionRobotDemo ``thumbFlexionCurl`` (OpenXR has no thumb intermediate)."""
    knuckle = _joint_xyz(hand_poses, "thumb_metacarpal")
    if knuckle is None:
        knuckle = _joint_xyz(hand_poses, "thumb_proximal")
    intermediate_base = _joint_xyz(hand_poses, "thumb_proximal")
    intermediate_tip = _joint_xyz(hand_poses, "thumb_distal")
    tip = _joint_xyz(hand_poses, "thumb_tip")

    flexion_sum = 0.0
    measured = 0

    if knuckle is not None and intermediate_base is not None and intermediate_tip is not None:
        angle = _segment_flexion(knuckle, intermediate_base, intermediate_tip)
        if angle is not None:
            flexion_sum += angle
            measured += 1
    if intermediate_base is not None and intermediate_tip is not None and tip is not None:
        angle = _segment_flexion(intermediate_base, intermediate_tip, tip)
        if angle is not None:
            flexion_sum += angle
            measured += 1

    if measured > 0:
        normalizer = _THUMB_CURL_NORMALIZER * measured / 2.0
        return _CurlMeasurement(
            _normalized_curl(flexion_sum, normalizer, _THUMB_CURL_DEADZONE), measured
        )

    if knuckle is not None and intermediate_base is not None and tip is not None:
        angle = _segment_flexion(knuckle, intermediate_base, tip)
        if angle is not None:
            return _CurlMeasurement(
                _normalized_curl(angle, _THUMB_FALLBACK_NORMALIZER, _THUMB_CURL_DEADZONE), 1
            )

    return _CurlMeasurement(0.0, 0)


def _thumb_opposition(hand_poses: dict[str, np.ndarray], side: str) -> tuple[float, float] | None:
    """OpenXR thumb opposition in the palm plane → [0, 1] via signed angle calibration."""
    wrist = _joint_xyz(hand_poses, "wrist")
    thumb_knuckle = _joint_xyz(hand_poses, "thumb_metacarpal")
    if thumb_knuckle is None:
        thumb_knuckle = _joint_xyz(hand_poses, "thumb_proximal")
    thumb_tip = _joint_xyz(hand_poses, "thumb_tip")
    index_mc = _joint_xyz(hand_poses, "index_metacarpal")
    little_mc = _joint_xyz(hand_poses, "little_metacarpal")
    if (
        wrist is None
        or thumb_knuckle is None
        or thumb_tip is None
        or index_mc is None
        or little_mc is None
    ):
        return None

    forward = 0.5 * (index_mc + little_mc) - wrist
    lateral = index_mc - little_mc
    if np.linalg.norm(forward) < 1e-6 or np.linalg.norm(lateral) < 1e-6:
        return None
    palm_forward = forward / np.linalg.norm(forward)
    palm_lateral = lateral / np.linalg.norm(lateral)
    palm_normal = np.cross(palm_forward, palm_lateral)
    n = np.linalg.norm(palm_normal)
    if n < 1e-6:
        return None
    palm_normal /= n

    thumb = thumb_tip - thumb_knuckle
    thumb_in_plane = thumb - palm_normal * np.dot(thumb, palm_normal)
    if np.linalg.norm(thumb_in_plane) < 1e-6:
        return None
    thumb_dir = thumb_in_plane / np.linalg.norm(thumb_in_plane)
    lat_comp = float(np.dot(thumb_dir, palm_lateral))
    fwd_comp = float(np.dot(thumb_dir, palm_forward))
    signed = -lat_comp
    angle_deg = float(np.rad2deg(np.arctan2(signed, max(fwd_comp, 1e-6))))
    span = _THUMB_ANGLE_FULL_DEG - _THUMB_ANGLE_OPEN_DEG
    opposition = float(np.clip((angle_deg - _THUMB_ANGLE_OPEN_DEG) / span, 0.0, 1.0))
    return opposition, signed


def _resolve_curl(measurement: _CurlMeasurement, prev: float | None) -> float:
    if measurement.measured_segments > 0:
        return measurement.value
    if prev is not None:
        return prev
    return 0.0


def _span_target(open_value: float, closed_value: float, fraction: float) -> float:
    return open_value + fraction * (closed_value - open_value)


def _squeeze_fraction(base: float, squeeze: float) -> float:
    """Blend a retargeted closure fraction toward fully closed.

    Grip force on a position-controlled hand is ``k_p * (commanded - achieved)``,
    and the object -- not the command -- limits ``achieved``. So grip comes from
    commanding closure *past* the contact point, which is exactly how
    green_challenge is driven (its socket API and keyboard control push finger
    targets to 1.7+ rad). XR retargeting never gets there on its own: a human
    wrapping an object stops curling when their *own* fingers touch, so the
    commanded angle lands right about where the object already blocks the robot
    finger, leaving error -- and therefore grip -- near zero.

    Gated on ``base > 0`` so the curl deadzone still guarantees that a fully open
    hand stays exactly open rather than creeping closed while squeeze is held.
    """
    if base <= 0.0 or squeeze <= 0.0:
        return base
    return base + (1.0 - base) * min(1.0, squeeze)


def _retarget_one_hand_joints(
    hand_poses: dict[str, np.ndarray],
    side: str,
    prev_targets: dict[str, float] | None,
    prev_curls: dict[str, float] | None,
    squeeze: float = 0.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Map OpenXR bones → proximal joint targets (VRD curl → joint span).

    ``squeeze`` (0..1) drives the closure fraction toward 1.0 so a grasp
    generates real force; see :func:`_squeeze_fraction`.
    """
    prev_targets = prev_targets or {}
    prev_curls = prev_curls or {}
    curls: dict[str, float] = {}
    out: dict[str, float] = {}

    thumb_meas = _thumb_flexion_curl(hand_poses)
    curls["thumb"] = _resolve_curl(thumb_meas, prev_curls.get("thumb"))

    for openxr_digit, robot_digit in _DIGIT_STEMS:
        meas = _digit_curl(hand_poses, openxr_digit)
        curls[robot_digit] = _resolve_curl(meas, prev_curls.get(robot_digit))

    opp_meas = _thumb_opposition(hand_poses, side)
    if opp_meas is None:
        opposition = float(prev_curls.get("opposition", 0.0))
        signed = float(prev_curls.get("opposition_signed", 0.0))
    else:
        opposition, signed = opp_meas
    curls["opposition"] = float(opposition)
    curls["opposition_signed"] = float(signed)

    # Nothing downstream clamps finger targets (the Pink action term passes hand
    # joints straight to set_joint_position_target), so clamp here.
    def _target(
        limit: tuple[float, float],
        frac: float,
        squeeze_amount: float,
        cap: float | None = None,
    ) -> float:
        lo, hi = limit
        if cap is not None:
            hi = min(hi, cap)
        return float(np.clip(_span_target(*limit, _squeeze_fraction(frac, squeeze_amount)), lo, hi))

    # Thumb yaw is opposition, not curl: deepening it under squeeze sweeps the
    # thumb across an already-closed fist and jams it against the fingers, so
    # squeeze is NOT applied there. Pitch is capped at the mimic-reachable max.
    yaw_name = f"{side}_thumb_proximal_yaw_joint"
    out[yaw_name] = _target(
        _THUMB_YAW_LIMIT, _CLOSE_FRACTION["thumb_yaw"] * curls["opposition"], 0.0
    )

    pitch_name = f"{side}_thumb_proximal_pitch_joint"
    out[pitch_name] = _target(
        _THUMB_PITCH_LIMIT,
        _CLOSE_FRACTION["thumb_pitch"] * curls["thumb"],
        squeeze,
        cap=_THUMB_PITCH_MAX_REACHABLE,
    )

    for _, robot_digit in _DIGIT_STEMS:
        joint_name = f"{side}_{robot_digit}_proximal_joint"
        frac = _CLOSE_FRACTION[robot_digit] * curls[robot_digit]
        out[joint_name] = _target(_DIGIT_PROXIMAL_LIMIT, frac, squeeze)

    return out, curls


def _pack_hand_joint_targets(
    hand_joint_names: list[str],
    left_targets: dict[str, float],
    right_targets: dict[str, float],
) -> np.ndarray:
    targets = np.zeros(len(hand_joint_names), dtype=np.float32)
    merged = {**left_targets, **right_targets}
    missing = []
    for i, name in enumerate(hand_joint_names):
        if name in merged:
            targets[i] = merged[name]
        else:
            missing.append(name)
    if missing:
        logger.warning("[SberchelRetargeter] missing finger targets for: %s", missing)
    return targets


def _blend_targets(
    prev: dict[str, float] | None, current: dict[str, float], alpha: float
) -> dict[str, float]:
    if not prev:
        return dict(current)
    out = {}
    for name, value in current.items():
        out[name] = (1.0 - alpha) * prev.get(name, value) + alpha * value
    return out


class SberchelRetargeter(GR1T2Retargeter):
    """Absolute OpenXR wrist poses + VRD-style proximal finger curl mapping."""

    def __init__(self, cfg: SberchelRetargeterCfg):
        RetargeterBase.__init__(self, cfg)
        self._hand_joint_names = cfg.hand_joint_names
        self._enable_visualization = cfg.enable_visualization
        self._num_open_xr_hand_joints = cfg.num_open_xr_hand_joints
        self._sim_device = cfg.sim_device
        self._prev_left_fingers: dict[str, float] | None = None
        self._prev_right_fingers: dict[str, float] | None = None
        self._prev_left_curls: dict[str, float] | None = None
        self._prev_right_curls: dict[str, float] | None = None
        self._last_opposition_signed: dict[str, float] = {"left": 0.0, "right": 0.0}
        # Extra closure commanded past the human's curl, driven live by the
        # operator (gamepad L2 / keyboard G). See ``_squeeze_fraction``.
        self._squeeze_scale = float(getattr(cfg, "squeeze_scale", 1.0))
        self._squeeze: float = 0.0

        if self._enable_visualization:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/markers",
                markers={
                    "joint": sim_utils.SphereCfg(
                        radius=0.005,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
            self._markers = VisualizationMarkers(marker_cfg)

        print(
            "[SberchelRetargeter] ready — absolute wrist tracking, "
            f"VRD finger curl ({len(self._hand_joint_names)}): {self._hand_joint_names}",
            flush=True,
        )

    def recalibrate_wrists(self) -> None:
        return

    def set_squeeze(self, value: float) -> None:
        """Set the grasp squeeze command (0 = follow XR hand exactly, 1 = full close)."""
        self._squeeze = float(np.clip(value * self._squeeze_scale, 0.0, 1.0))

    @property
    def squeeze(self) -> float:
        return self._squeeze

    def _compute_finger_targets(
        self, left_hand_poses: dict, right_hand_poses: dict
    ) -> tuple[dict[str, float], dict[str, float]]:
        """VRD curl mapping: OpenXR bone flexion -> proximal joint targets.

        Overridden by :class:`SberchelDexRetargeter` to use the DexPilot
        optimizer instead. Everything else in ``retarget`` -- wrist poses,
        packing order, logging -- is shared.
        """
        left_raw, left_curls = _retarget_one_hand_joints(
            left_hand_poses, "left", self._prev_left_fingers, self._prev_left_curls, self._squeeze
        )
        right_raw, right_curls = _retarget_one_hand_joints(
            right_hand_poses, "right", self._prev_right_fingers, self._prev_right_curls, self._squeeze
        )
        left_targets = _blend_targets(self._prev_left_fingers, left_raw, _FINGER_BLEND)
        right_targets = _blend_targets(self._prev_right_fingers, right_raw, _FINGER_BLEND)
        self._prev_left_fingers = left_targets
        self._prev_right_fingers = right_targets
        self._prev_left_curls = left_curls
        self._prev_right_curls = right_curls
        self._last_opposition_signed = {
            "left": float(left_curls.get("opposition_signed", 0.0)),
            "right": float(right_curls.get("opposition_signed", 0.0)),
        }
        return left_targets, right_targets

    def retarget(self, data: dict) -> torch.Tensor:
        left_hand_poses = data[DeviceBase.TrackingTarget.HAND_LEFT]
        right_hand_poses = data[DeviceBase.TrackingTarget.HAND_RIGHT]

        left_wrist = left_hand_poses.get("wrist")
        right_wrist = right_hand_poses.get("wrist")

        if self._enable_visualization:
            joints_position = np.zeros((self._num_open_xr_hand_joints, 3))
            joints_position[::2] = np.array([pose[:3] for pose in left_hand_poses.values()])
            joints_position[1::2] = np.array([pose[:3] for pose in right_hand_poses.values()])
            self._markers.visualize(translations=torch.tensor(joints_position, device=self._sim_device))

        left_targets, right_targets = self._compute_finger_targets(left_hand_poses, right_hand_poses)

        retargeted_hand_joints = _pack_hand_joint_targets(
            self._hand_joint_names, left_targets, right_targets
        )
        if not hasattr(self, "_finger_log_counter"):
            self._finger_log_counter = 0
        self._finger_log_counter += 1
        if self._finger_log_counter <= 3 or self._finger_log_counter % 600 == 0:
            msg = (
                f"[{type(self).__name__}] squeeze={self._squeeze:.2f} "
                f"packed={np.array2string(retargeted_hand_joints, precision=2)}"
            )
            print(msg, flush=True)
            logger.info(msg)

        left_wrist_pose = _apply_hand_base_frame_offset(left_wrist, _LEFT_WRIST_FRAME_OFFSET_WXYZ)
        right_wrist_pose = _apply_hand_base_frame_offset(right_wrist, _RIGHT_WRIST_FRAME_OFFSET_WXYZ)

        left_wrist_tensor = torch.tensor(left_wrist_pose, dtype=torch.float32, device=self._sim_device)
        right_wrist_tensor = torch.tensor(right_wrist_pose, dtype=torch.float32, device=self._sim_device)
        hand_joints_tensor = torch.tensor(retargeted_hand_joints, dtype=torch.float32, device=self._sim_device)

        return torch.cat([left_wrist_tensor, right_wrist_tensor, hand_joints_tensor])


# --- DexPilot input conversion ----------------------------------------------
# OpenXR reports 26 hand joints; dex-retargeting expects the 21-joint MANO
# layout. These mirror IsaacLab's GR1T2 helper
# (gr1_t2_dex_retargeting_utils.py) so our input matches what the shipped
# configs were tuned against.
_DEX_HAND_JOINTS_INDEX = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]
# Operator hand frame -> robot hand frame. This is ROBOT-SPECIFIC: it must align
# the operator's fingertip directions with the robot's own, so GR1's matrix is
# wrong here (Sberchel's hand_base_link needs a 120 deg offset from the OpenXR
# wrist where GR1's does not).
#
# RIGHT: measured against the *extended* hand model (full finger chains, built by
# scripts/sberchel_build_dex_hand_urdf.py), Kabsch-fitted over 6 open-hand and 3
# fist captures -- averaged within each pose group, then fitted jointly so one
# rotation has to serve both. Residual: 3.1 deg open / 3.8 deg fist.
#
# Two earlier attempts, both worse for instructive reasons:
#   * Open pose only, against the 1-DOF stub: 3.8 deg open but 32.7 deg fist, and
#     the fingers moved OPPOSITE to the operator's. Aligning one pose does not
#     constrain the direction of travel.
#   * One open + one fist capture, extended model: 8.2 / 14.0 deg. A single frame
#     of hand tracking is noisy enough to dominate the fit; averaging several
#     captures per pose cut the error by ~4x. Do not fit from one frame.
#
# LEFT: measured via mirror symmetry rather than direct capture. OpenXR's hands
# are mirror images in the operator frame (x flip -- confirmed by comparing
# left/right captures of similar poses) and the robot's hands are mirrored in the
# MJCF (z flip), so the right-hand captures, x-flipped, were Kabsch-fitted onto
# the LEFT model's open/fist directions: 2.5 deg open / 5.3 deg fist residual,
# and the result agrees with the analytic prediction
# diag(-1,1,1) @ R_right @ diag(1,1,-1) to 3.4 deg -- two independent routes to
# the same rotation. If the left hand still misbehaves, capture left open+fist
# TIPFIT lines and fit it directly with scripts/sberchel_fit_dex_frame.py.
_OPERATOR2MANO = {
    "left": np.array(
        [
            [-0.162853, +0.274354, +0.947739],
            [+0.985250, +0.096384, +0.141397],
            [-0.052554, +0.956786, -0.286004],
        ]
    ),
    "right": np.array(
        [
            [+0.214515, -0.295084, +0.931079],
            [+0.974730, +0.125511, -0.184794],
            [-0.062331, +0.947192, +0.314552],
        ]
    ),
}


def _dex_operator_frame_landmarks(hand_poses: dict[str, np.ndarray]) -> np.ndarray:
    """OpenXR hand poses -> (21, 3) MANO-ordered points in the *operator* frame.

    Wrist-relative and de-rotated by the wrist orientation, but NOT yet mapped
    into the robot's canonical frame. Split out from
    :func:`_dex_convert_hand_joints` so the operator->robot rotation can be
    *measured* rather than assumed -- see ``scripts/sberchel_fit_dex_frame.py``.
    """
    from scipy.spatial.transform import Rotation

    joint_position = np.zeros((21, 3))
    joints = list(hand_poses.values())
    for i, src in enumerate(_DEX_HAND_JOINTS_INDEX):
        joint_position[i] = joints[src][:3]
    joint_position = joint_position - joint_position[0:1, :]
    wxyz = hand_poses["wrist"][3:]
    # OpenXR quaternions are wxyz; scipy wants xyzw.
    wrist_rot = Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
    return joint_position @ wrist_rot


def _dex_convert_hand_joints(hand_poses: dict[str, np.ndarray], operator2mano: np.ndarray) -> np.ndarray:
    """OpenXR hand poses -> (21, 3) MANO joint positions in the canonical frame."""
    return _dex_operator_frame_landmarks(hand_poses) @ operator2mano


def _dex_compute_ref_value(joint_position: np.ndarray, indices: np.ndarray, retargeting_type: str) -> np.ndarray:
    """Reference value for the optimizer: tip positions, or inter-tip vectors."""
    if retargeting_type == "POSITION":
        return joint_position[indices, :]
    return joint_position[indices[1, :], :] - joint_position[indices[0, :], :]


class SberchelDexRetargeter(SberchelRetargeter):
    """Same wrist tracking, but fingers come from the DexPilot optimizer.

    Where the VRD mapping converts each digit's measured flexion into a joint
    angle independently, DexPilot solves for joint angles that reproduce the
    *inter-fingertip vectors* of the operator's hand -- notably thumb-to-finger
    distances, which is what makes a pinch a pinch. That is the argument for it
    on precision grasps.

    STATUS: does not track a real hand yet. Keep using the VRD mapping for data
    collection. Root cause NOT yet established -- below is what has been ruled in
    and out, measured live in XR on 2026-08-13.

    Ruled OUT -- input conversion. The debug dump shows 26 keys in the expected
    OpenXR order (palm, wrist, thumb_metacarpal, ...) and human wrist-to-tip
    distances that are anatomically sane and respond correctly to hand motion:
    an open hand reads thumb 14 / index 18 / middle 18 / ring 17 / pinky 15 cm, a
    closed fist reads index 9.9 / middle 6.8 / ring 6.2. The landmarks reaching
    the optimizer are a real hand, correctly ordered and scaled.

    Ruled OUT -- fingertip workspace. An earlier theory held that the robot's
    4 cm of fingertip travel (18.0 -> 14.1 cm) could not cover a human's ~11 cm,
    so the solver saturated. The data contradicts it: an OPEN hand at 17.7 cm --
    comfortably inside the robot's reachable band -- still returns q = 1.70
    (fully closed) on all four digits. And one frame with near-identical input
    (index 17.1, middle 18.5) returned a correct open pose (middle 0.42, pinky
    0.05, ring 0.35). Reachable targets *can* produce open poses, so the
    workspace ceiling is not the binding constraint.

    PRIME SUSPECT -- the canonical frame transform. ``_OPERATOR2MANO`` was copied
    verbatim from IsaacLab's GR1T2 helper, but that matrix aligns the operator's
    hand to a *specific robot's* hand frame, so it is robot-specific. Sberchel's
    ``hand_base_link`` is plainly not GR1's: our own wrist path applies a 120 deg
    ``(0.5, 0.5, 0.5, 0.5)`` offset to reach it. Consistent with this, the output
    depends on hand *orientation* rather than curl alone and flips between full
    closure and a sensible pose -- the signature of landmarks rotated into a frame
    the solver does not expect.

    To settle it: dump the five fingertip unit vectors for a known open hand and
    Kabsch-fit the rotation onto the robot's own open-hand tip directions. That
    yields the matrix directly instead of guessing.

    Two integration details matter:

    * ``dof_joint_names`` from the optimizer is NOT our ``hand_joint_names``
      order (it comes out as index/middle/pinky/ring/thumb_yaw/thumb_pitch), so
      results are mapped back by name, never by position.
    * The DexPilot config carries its own ``low_pass_alpha``, so this path does
      not additionally apply ``_FINGER_BLEND``. Stacking both filters would
      double the lag for no benefit.

    Validate the hand geometry with ``scripts/sberchel_check_dex_retargeting.py``
    -- a wrong thumb frame silently produces a hand that cannot oppose.
    """

    def __init__(self, cfg: SberchelDexRetargeterCfg):
        super().__init__(cfg)
        import yaml
        from dex_retargeting.retargeting_config import RetargetingConfig

        from dsynth_isaac.devices.sberchel_hand_urdf import (
            prepare_sberchel_dex_retargeting_configs,
        )

        cache = cfg.dex_cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "dsynth_isaac", "sberchel_dex"
        )
        left_cfg, right_cfg = prepare_sberchel_dex_retargeting_configs(cache)

        self._dex = {}
        self._dex_limits: dict[str, tuple[float, float]] = {}
        for side, path in (("left", left_cfg), ("right", right_cfg)):
            # Global scaling is forced to 1.0 (or the explicit override): absolute
            # hand size is handled per digit below, and leaving the YAML's 0.92 on
            # top of that would shrink every target inside the model's reach.
            doc = yaml.safe_load(open(path))
            doc["retargeting"]["scaling_factor"] = float(cfg.dex_scaling_factor or 1.0)
            with open(path, "w") as fh:
                yaml.dump(doc, fh)
            rt = RetargetingConfig.load_from_file(path).build()
            self._dex[side] = rt
            for name, (lo, hi) in zip(rt.optimizer.robot.dof_joint_names, rt.optimizer.robot.joint_limits):
                self._dex_limits[name] = (float(lo), float(hi))

        # Per-digit fingertip scale: model open length / operator open length.
        # A single global factor cannot reconcile unequal human digits with the
        # robot's equal-length ones -- calibrated per digit, an open operator
        # hand lands exactly on the model's open pose. (Reshaping the model
        # instead was tried and failed: the pinky's target length is shorter
        # than its joint chain, which collapses the tip onto the intermediate
        # joint and disconnects that joint from the objective entirely.)
        self._dex_tip_scale: dict[str, np.ndarray] = {}
        human_open = np.asarray(cfg.dex_human_open_cm, dtype=float) / 100.0
        for side, rt in self._dex.items():
            robot = rt.optimizer.robot
            lo_q = np.array([l for l, _ in robot.joint_limits])
            robot.compute_forward_kinematics(lo_q)
            origin = robot.get_link_pose(robot.get_link_index(f"{side}_hand_base_link"))[:3, 3]
            model_open = np.array([
                np.linalg.norm(robot.get_link_pose(
                    robot.get_link_index(f"{side}_{d}_tip"))[:3, 3] - origin)
                for d in ("thumb", "index", "middle", "ring", "pinky")
            ])
            self._dex_tip_scale[side] = model_open / human_open

        missing = [n for n in self._hand_joint_names if n not in self._dex_limits]
        if missing:
            raise RuntimeError(
                f"DexPilot configs do not cover these hand joints: {missing}. "
                "Check target_joint_names in dsynth_isaac/assets/sberchel_xr/*.yml."
            )

        # Commandable joint -> the intermediate joint that is mimic-coupled to it
        # on the real robot. pinocchio ignores <mimic>, so the optimizer treats
        # the pair as two free DOFs about parallel axes and only their SUM is
        # determined -- the split between them is a null direction it may resolve
        # arbitrarily. Reading the proximal alone therefore samples a free
        # variable: in one capture three near-identical hand poses gave proximal
        # 0.49 / 0.48 / 0.15 while the coupled projection gave 0.62 / 0.63 / 0.64.
        # Thumb yaw has no partner -- it abducts rather than curls.
        self._dex_coupled: dict[str, str] = {}
        for side in ("left", "right"):
            for digit in ("index", "middle", "ring", "pinky"):
                self._dex_coupled[f"{side}_{digit}_proximal_joint"] = f"{side}_{digit}_intermediate_joint"
            self._dex_coupled[f"{side}_thumb_proximal_pitch_joint"] = f"{side}_thumb_intermediate_joint"
        self._dex_coupled = {
            k: v for k, v in self._dex_coupled.items()
            if k in self._dex_limits and v in self._dex_limits
        }
        scales = " ".join(
            f"{d}={v:.2f}" for d, v in zip(
                ("thumb", "index", "middle", "ring", "pinky"), self._dex_tip_scale["right"])
        )
        print(
            f"[SberchelDexRetargeter] ready — DexPilot fingers "
            f"(global scaling={cfg.dex_scaling_factor or 1.0}, per-digit tip scale "
            f"{scales}), configs in {cache}",
            flush=True,
        )

    def _debug_dump(self, side: str, hand_poses: dict, joint_pos: np.ndarray, q) -> None:
        """One-shot dump of what the optimizer is actually being asked for.

        Separates the two failure modes that look alike from the outside:

        * **Bad input** -- the converted landmarks do not look like a hand
          (wrong key order, wrong frame). Wrist-to-tip distances will be
          nonsense, or the fingers will not lie in a sensible spread.
        * **Unreachable input** -- the landmarks are a sane human hand but the
          targets sit outside the robot's fingertip workspace, so the solver
          saturates. Distances look human (~18 cm) and the robot pins to limits.
        """
        names = list(hand_poses.keys())
        tips = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
        d = {k: float(np.linalg.norm(joint_pos[i] - joint_pos[0])) * 100 for k, i in tips.items()}

        # Fingertip unit vectors in the OPERATOR frame -- i.e. before the
        # suspect _OPERATOR2MANO rotation. Feeding these to
        # scripts/sberchel_fit_dex_frame.py measures the correct
        # operator->robot rotation instead of guessing at it.
        op = _dex_operator_frame_landmarks(hand_poses)
        units = []
        for i in tips.values():
            v = op[i] - op[0]
            n = float(np.linalg.norm(v))
            units.extend((v / n if n > 1e-9 else v).tolist())

        # Report the COMMANDED angle (after collapsing the mimic-coupled pair),
        # not the raw solver DOFs. The raw proximal is a null direction of the
        # QP and swings wildly on a still hand, which reads as a bug that isn't.
        dof_names = list(self._dex[side].optimizer.robot.dof_joint_names)
        solved = dict(zip(dof_names, (float(v) for v in q)))
        cmd = []
        for n in dof_names:
            if n not in self._hand_joint_names:
                continue
            lo, hi = self._dex_limits[n]
            frac = self._coupled_fraction(n, solved, lo, hi)
            label = n.replace(f"{side}_", "").replace("_joint", "").replace("_proximal", "")
            cmd.append(f"{label[:11]}={lo + frac * (hi - lo):+.2f}")

        print(
            f"[dex-debug {side}] keys={len(names)} first5={names[:5]}\n"
            f"    human wrist->tip cm: " + " ".join(f"{k}={v:.1f}" for k, v in d.items()) + "\n"
            f"    commanded: " + " ".join(cmd) + "\n"
            f"    raw solver q: " + " ".join(f"{n.split('_')[1][:5]}={v:+.2f}"
                                             for n, v in zip(dof_names, q)) + "\n"
            f"    TIPFIT {side} " + " ".join(f"{x:+.5f}" for x in units),
            flush=True,
        )

    def _retarget_side(self, side: str, hand_poses: dict | None) -> dict[str, float]:
        rt = self._dex[side]
        opt = rt.optimizer
        if not hand_poses:
            return {}
        joint_pos = _dex_convert_hand_joints(hand_poses, _OPERATOR2MANO[side])
        # Stretch each fingertip to the model's digit length (see __init__);
        # joint_pos is wrist-relative, so scaling the tip rows is exact.
        scaled = joint_pos.copy()
        for slot, factor in zip((4, 8, 12, 16, 20), self._dex_tip_scale[side]):
            scaled[slot] *= factor
        ref = _dex_compute_ref_value(
            scaled,
            indices=np.asarray(opt.target_link_human_indices),
            retargeting_type=opt.retargeting_type,
        )
        # dex-retargeting uses autograd internally; the teleop loop runs under
        # torch.inference_mode(), which would otherwise make retarget() throw.
        with torch.enable_grad(), torch.inference_mode(False):
            q = rt.retarget(ref)

        self._dex_debug_count = getattr(self, "_dex_debug_count", 0) + 1
        if self._dex_debug_count <= 4 or self._dex_debug_count % 300 == 0:
            self._debug_dump(side, hand_poses, joint_pos, q)

        solved = dict(zip(opt.robot.dof_joint_names, (float(v) for v in q)))

        out: dict[str, float] = {}
        for name in opt.robot.dof_joint_names:
            if name not in self._hand_joint_names:
                continue
            lo, hi = self._dex_limits[name]
            frac = self._coupled_fraction(name, solved, lo, hi)
            # Thumb yaw is opposition, not curl -- squeeze-deepening it sweeps
            # the thumb across an already-closed fist and jams it against the
            # fingers, so squeeze applies to every digit except it.
            if not name.endswith("thumb_proximal_yaw_joint"):
                frac = _squeeze_fraction(frac, self._squeeze)
            target = lo + frac * (hi - lo)
            if name.endswith("thumb_proximal_pitch_joint"):
                target = min(target, _THUMB_PITCH_MAX_REACHABLE)
            out[name] = float(np.clip(target, lo, hi))
        return out

    def _coupled_fraction(
        self, name: str, solved: dict[str, float], lo: float, hi: float
    ) -> float:
        """How far closed this digit is, as a fraction, honouring mimic coupling.

        The real finger has one degree of freedom: proximal and intermediate move
        together, so the physical configurations form a line through joint space
        from fully open (both at their lower limits) to fully closed (both upper).
        The optimizer does not know that and returns a point off the line. Project
        onto it -- a least-squares fit in joint space -- and report where it lands.

        Using the raw proximal instead reads a quantity the solver never pinned
        down, which is what made the fingers jump between 0.0 and 1.3 while the
        operator's hand sat still. See ``_dex_coupled``.
        """
        partner = self._dex_coupled.get(name)
        if partner is None:
            span = hi - lo
            return float(np.clip((solved[name] - lo) / span, 0.0, 1.0)) if span > 0 else 0.0

        p_lo, p_hi = lo, hi
        i_lo, i_hi = self._dex_limits[partner]
        dp, di = p_hi - p_lo, i_hi - i_lo
        denom = dp * dp + di * di
        if denom <= 0.0:
            return 0.0
        t = ((solved[name] - p_lo) * dp + (solved[partner] - i_lo) * di) / denom
        return float(np.clip(t, 0.0, 1.0))

    def _compute_finger_targets(
        self, left_hand_poses: dict, right_hand_poses: dict
    ) -> tuple[dict[str, float], dict[str, float]]:
        left = self._retarget_side("left", left_hand_poses)
        right = self._retarget_side("right", right_hand_poses)
        # Hold the last solution when tracking drops out, rather than snapping open.
        if left:
            self._prev_left_fingers = left
        if right:
            self._prev_right_fingers = right
        return (self._prev_left_fingers or {}), (self._prev_right_fingers or {})


@dataclass
class SberchelRetargeterCfg(GR1T2RetargeterCfg):
    """Sberchel retargeter config."""

    full_robot_urdf_path: str = ""
    dex_cache_dir: str = ""
    left_idle_wrist_pose: tuple[float, ...] = (0.0,) * 7
    right_idle_wrist_pose: tuple[float, ...] = (0.0,) * 7
    # Multiplier on the operator's squeeze input before it is clamped to 0..1.
    # >1 reaches full closure at partial trigger travel.
    squeeze_scale: float = 1.0
    retargeter_type: type = SberchelRetargeter


@dataclass
class SberchelDexRetargeterCfg(SberchelRetargeterCfg):
    """Config for the DexPilot finger mapping."""

    # Where the patched hand URDFs / DexPilot YAMLs are staged.
    dex_cache_dir: str = ""
    # Human-hand-to-robot-hand size calibration. 0 keeps whatever the YAML says
    # (0.92). Lower = the robot curls further for the same human pose; this is
    # per-operator and can only be tuned wearing the headset.
    dex_scaling_factor: float = 0.0
    # The operator's open-hand wrist->fingertip distance per digit, in cm,
    # measured from OpenXR captures. Each fingertip is scaled by
    # (model open length / this) before solving, which corrects the digit
    # PROPORTION mismatch that a single global scaling factor cannot: human
    # digits are unequal (pinky ~18% shorter than middle) while the robot's are
    # equal, so without this an open pinky target lands well inside the model's
    # reach and the solver curls it (~0.8 rad on a flat hand). Re-measure for a
    # different operator by reading "human wrist->tip cm" off the dex-debug log
    # with a flat open hand.
    dex_human_open_cm: tuple[float, float, float, float, float] = (
        15.3, 19.9, 21.0, 19.6, 17.2)  # thumb, index, middle, ring, pinky
    retargeter_type: type = SberchelDexRetargeter
