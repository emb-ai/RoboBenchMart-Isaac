"""Sberchel OpenXR retargeter: wrist frame offsets + VRD-style finger curl."""

from __future__ import annotations

import logging
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
    def _target(limit: tuple[float, float], frac: float) -> float:
        lo, hi = limit
        return float(np.clip(_span_target(lo, hi, _squeeze_fraction(frac, squeeze)), lo, hi))

    yaw_name = f"{side}_thumb_proximal_yaw_joint"
    out[yaw_name] = _target(_THUMB_YAW_LIMIT, _CLOSE_FRACTION["thumb_yaw"] * curls["opposition"])

    pitch_name = f"{side}_thumb_proximal_pitch_joint"
    out[pitch_name] = _target(_THUMB_PITCH_LIMIT, _CLOSE_FRACTION["thumb_pitch"] * curls["thumb"])

    for _, robot_digit in _DIGIT_STEMS:
        joint_name = f"{side}_{robot_digit}_proximal_joint"
        frac = _CLOSE_FRACTION[robot_digit] * curls[robot_digit]
        out[joint_name] = _target(_DIGIT_PROXIMAL_LIMIT, frac)

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

        retargeted_hand_joints = _pack_hand_joint_targets(
            self._hand_joint_names, left_targets, right_targets
        )
        if not hasattr(self, "_finger_log_counter"):
            self._finger_log_counter = 0
        self._finger_log_counter += 1
        if self._finger_log_counter <= 3 or self._finger_log_counter % 120 == 0:
            msg = (
                f"[SberchelRetargeter] curls L={left_curls} R={right_curls} "
                f"signed L={self._last_opposition_signed['left']:.3f} "
                f"R={self._last_opposition_signed['right']:.3f} "
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
