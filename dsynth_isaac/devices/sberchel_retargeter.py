"""Sberchel OpenXR retargeter: wrist frame offsets + joint-to-joint fingers."""

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
# Derived by scripts/derive_sberchel_openxr_wrist_offset.py (palm-down /
# fingers-forward equate; OpenXR Khronos wrist axes vs open-MJCF hand_base
# distal/palm). Composition: desired = openxr_wrist * C_side (VRD-style).
# Both sides share C because OpenXR and hand_base use the same distal/palm
# axes bilaterally (unlike H12 Inspire RPY offsets).
_LEFT_WRIST_FRAME_OFFSET_WXYZ = (0.5, 0.5, 0.5, 0.5)
_RIGHT_WRIST_FRAME_OFFSET_WXYZ = (0.5, 0.5, 0.5, 0.5)

# Proximal joint limits from Sberchel hand MJCF / dex hand URDFs.
_DIGIT_PROXIMAL_LIMIT = (0.0, 1.7)
_THUMB_PITCH_LIMIT = (0.0, 0.5)
_THUMB_YAW_LIMIT = (-0.1, 1.3)

# OpenXR digit stem -> robot joint stem (little == pinky).
_DIGIT_STEMS = (
    ("index", "index"),
    ("middle", "middle"),
    ("ring", "ring"),
    ("little", "pinky"),
)

# Light EMA so single-frame OpenXR dropouts do not snap fingers (borrowed
# from VRD blend idea; not a curl normalizer).
_FINGER_BLEND = 0.35


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
    """Map OpenXR wrist pose in world to Sberchel hand_base_link control frame.

    Rotation-only right multiply: ``T_hand_base^world = T_openxr^world * C``.
    """
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
    """Angle (rad) between bones a→b and b→c. Straight ≈ 0, flexed > 0."""
    first = b - a
    second = c - b
    n1 = np.linalg.norm(first)
    n2 = np.linalg.norm(second)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_ang = float(np.dot(first / n1, second / n2))
    return float(np.arccos(np.clip(cos_ang, -1.0, 1.0)))


def _clip_joint(value: float, limits: tuple[float, float]) -> float:
    return float(np.clip(value, limits[0], limits[1]))


def _digit_proximal_flexion(hand_poses: dict[str, np.ndarray], openxr_digit: str) -> float | None:
    """OpenXR MCP flexion for one digit → maps 1:1 onto robot proximal joint."""
    metacarpal = _joint_xyz(hand_poses, f"{openxr_digit}_metacarpal")
    proximal = _joint_xyz(hand_poses, f"{openxr_digit}_proximal")
    intermediate = _joint_xyz(hand_poses, f"{openxr_digit}_intermediate")
    distal = _joint_xyz(hand_poses, f"{openxr_digit}_distal")
    tip = _joint_xyz(hand_poses, f"{openxr_digit}_tip")
    wrist = _joint_xyz(hand_poses, "wrist")

    base = metacarpal if metacarpal is not None else wrist
    if base is not None and proximal is not None and intermediate is not None:
        return _segment_flexion(base, proximal, intermediate)
    if proximal is not None and intermediate is not None and distal is not None:
        return _segment_flexion(proximal, intermediate, distal)
    if proximal is not None and intermediate is not None and tip is not None:
        return _segment_flexion(proximal, intermediate, tip)
    return None


def _thumb_proximal_pitch(hand_poses: dict[str, np.ndarray]) -> float | None:
    """Thumb flexion at the proximal joint (metacarpal → proximal → distal)."""
    metacarpal = _joint_xyz(hand_poses, "thumb_metacarpal")
    proximal = _joint_xyz(hand_poses, "thumb_proximal")
    distal = _joint_xyz(hand_poses, "thumb_distal")
    tip = _joint_xyz(hand_poses, "thumb_tip")
    if metacarpal is not None and proximal is not None and distal is not None:
        return _segment_flexion(metacarpal, proximal, distal)
    if proximal is not None and distal is not None and tip is not None:
        return _segment_flexion(proximal, distal, tip)
    return None


def _thumb_opposition(hand_poses: dict[str, np.ndarray]) -> float | None:
    """Thumb yaw cue in [0, 1]: 0 = adducted along palm, 1 = opposed.

    Geometry from VisionRobotDemo ``thumbOpposition`` (not total-curl).
    """
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
    # Negative lateral component → opposition toward the palm center.
    signed = -float(np.dot(thumb_in_plane / np.linalg.norm(thumb_in_plane), palm_lateral))
    return float(np.clip(0.5 * (signed + 1.0), 0.0, 1.0))


def _retarget_one_hand_joints(
    hand_poses: dict[str, np.ndarray],
    side: str,
    prev: dict[str, float] | None,
) -> dict[str, float]:
    """Map OpenXR finger bones to Sberchel proximal joint targets for one hand."""
    out: dict[str, float] = {}
    prev = prev or {}

    for openxr_digit, robot_digit in _DIGIT_STEMS:
        joint_name = f"{side}_{robot_digit}_proximal_joint"
        flex = _digit_proximal_flexion(hand_poses, openxr_digit)
        if flex is None:
            out[joint_name] = prev.get(joint_name, 0.0)
        else:
            out[joint_name] = _clip_joint(flex, _DIGIT_PROXIMAL_LIMIT)

    pitch_name = f"{side}_thumb_proximal_pitch_joint"
    pitch = _thumb_proximal_pitch(hand_poses)
    if pitch is None:
        out[pitch_name] = prev.get(pitch_name, 0.0)
    else:
        out[pitch_name] = _clip_joint(pitch, _THUMB_PITCH_LIMIT)

    yaw_name = f"{side}_thumb_proximal_yaw_joint"
    opposition = _thumb_opposition(hand_poses)
    if opposition is None:
        out[yaw_name] = prev.get(yaw_name, 0.0)
    else:
        lo, hi = _THUMB_YAW_LIMIT
        out[yaw_name] = _clip_joint(lo + opposition * (hi - lo), _THUMB_YAW_LIMIT)

    return out


def _pack_hand_joint_targets(
    hand_joint_names: list[str],
    left_targets: dict[str, float],
    right_targets: dict[str, float],
) -> np.ndarray:
    targets = np.zeros(len(hand_joint_names), dtype=np.float32)
    merged = {**left_targets, **right_targets}
    for i, name in enumerate(hand_joint_names):
        if name in merged:
            targets[i] = merged[name]
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
    """Absolute OpenXR wrist poses + joint-to-joint proximal finger mapping."""

    def __init__(self, cfg: SberchelRetargeterCfg):
        RetargeterBase.__init__(self, cfg)
        self._hand_joint_names = cfg.hand_joint_names
        self._enable_visualization = cfg.enable_visualization
        self._num_open_xr_hand_joints = cfg.num_open_xr_hand_joints
        self._sim_device = cfg.sim_device
        self._prev_left_fingers: dict[str, float] | None = None
        self._prev_right_fingers: dict[str, float] | None = None

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
            f"joint-to-joint fingers ({len(self._hand_joint_names)}): {self._hand_joint_names}",
            flush=True,
        )

    def recalibrate_wrists(self) -> None:
        """No-op: absolute wrist retargeting has no calibration reference."""
        return

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

        left_raw = _retarget_one_hand_joints(left_hand_poses, "left", self._prev_left_fingers)
        right_raw = _retarget_one_hand_joints(right_hand_poses, "right", self._prev_right_fingers)
        left_targets = _blend_targets(self._prev_left_fingers, left_raw, _FINGER_BLEND)
        right_targets = _blend_targets(self._prev_right_fingers, right_raw, _FINGER_BLEND)
        self._prev_left_fingers = left_targets
        self._prev_right_fingers = right_targets

        retargeted_hand_joints = _pack_hand_joint_targets(
            self._hand_joint_names, left_targets, right_targets
        )
        if not hasattr(self, "_finger_log_counter"):
            self._finger_log_counter = 0
        self._finger_log_counter += 1
        if self._finger_log_counter <= 3 or self._finger_log_counter % 120 == 0:
            msg = (
                f"[SberchelRetargeter] fingers packed="
                f"{np.array2string(retargeted_hand_joints, precision=2)}"
            )
            print(msg, flush=True)
            logger.info(msg)

        # Absolute world poses: OpenXR wrist * hand_base frame offset C.
        left_wrist_pose = _apply_hand_base_frame_offset(left_wrist, _LEFT_WRIST_FRAME_OFFSET_WXYZ)
        right_wrist_pose = _apply_hand_base_frame_offset(right_wrist, _RIGHT_WRIST_FRAME_OFFSET_WXYZ)

        left_wrist_tensor = torch.tensor(left_wrist_pose, dtype=torch.float32, device=self._sim_device)
        right_wrist_tensor = torch.tensor(right_wrist_pose, dtype=torch.float32, device=self._sim_device)
        hand_joints_tensor = torch.tensor(retargeted_hand_joints, dtype=torch.float32, device=self._sim_device)

        return torch.cat([left_wrist_tensor, right_wrist_tensor, hand_joints_tensor])


@dataclass
class SberchelRetargeterCfg(GR1T2RetargeterCfg):
    """Sberchel retargeter config.

    ``full_robot_urdf_path`` / ``dex_cache_dir`` are unused (kept for older env
    cfgs that still pass them). Fingers use analytic OpenXR joint mapping.
    """

    full_robot_urdf_path: str = ""
    dex_cache_dir: str = ""
    # Kept for backward compatibility with older env cfgs; unused by absolute retargeting.
    left_idle_wrist_pose: tuple[float, ...] = (0.0,) * 7
    right_idle_wrist_pose: tuple[float, ...] = (0.0,) * 7
    retargeter_type: type = SberchelRetargeter
