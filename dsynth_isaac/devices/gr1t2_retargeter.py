"""GR1T2 OpenXR retargeter that uses locally generated hand URDFs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from dex_retargeting.retargeting_config import RetargetingConfig

import isaaclab.utils.math as PoseUtils
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.openxr.retargeters.humanoid.fourier import gr1_t2_dex_retargeting_utils as dex_utils
from isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1t2_retargeter import (
    GR1T2Retargeter,
    GR1T2RetargeterCfg,
)

from dsynth_isaac.devices.gr1t2_hand_urdf import prepare_gr1t2_dex_hand_urdfs, prepare_gr1t2_dex_retargeting_configs

logger = logging.getLogger(__name__)


class DsynthGR1TR2DexRetargeting(dex_utils.GR1TR2DexRetargeting):
    """Dex-retargeting backend that avoids broken Nucleus hand URDF assets."""

    def __init__(
        self,
        hand_joint_names: list[str],
        left_hand_urdf_path: str,
        right_hand_urdf_path: str,
        cache_dir: str | Path,
        right_hand_config_filename: str = "fourier_hand_right_dexpilot.yml",
        left_hand_config_filename: str = "fourier_hand_left_dexpilot.yml",
    ):
        self.isaac_lab_hand_joint_names = hand_joint_names
        left_config_path, right_config_path = prepare_gr1t2_dex_retargeting_configs(cache_dir)

        self._update_yaml_with_urdf_path(left_config_path, left_hand_urdf_path)
        self._update_yaml_with_urdf_path(right_config_path, right_hand_urdf_path)

        self._dex_left_hand = RetargetingConfig.load_from_file(left_config_path).build()
        self._dex_right_hand = RetargetingConfig.load_from_file(right_config_path).build()

        self.left_dof_names = self._dex_left_hand.optimizer.robot.dof_joint_names
        self.right_dof_names = self._dex_right_hand.optimizer.robot.dof_joint_names
        self.dof_names = self.left_dof_names + self.right_dof_names

        logger.info("[DsynthGR1T2DexRetargeting] initialized with local hand URDFs.")

    @staticmethod
    def _update_yaml_with_urdf_path(yaml_path: str, urdf_path: str) -> None:
        with open(yaml_path) as file:
            config = yaml.safe_load(file)
        config["retargeting"]["urdf_path"] = urdf_path
        with open(yaml_path, "w") as file:
            yaml.dump(config, file)


def _pose_matrix_from_7d(pose_7d: np.ndarray) -> torch.Tensor:
    position = torch.tensor(pose_7d[:3], dtype=torch.float32)
    quaternion = torch.tensor(pose_7d[3:], dtype=torch.float32)
    return PoseUtils.make_pose(position, PoseUtils.matrix_from_quat(quaternion))


def _pose_7d_from_matrix(pose_matrix: torch.Tensor) -> np.ndarray:
    position, rotation_matrix = PoseUtils.unmake_pose(pose_matrix)
    quaternion = PoseUtils.quat_from_matrix(rotation_matrix)
    return np.concatenate([position.numpy(), quaternion.numpy()])


class DsynthGR1T2Retargeter(GR1T2Retargeter):
    """GR1T2 retargeter wired to local dex-retargeting hand URDFs."""

    def __init__(self, cfg: DsynthGR1T2RetargeterCfg):
        from isaaclab.devices.retargeter_base import RetargeterBase

        RetargeterBase.__init__(self, cfg)
        self._hand_joint_names = cfg.hand_joint_names
        left_urdf_path, right_urdf_path = prepare_gr1t2_dex_hand_urdfs(cfg.full_robot_urdf_path, cfg.dex_cache_dir)
        self._hands_controller = DsynthGR1TR2DexRetargeting(
            hand_joint_names=self._hand_joint_names,
            left_hand_urdf_path=left_urdf_path,
            right_hand_urdf_path=right_urdf_path,
            cache_dir=cfg.dex_cache_dir,
        )

        self._idle_left_wrist = np.asarray(cfg.left_idle_wrist_pose, dtype=np.float32)
        self._idle_right_wrist = np.asarray(cfg.right_idle_wrist_pose, dtype=np.float32)
        self._wrist_reference: dict[str, np.ndarray | None] = {"left": None, "right": None}

        self._enable_visualization = cfg.enable_visualization
        self._num_open_xr_hand_joints = cfg.num_open_xr_hand_joints
        self._sim_device = cfg.sim_device
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

    def recalibrate_wrists(self) -> None:
        """Clear wrist calibration so the next frame re-centers on the robot idle pose."""
        self._wrist_reference = {"left": None, "right": None}

    def _process_openxr_wrist(self, wrist: np.ndarray, is_right: bool) -> np.ndarray:
        """Convert an OpenXR wrist pose into the USD control frame."""
        if is_right:
            return self._retarget_abs(wrist)
        return wrist.copy()

    def _map_wrist_to_robot(self, wrist: np.ndarray, is_right: bool) -> np.ndarray:
        """Map tracked wrist motion to the robot workspace via calibrated deltas."""
        processed = self._process_openxr_wrist(wrist, is_right)
        side = "right" if is_right else "left"
        idle_pose = self._idle_right_wrist if is_right else self._idle_left_wrist

        if self._wrist_reference[side] is None:
            self._wrist_reference[side] = processed.copy()
            return idle_pose.copy()

        reference_pose = self._wrist_reference[side]
        reference_transform = _pose_matrix_from_7d(reference_pose)
        current_transform = _pose_matrix_from_7d(processed)
        idle_transform = _pose_matrix_from_7d(idle_pose)

        delta_transform = torch.matmul(PoseUtils.pose_inv(reference_transform), current_transform)
        target_transform = torch.matmul(idle_transform, delta_transform)
        return _pose_7d_from_matrix(target_transform)

    def retarget(self, data: dict) -> torch.Tensor:
        """Map OpenXR hand poses to Pink IK wrist targets and dex-retargeted finger joints."""
        left_hand_poses = data[DeviceBase.TrackingTarget.HAND_LEFT]
        right_hand_poses = data[DeviceBase.TrackingTarget.HAND_RIGHT]

        left_wrist = left_hand_poses.get("wrist")
        right_wrist = right_hand_poses.get("wrist")

        if self._enable_visualization:
            joints_position = np.zeros((self._num_open_xr_hand_joints, 3))
            joints_position[::2] = np.array([pose[:3] for pose in left_hand_poses.values()])
            joints_position[1::2] = np.array([pose[:3] for pose in right_hand_poses.values()])
            self._markers.visualize(translations=torch.tensor(joints_position, device=self._sim_device))

        left_hands_pos = self._hands_controller.compute_left(left_hand_poses)
        left_indexes = [self._hand_joint_names.index(name) for name in self._hands_controller.get_left_joint_names()]
        left_retargeted_hand_joints = np.zeros(len(self._hands_controller.get_joint_names()))
        left_retargeted_hand_joints[left_indexes] = left_hands_pos
        left_hand_joints = left_retargeted_hand_joints

        right_hands_pos = self._hands_controller.compute_right(right_hand_poses)
        right_indexes = [self._hand_joint_names.index(name) for name in self._hands_controller.get_right_joint_names()]
        right_retargeted_hand_joints = np.zeros(len(self._hands_controller.get_joint_names()))
        right_retargeted_hand_joints[right_indexes] = right_hands_pos
        right_hand_joints = right_retargeted_hand_joints
        retargeted_hand_joints = left_hand_joints + right_hand_joints

        left_wrist_pose = self._map_wrist_to_robot(left_wrist, is_right=False)
        right_wrist_pose = self._map_wrist_to_robot(right_wrist, is_right=True)

        left_wrist_tensor = torch.tensor(left_wrist_pose, dtype=torch.float32, device=self._sim_device)
        right_wrist_tensor = torch.tensor(right_wrist_pose, dtype=torch.float32, device=self._sim_device)
        hand_joints_tensor = torch.tensor(retargeted_hand_joints, dtype=torch.float32, device=self._sim_device)

        return torch.cat([left_wrist_tensor, right_wrist_tensor, hand_joints_tensor])


@dataclass
class DsynthGR1T2RetargeterCfg(GR1T2RetargeterCfg):
    """GR1T2 retargeter config with paths to locally generated dex-retargeting assets."""

    full_robot_urdf_path: str = ""
    dex_cache_dir: str = ""
    left_idle_wrist_pose: tuple[float, ...] = (0.0,) * 7
    right_idle_wrist_pose: tuple[float, ...] = (0.0,) * 7
    retargeter_type: type = DsynthGR1T2Retargeter
