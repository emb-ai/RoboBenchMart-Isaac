from __future__ import annotations

import math
import random
import torch
from typing import TYPE_CHECKING

import omni.usd
from pxr import UsdGeom, Usd, Gf

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # Set the default pose for robots in all envs
    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(env.num_envs, 1)


def set_default_base_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # Set the default pose for robots in all envs
    asset = env.scene[asset_cfg.name]
    
    stage = omni.usd.get_context().get_stage()
    default_positions = []
    default_orientations = []
    env_ids_list = list(range(env.num_envs)) if env_ids is None else env_ids.tolist()
    for env_id in env_ids_list:
        prim_path = f"/World/envs/env_{env_id}/dsynth_scene/robot_default_pose"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            # fallback: use zero pose
            default_positions.append([0.0, 0.0, 0.0])
            default_orientations.append([1.0, 0.0, 0.0, 0.0])
            continue
    
        xformable = UsdGeom.Xformable(prim)
        # world_transform: Gf.Matrix4d = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world_transform: Gf.Matrix4d = xformable.GetLocalTransformation()
        translation = world_transform.ExtractTranslation()   # Gf.Vec3d
        rotation_quat = world_transform.ExtractRotationQuat()  # Gf.Quatd  (real, i, j, k)
        default_positions.append([translation[0], translation[1], translation[2]])
        imag = rotation_quat.GetImaginary()
        default_orientations.append([rotation_quat.GetReal(), imag[0], imag[1], imag[2]])
    
    pos_tensor = torch.tensor(default_positions, device=env.device, dtype=torch.float32)
    quat_tensor = torch.tensor(default_orientations, device=env.device, dtype=torch.float32)
    # Now use pos_tensor / quat_tensor to set the robot's default base pose
    asset.data.default_root_state[env_ids_list, :3] = pos_tensor
    asset.data.default_root_state[env_ids_list, 3:7] = quat_tensor

    # asset.write_root_pose_to_sim(asset.data.default_root_state[:, :7], env_ids=env_ids)
    # asset.write_root_velocity_to_sim(asset.data.default_root_state[:, 7:], env_ids=env_ids)