from __future__ import annotations

import math
import random
import torch
import re
from typing import TYPE_CHECKING

import omni.usd
from pxr import UsdGeom, Usd, Gf
import carb
import omni.physics.tensors.impl.api as physx
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt

from isaaclab.managers import SceneEntityCfg
from isaacsim.core.prims import RigidPrim
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaacsim.core.utils.extensions import enable_extension
import isaaclab.sim as sim_utils
from isaaclab.utils.version import compare_versions
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.assets import RigidObject
import isaaclab.utils.math as math_utils

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