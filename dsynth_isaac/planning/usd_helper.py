#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# Standard Library
import math
from typing import Dict, List, Optional, Union, Tuple

# Third Party
import numpy as np
import torch
from tqdm import tqdm

from isaaclab.assets import Articulation
import isaaclab.utils.math as PoseUtils
# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.geom.types import (
    Capsule,
    Cuboid,
    Cylinder,
    Material,
    Mesh,
    Obstacle,
    Sphere,
    WorldConfig,
)
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util_file import (
    file_exists,
    get_assets_path,
    get_filename,
    get_files_from_dir,
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import MotionGenResult

try:
    # Third Party
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    raise ImportError(
        "usd-core failed to import, install with pip install usd-core"
        + " NOTE: Do not install this if using with ISAAC SIM."
    )


from curobo.util.usd_helper import (
    UsdHelper,
    get_prim_world_pose,
    get_cylinder_attrs,
    get_capsule_attrs,
    get_cube_attrs,
    get_sphere_attrs,
    get_mesh_attrs,
)


class DsUsdHelper(UsdHelper):
    """Dsynth specific USD helper."""
    def load_robot(self, robot: Articulation):
        self.robot = robot

    def get_obstacles_from_stage(
        self,
        only_paths: Optional[List[str]] = None,
        ignore_paths: Optional[List[str]] = None,
        only_substring: Optional[List[str]] = None,
        ignore_substring: Optional[List[str]] = None,
        reference_prim_path: Optional[str] = None,
        timecode: float = 0,
    ) -> WorldConfig:
        # read obstacles from usd by iterating through all prims:
        obstacles = {"cuboid": [], "sphere": None, "mesh": None, "cylinder": None, "capsule": None}
        r_T_w = None
        self._xform_cache.Clear()
        self._xform_cache.SetTime(timecode)
        # if reference_prim_path is not None:
        #     reference_prim = self.stage.GetPrimAtPath(reference_prim_path)
        #     r_T_w, _ = get_prim_world_pose(self._xform_cache, reference_prim, inverse=True)
        w_T_r = PoseUtils.make_pose(self.robot.data.root_pos_w, PoseUtils.matrix_from_quat(self.robot.data.root_quat_w))
        r_T_w = w_T_r.inverse().cpu().numpy()[0]
            
        all_items = self.stage.Traverse()
        for x in all_items:
            if only_paths is not None:
                if not any([str(x.GetPath()).startswith(k) for k in only_paths]):
                    continue
            if ignore_paths is not None:
                if any([str(x.GetPath()).startswith(k) for k in ignore_paths]):
                    continue
            if only_substring is not None:
                if not any([k in str(x.GetPath()) for k in only_substring]):
                    continue
            if ignore_substring is not None:
                if any([k in str(x.GetPath()) for k in ignore_substring]):
                    continue
            if x.IsA(UsdGeom.Cube):
                if obstacles["cuboid"] is None:
                    obstacles["cuboid"] = []
                cube = get_cube_attrs(x, cache=self._xform_cache, transform=r_T_w)
                obstacles["cuboid"].append(cube)
            elif x.IsA(UsdGeom.Sphere):
                if obstacles["sphere"] is None:
                    obstacles["sphere"] = []
                obstacles["sphere"].append(
                    get_sphere_attrs(x, cache=self._xform_cache, transform=r_T_w)
                )
            elif x.IsA(UsdGeom.Mesh):
                if obstacles["mesh"] is None:
                    obstacles["mesh"] = []
                m_data = get_mesh_attrs(x, cache=self._xform_cache, transform=r_T_w)
                if m_data is not None:
                    obstacles["mesh"].append(m_data)
            elif x.IsA(UsdGeom.Cylinder):
                if obstacles["cylinder"] is None:
                    obstacles["cylinder"] = []
                cube = get_cylinder_attrs(x, cache=self._xform_cache, transform=r_T_w)
                obstacles["cylinder"].append(cube)
            elif x.IsA(UsdGeom.Capsule):
                if obstacles["capsule"] is None:
                    obstacles["capsule"] = []
                cap = get_capsule_attrs(x, cache=self._xform_cache, transform=r_T_w)
                obstacles["capsule"].append(cap)
        world_model = WorldConfig(**obstacles)
        return world_model
