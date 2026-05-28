# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModelState
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

import isaaclab.utils.math as PoseUtils
from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.sim.spawners.meshes import MeshSphereCfg, spawn_mesh_sphere

from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
from isaaclab_mimic.motion_planners.motion_planner_base import MotionPlannerBase
from isaaclab_mimic.motion_planners.curobo.curobo_planner import PlannerLogger, Attachment, CuroboPlanner
from isaaclab.assets import Articulation

from dsynth_isaac.planning.usd_helper import DsUsdHelper

class DsCuroboPlanner(CuroboPlanner):
    """Dsynth specific cuRobo planner."""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.usd_helper: DsUsdHelper = DsUsdHelper()
        self.usd_helper.load_stage(self.env.scene.stage)
        self.usd_helper.load_robot(self.robot)

        if self.visualize_plan:
            from isaaclab_mimic.motion_planners.curobo.plan_visualizer import PlanVisualizer

            # Use env-local base translation for multi-env rendering consistency
            # env_origin = self.env.scene.env_origins[self.env_id, :3]
            # base_translation = (self.robot.data.root_pos_w[self.env_id, :3] - env_origin).detach().cpu().numpy()
            base_translation = np.zeros(3)
            self.plan_visualizer = PlanVisualizer(
                robot_name=self.config.robot_name,
                recording_id=f"curobo_plan_{self.env_id}",
                debug=self.config.debug_planner,
                base_translation=base_translation,
            )

            self.plan_visualizer.set_motion_generator_reference(self.motion_gen)

        self._initialize_static_world()

    def _create_sphere_config(self, sphere_idx: int, sphere, robot_link_count: int) -> dict:
        """Create sphere configuration with position and visual properties for USD rendering.

        Determines sphere type (robot link vs attached object), calculates world position,
        and creates the appropriate visual configuration including colors and materials.
        Robot link spheres are green with lower opacity, while attached object spheres
        are orange with higher opacity for better distinction.

        Args:
            sphere_idx: Index of the sphere in the sphere list
            sphere: Sphere object containing position and radius data
            robot_link_count: Total number of robot link spheres (for type determination)

        Returns:
            Dictionary containing 'position' (world coordinates) and 'cfg' (MeshSphereCfg)
        """

        is_attached = sphere_idx >= robot_link_count
        color = (1.0, 0.5, 0.0) if is_attached else (0.0, 1.0, 0.0)
        opacity = 0.9 if is_attached else 0.5

        # Calculate position in world frame (do not use env_origin)
        root_translation = (self.robot.data.root_pos_w[self.env_id, :3]).detach().cpu().numpy()
        # root_rotation = self.robot.data.root_quat_w[self.env_id, :4]
        root_rotation = PoseUtils.matrix_from_quat(self.robot.data.root_quat_w)[self.env_id].detach().cpu().numpy()
        position = sphere.position.cpu().numpy() if hasattr(sphere.position, "cpu") else sphere.position
        # if not is_attached:
        #     position = root_rotation @ position

        return {
            "position": position,
            "cfg": MeshSphereCfg(
                radius=float(sphere.radius),
                visual_material=PreviewSurfaceCfg(diffuse_color=color, opacity=opacity, emissive_color=color),
            ),
        }