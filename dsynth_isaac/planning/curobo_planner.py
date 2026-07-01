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
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation

from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
from isaaclab_mimic.motion_planners.motion_planner_base import MotionPlannerBase
from isaaclab_mimic.motion_planners.curobo.curobo_planner import PlannerLogger, Attachment, CuroboPlanner

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

    def _sync_object_poses_with_isaaclab(self) -> None:
        """Synchronize cuRobo collision world with Isaac Lab object positions.

        Updates all dynamic object poses in cuRobo's world model to match their current
        positions in Isaac Lab. This ensures accurate collision checking after simulation
        steps or manual object movements. Static objects (bins, tables, walls) are skipped
        for performance as they shouldn't move during simulation.

        The method updates both the world model and the collision checker to ensure
        consistency across all cuRobo components.
        """
        # Get cached object mappings and world model
        object_mappings = self._get_object_mappings()
        world_model = self.motion_gen.world_coll_checker.world_model
        rigid_objects = self.env.scene.rigid_objects

        updated_count = 0

        env_origin = self.env.scene.env_origins
        robot_pos = self.robot.data.root_pos_w - env_origin
        robot_quat = self.robot.data.root_quat_w

        for object_name, object_path in object_mappings.items():
            if object_name not in rigid_objects:
                continue

            # Skip static mesh objects - they should not be dynamically updated
            static_objects = getattr(self.config, "static_objects", [])
            if any(static_name in object_name.lower() for static_name in static_objects):
                self.logger.debug(f"SYNC: Skipping static object {object_name}")
                continue

            # Get current pose from Lab (may be on CPU or CUDA depending on --device flag)
            obj = rigid_objects[object_name]
            # env_origin = self.env.scene.env_origins[self.env_id]
            # current_pos_raw = obj.data.root_pos_w[self.env_id] - env_origin
            # current_quat_raw = obj.data.root_quat_w[self.env_id]  # (w, x, y, z)
            current_pos_raw = obj.data.root_pos_w
            current_quat_raw = obj.data.root_quat_w
            current_pos_raw_b, current_quat_raw_b = math_utils.subtract_frame_transforms(
                robot_pos,
                robot_quat,
                current_pos_raw,
                current_quat_raw
            )

            # Convert to cuRobo device and extract float values for pose list
            current_pos = self._to_curobo_device(current_pos_raw_b[self.env_id])
            current_quat = self._to_curobo_device(current_quat_raw_b[self.env_id])

            # Convert to cuRobo pose format [x, y, z, w, x, y, z]
            pose_list = [
                float(current_pos[0].item()),
                float(current_pos[1].item()),
                float(current_pos[2].item()),
                float(current_quat[0].item()),
                float(current_quat[1].item()),
                float(current_quat[2].item()),
                float(current_quat[3].item()),
            ]

            # Update object pose in cuRobo's world model
            if self._update_object_in_world_model(world_model, object_name, object_path, pose_list):
                updated_count += 1

        self.logger.debug(f"SYNC: Updated {updated_count} object poses in cuRobo world model")

        # Sync object poses with collision checker
        if updated_count > 0:
            # Update individual obstacle poses in collision checker
            # This preserves static mesh objects unlike load_collision_model which rebuilds everything
            for object_name, object_path in object_mappings.items():
                if object_name not in rigid_objects:
                    continue

                # Skip static mesh objects - they should not be dynamically updated
                static_objects = getattr(self.config, "static_objects", [])
                if any(static_name in object_name.lower() for static_name in static_objects):
                    continue

                # Get current pose and update in collision checker
                obj = rigid_objects[object_name]
                current_pos_raw = obj.data.root_pos_w
                current_quat_raw = obj.data.root_quat_w
                current_pos_raw_b, current_quat_raw_b = math_utils.subtract_frame_transforms(
                    robot_pos,
                    robot_quat,
                    current_pos_raw,
                    current_quat_raw
                )

                # Convert to cuRobo device and extract float values for pose list
                current_pos = self._to_curobo_device(current_pos_raw_b[self.env_id])
                current_quat = self._to_curobo_device(current_quat_raw_b[self.env_id])

                # Create cuRobo pose and update collision checker directly
                curobo_pose = self._make_pose(position=current_pos, quaternion=current_quat)
                self.motion_gen.world_coll_checker.update_obstacle_pose(  # type: ignore
                    object_path, curobo_pose, update_cpu_reference=True
                )

            self.logger.debug(f"Updated {updated_count} object poses in collision checker")

    def _plan_to_contact(
        self,
        start_state: JointState,
        goal_pose: Pose,
        retreat_distance: float,
        approach_distance: float,
        contact: bool = False,
        retime_plan: bool = False,
        step_size: float | None = None,
    ) -> bool:
        """Execute multi-phase contact planning with approach and retreat phases.

        Implements a planning strategy for manipulation tasks that require approach and contact handling.
        Plans multiple trajectory segments with different collision checking configurations.

        Args:
            start_state: Starting joint state for planning
            goal_pose: Target pose to reach
            retreat_distance: Distance to retreat before transition to contact
            approach_distance: Distance to approach before final pose
            contact: Whether to enable contact planning mode
            retime_plan: Whether to retime the resulting plan
            step_size: Step size for retiming (only used if retime_plan is True)

        Returns:
            True if all planning phases succeeded, False if any phase failed
        """
        self.logger.debug(f"Multi-phase planning: retreat={retreat_distance}, approach={approach_distance}")

        target_poses: list[Pose] = []
        contacts: list[bool] = []

        if retreat_distance is not None and retreat_distance > 0:
            ee_pose: Pose = self.get_ee_pose(start_state)
            retreat_pose: Pose = ee_pose.multiply(
                self._make_pose(
                    position=[retreat_distance, 0.0, 0.0],
                )
            )
            target_poses.append(retreat_pose)
            contacts.append(True)
        contacts.append(contact)
        if approach_distance is not None and approach_distance > 0:
            approach_pose: Pose = goal_pose.multiply(
                self._make_pose(
                    position=[0.0, 0.0, -approach_distance],
                )
            )
            target_poses.append(approach_pose)
            contacts.append(True)

        target_poses.append(goal_pose)

        current_state: JointState = start_state
        full_plan: JointState | None = None

        for i, (target_pose, contact_flag) in enumerate(zip(target_poses, contacts)):
            self.logger.debug(
                f"Planning phase {i + 1} of {len(target_poses)}: contact={contact_flag} (collision"
                f" {'disabled' if contact_flag else 'enabled'})"
            )

            success: bool = self._plan_to_contact_pose(
                start_state=current_state,
                goal_pose=target_pose,
                contact=contact_flag,
            )

            if not success:
                self.logger.debug(f"Phase {i + 1} planning failed")
                return False

            if full_plan is None:
                full_plan = self._current_plan
            else:
                full_plan = full_plan.stack(self._current_plan)

            last_waypoint: torch.Tensor = self._current_plan.position[-1]
            current_state = JointState(
                position=last_waypoint.unsqueeze(0),
                velocity=torch.zeros_like(last_waypoint.unsqueeze(0)),
                acceleration=torch.zeros_like(last_waypoint.unsqueeze(0)),
                joint_names=self._current_plan.joint_names,
            )
            current_state = current_state.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        self._current_plan = full_plan
        self._plan_index = 0

        if retime_plan and step_size is not None:
            original_length: int = len(self._current_plan.position)
            self._current_plan = self._linearly_retime_plan(step_size=step_size, plan=self._current_plan)
            self.logger.debug(
                f"Retimed complete plan from {original_length} to {len(self._current_plan.position)} waypoints"
            )

        self.logger.debug(f"Multi-phase planning succeeded with {len(self._current_plan.position)} total waypoints")

        return True
