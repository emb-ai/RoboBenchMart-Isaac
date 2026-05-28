import argparse
from isaaclab.app import AppLauncher
import random
SEED: int = 42
random.seed(SEED)

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env_cfg import FrankaCubeStackIKRelMimicEnvCfg
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
import isaaclab.utils.math as PoseUtils

from curobo.geom.types import WorldConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml

import omni.usd
import omni.appwindow
import carb
from pxr import UsdGeom, Usd, Gf
from collections.abc import Sequence

import gymnasium as gym
import sys
sys.path.append('.')
from dsynth_isaac.tasks.manager_based.fetch_base_env_cfg import FetchBaseEnvCfg, FetchIKBaseEnvCfg
from dsynth_isaac.tasks.manager_based.panda_base_env_cfg import PandaBaseEnvCfg, PandaDsynthIKRelMimicEnvCfg
import dsynth_isaac.tasks.manager_based
from dsynth_isaac import DSYNTH_ISAAC_PATH
from dsynth_isaac.scene.scene_manager import SceneManager
from dsynth_isaac.planning.utils import compute_box_grasp_thin_side_info, build_grasp_pose
from dsynth_isaac.planning.curobo_planner import DsCuroboPlanner

GRIPPER_OPEN_CMD: float = 1.0
GRIPPER_CLOSE_CMD: float = -1.0

def _eef_name(env: ManagerBasedEnv) -> str:
    return list(env.cfg.subtask_configs.keys())[0]


def _action_from_pose(
    env: ManagerBasedEnv, target_pose: torch.Tensor, base_pose: torch.Tensor, gripper_binary_action: float, env_id: int = 0
) -> torch.Tensor:
    eef = _eef_name(env)
    target_pose_wrt_world = base_pose @ target_pose
    # target_pose_wrt_world = target_pose
    print(f"target_pose_wrt_world: {target_pose_wrt_world[:3, 3]}")
    play_action = target_eef_pose_to_action(
        env=env,
        target_eef_pose_dict={eef: target_pose_wrt_world},
        gripper_action_dict={eef: torch.tensor([gripper_binary_action], device=env.device, dtype=torch.float32)},
        env_id=env_id,
    )
    if play_action.dim() == 1:
        play_action = play_action.unsqueeze(0)
    return play_action


def _env_step_with_action(env: ManagerBasedEnv, action: torch.Tensor) -> None:
    env.step(action)

def _rotation_error_angle(R_target: torch.Tensor, R_current: torch.Tensor) -> torch.Tensor:
    """Return geodesic rotation error angle in radians."""
    R_rel = R_target @ R_current.transpose(-1, -2)
    trace = torch.trace(R_rel)
    cos_theta = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    return torch.acos(cos_theta)

def _drive_to_pose_until_reached(
    env: ManagerBasedEnv,
    target_pose_wrt_base: torch.Tensor,
    base_pose_w: torch.Tensor,
    gripper_binary_action: float,
    env_id: int = 0,
    pos_tol: float = 0.01,
    rot_tol_rad: float = 0.08,
    max_steps: int = 40,
) -> None:
    """Track one waypoint with repeated IK commands until convergence."""
    eef_name = _eef_name(env)
    for _ in range(max_steps):
        action = _action_from_pose(env, target_pose_wrt_base, base_pose_w, gripper_binary_action, env_id=env_id)
        _env_step_with_action(env, action)

        curr_pose_w = get_robot_eef_pose(env, eef_name)[0][env_id]
        curr_pose_b = base_pose_w.inverse() @ curr_pose_w
        pos_err = torch.linalg.vector_norm(target_pose_wrt_base[:3, 3] - curr_pose_b[:3, 3])
        rot_err = _rotation_error_angle(target_pose_wrt_base[:3, :3], curr_pose_b[:3, :3])
        if float(pos_err) < pos_tol and float(rot_err) < rot_tol_rad:
            break

def _pause_simulation_until_play(
    env: ManagerBasedEnv,
    message: str = "Simulation paused. Focus viewport and press R to continue...",
) -> None:
    """Pause script-driven stepping while keeping viewport interactive."""
    del env  # pause behavior is independent from env internals
    print(message)
    app_window = omni.appwindow.get_default_app_window()
    if app_window is None:
        print("No app window found; continuing without pause.")
        return

    input_iface = carb.input.acquire_input_interface()
    keyboard = app_window.get_keyboard()
    if keyboard is None:
        print("No keyboard detected; continuing without pause.")
        return

    should_resume = False

    def _on_keyboard_event(event, *args, **kwargs):
        nonlocal should_resume
        if (
            event.type == carb.input.KeyboardEventType.KEY_PRESS
            and event.input == carb.input.KeyboardInput.R
        ):
            should_resume = True
        return True

    sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_keyboard_event)
    while simulation_app.is_running() and not should_resume:
        simulation_app.update()
    input_iface.unsubscribe_to_keyboard_events(keyboard, sub)

def get_robot_eef_pose(env, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    """
    Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

    Args:
        eef_name: Name of the end effector.
        env_ids: Environment indices to get the pose for. If None, all envs are considered.

    Returns:
        A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
    """
    if env_ids is None:
        env_ids = slice(None)

    # Retrieve end effector pose from the observation buffer
    eef_pos_base = env.obs_buf["policy"]["eef_pos_base"][env_ids]
    eef_quat_base = env.obs_buf["policy"]["eef_quat_base"][env_ids]

    eef_pos = env.obs_buf["policy"]["eef_pos"][env_ids]
    eef_quat = env.obs_buf["policy"]["eef_quat"][env_ids]
    # Quaternion format is w,x,y,z
    return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat)), \
        PoseUtils.make_pose(eef_pos_base, PoseUtils.matrix_from_quat(eef_quat_base))

def get_base_pose(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    if env_ids is None:
        env_ids = slice(None)
    base_pos = env.obs_buf["policy"]["base_pos"][env_ids]
    base_quat = env.obs_buf["policy"]["base_quat"][env_ids]
    return PoseUtils.make_pose(base_pos, PoseUtils.matrix_from_quat(base_quat))


class DsynthCuroboPlannerCfg(CuroboPlannerCfg):

    def _get_world_config_with_table_adjustment(self) -> WorldConfig:
        """Load world config with standard table adjustments.

        This is a helper method that implements the common pattern of adjusting
        table height and combining mesh/cuboid worlds. Used by specific task configs.

        Returns:
            WorldConfig: World configuration with adjusted table
        """
        # Load the base world config
        # world_cfg_table = WorldConfig.from_dict(load_yaml(self.world_config_file))
        world_cfg_table = WorldConfig.from_dict({})

        # Adjust table height if cuboid exists and has a pose
        if world_cfg_table.cuboid and len(world_cfg_table.cuboid) > 0 and world_cfg_table.cuboid[0].pose:
            world_cfg_table.cuboid[0].pose[2] -= 0.02

        # Get mesh world for additional collision objects
        # world_cfg_mesh = WorldConfig.from_dict(
        #     load_yaml(join_path(get_world_configs_path(), self.world_config_file))
        # ).get_mesh_world()
        world_cfg_mesh = WorldConfig.from_dict(
            {}
        ).get_mesh_world()

        # Adjust mesh configuration if it exists
        if world_cfg_mesh.mesh and len(world_cfg_mesh.mesh) > 0:
            mesh_obj = world_cfg_mesh.mesh[0]
            if mesh_obj.name:
                mesh_obj.name += "_mesh"
            if mesh_obj.pose:
                mesh_obj.pose[2] = -10.5  # Move mesh below scene

        # Combine cuboid and mesh worlds
        world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg_mesh.mesh)
        return world_cfg


    @classmethod
    def franka_dsynth_config(cls) -> "CuroboPlannerCfg":
        """Create configuration for Franka stacking a normal cube."""
        config = cls.franka_config()
        config.static_objects = []
        config.visualize_plan = True
        config.debug_planner = True
        config.motion_noise_scale = 0.02
        config.collision_activation_distance = 0.02
        config.retreat_distance = 0.05
        config.approach_distance = 0.05
        config.surface_sphere_radius = 0.01
        # config.world_ignore_substrings = config.world_ignore_substrings + [
        #     '/World/envs/env_0/dsynth_scene/wall_0',
        #     '/World/envs/env_0/dsynth_scene/wall_1',
        #     '/World/envs/env_0/dsynth_scene/wall_2',
            # '/World/envs/env_0/dsynth_scene/wall_3'
            # ]
        config.world_config_file = f'{DSYNTH_ISAAC_PATH}/planning/dsynth_scene.yaml'
        config.get_world_config = lambda: config._get_world_config_with_table_adjustment()
        return config

def _pose_from_xy_quat(xy: torch.Tensor, z: float, quat: torch.Tensor) -> torch.Tensor:
    """Build a 4×4 pose given xy (Tensor[2]), z, and quaternion."""
    device = xy.device
    dtype = xy.dtype
    pos = torch.cat([xy, torch.tensor([z], dtype=dtype, device=device)])
    rot = math_utils.matrix_from_quat(quat.to(device).unsqueeze(0))[0]
    return math_utils.make_pose(pos, rot)

def np_make_pose(pos, quat):
    return math_utils.make_pose(torch.tensor(pos), math_utils.matrix_from_quat(torch.tensor(quat))).cpu().numpy()

def get_item_bbox(prim_path: str):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    T_world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())  # Gf.Matrix4d
    prim_world_pos = T_world.ExtractTranslation()
    prim_world_quat = T_world.ExtractRotationQuat()
    prim_world_quat = [prim_world_quat.GetReal(), *prim_world_quat.GetImaginary()]
    T = np_make_pose(prim_world_pos, prim_world_quat)

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    # Object-space bbox (before world/parent transforms)
    world_bbox = bbox_cache.ComputeWorldBound(prim)
    aabb = world_bbox.ComputeAlignedBox()
    bbox_min = np.array(aabb.GetMin())      # (x_min, y_min, z_min)
    bbox_max = np.array(aabb.GetMax())      
    bbox_center = 0.5 * (bbox_min + bbox_max)
    T[:3, 3] = bbox_center
    
    local_bbox = bbox_cache.ComputeLocalBound(prim)
    local_aabb = local_bbox.ComputeAlignedBox()
    bbox_min_local = np.array(local_aabb.GetMin())
    bbox_max_local = np.array(local_aabb.GetMax())
    bbox_extents_local = bbox_max_local - bbox_min_local
    
    target_bbox = {
        "center": bbox_center,
        "transform": T,
        "extents": bbox_extents_local
    }
    print(target_bbox)
    return target_bbox

def target_eef_pose_to_action(
        env,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """
        Takes a target pose and gripper action for the end effector controller and returns an action
        (usually a normalized delta pose action) to try and achieve that target pose.
        Noise is added to the target pose action if specified.

        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            noise: Noise to add to the action. If None, no noise is added.
            env_id: Environment index to get the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().
        """
        robot = env.scene["robot"]
        root_pos_w = robot.data.root_pos_w[0]
        root_quat_w = robot.data.root_quat_w[0]

        eef_name = list(env.cfg.subtask_configs.keys())[0]

        # target position and rotation
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)
        target_quat = PoseUtils.quat_from_matrix(target_rot)

        target_pos_b, target_quat_b = PoseUtils.subtract_frame_transforms(
            root_pos_w.unsqueeze(0),
            root_quat_w.unsqueeze(0),
            target_pos.unsqueeze(0),
            target_quat.unsqueeze(0),
        )

        target_rot_b = PoseUtils.matrix_from_quat(target_quat_b)[0]

        # current position and rotation
        curr_pose = env.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)
        curr_quat = PoseUtils.quat_from_matrix(curr_rot.unsqueeze(0))[0]
        curr_pos_b, curr_quat_b = PoseUtils.subtract_frame_transforms(
            root_pos_w.unsqueeze(0),
            root_quat_w.unsqueeze(0),
            curr_pos.unsqueeze(0),
            curr_quat.unsqueeze(0),
        )
        curr_rot_b = PoseUtils.matrix_from_quat(curr_quat_b)[0]

        # normalized delta position action
        delta_position = target_pos_b - curr_pos_b

        # normalized delta rotation action
        delta_rot_mat = target_rot_b.matmul(curr_rot_b.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        # get gripper action for single eef
        (gripper_action,) = gripper_action_dict.values()

        # add noise to action
        pose_action = torch.cat([delta_position[0], delta_rotation], dim=0)
        if action_noise_dict is not None:
            noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        return torch.cat([pose_action, gripper_action], dim=0)

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    headless = False

    env_cfg = PandaDsynthIKRelMimicEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.dsynth_scene.spawn.config_dir_path = "generated_envs/isaac_sample"
    env_cfg.disable_fabric = True
    env_cfg.viewer.origin_type = 'asset_root'
    env_cfg.viewer.asset_name = 'robot'
    env_cfg.viewer.eye = (1.0, 1.0, 1.0)
    env_cfg.viewer.lookat = (0.4, 0.0, 0.6)

    for frame in env_cfg.scene.ee_frame.target_frames:
        if frame.name == "end_effector":
            print(f"Setting end effector offset from {frame.offset.pos} to (0.0, 0.0, 0.0) for SkillGen parity")
            frame.offset.pos = (0.0, 0.0, 0.0)
    
    env: ManagerBasedEnv = gym.make(
        "Isaac-Dsynth-Panda-IKRelMimic-v0",
        cfg=env_cfg,
        headless=headless,
    ).unwrapped
    env.reset()

    manager: SceneManager = env.scene.cfg.dsynth_scene.spawn.manager

    target_shelf_name = "isaac_sample_addfa169_0_mixed_shelf"
    target_item_name = "food_CEREALS_NestleHoneyStars_0_1_0_0"
    shelving_xforms = manager.get_active_shelvings_xforms_poses()
    item_entities = manager.get_item_entities()
    target_shelf_pose = shelving_xforms['env_0'][target_shelf_name]
    target_rigid_object = item_entities[f'env_0'][target_shelf_name][target_item_name]
    target_shelf_direction = math_utils.make_pose(target_shelf_pose[0], math_utils.matrix_from_quat(target_shelf_pose[1]))[0, :3, 1]

    target_path = target_rigid_object.cfg.prim_path
    target_bbox = get_item_bbox(target_path)
    grasp_infos = compute_box_grasp_thin_side_info(
        target_bbox, 
        ee_direction=target_shelf_direction,
        depth=0.02,
    )
    print(grasp_infos)
    grasp_closing, grasp_center, grasp_approaching = grasp_infos["closing"], grasp_infos["center"], grasp_infos["approaching"]
    grasp_pose = build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)
    grasp_pose = torch.tensor(grasp_pose, dtype=torch.float32, device=target_rigid_object.device)
    print(f"grasp pose: {grasp_pose}")

    robot: Articulation = env.scene["robot"]
    planner_cfg = DsynthCuroboPlannerCfg.franka_dsynth_config()
    # planner_cfg = CuroboPlannerCfg.`franka_stack_cube_config`()
    planner_cfg.visualize_plan = True
    planner_cfg.visualize_spheres = True
    planner_cfg.debug_planner = True
    planner_cfg.retreat_distance = 0.05
    planner_cfg.approach_distance = 0.05
    planner_cfg.time_dilation_factor = 1.0

    planner = DsCuroboPlanner(
        env=env,
        robot=robot,
        config=planner_cfg,
        env_id=0,
    )
    
    goal_pose_visualizer = None
    if not headless:
        marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/goal_pose")
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        goal_pose_visualizer = VisualizationMarkers(marker_cfg)
    
   
    base_pose = get_base_pose(env)[0]
    print(f"base pose: {base_pose}")
    curr_pose, curr_pose_wrt_base = get_robot_eef_pose(env, _eef_name(env))
    curr_pose, curr_pose_wrt_base = curr_pose[0], curr_pose_wrt_base[0]

    target_shift = torch.eye(4, device=curr_pose_wrt_base.device)
    target_shift[2, 3] = -0.1
    gripper_offset = target_shift.clone()
    # target_pose = curr_pose @ target_shift
    pre_grasp_pose = grasp_pose @ target_shift
    # target_pose_wrt_base = curr_pose_wrt_base @ target_shift
    # target_pose_wrt_base = base_pose.inverse() @ target_pose
    pre_grasp_pose_wrt_base = base_pose.inverse() @ pre_grasp_pose
    # print(f"current pose: {target_pose}")
    # print(f"Pre-grasp pose: {target_pose_wrt_base}")
    if not headless:
        pos_pg = pre_grasp_pose[:3, 3].detach().cpu()
        quat_pg = math_utils.quat_from_matrix(pre_grasp_pose[:3, :3].unsqueeze(0))[0].detach().cpu()
        # self._visualize_goal_pose(pos_pg, quat_pg)
        goal_pose_visualizer.visualize(translations=pos_pg.unsqueeze(0), orientations=quat_pg.unsqueeze(0))


    _pause_simulation_until_play(env, "Paused before planning. Focus viewport and press R to continue...")
    res = planner.update_world_and_plan_motion(pre_grasp_pose_wrt_base @ gripper_offset)
    if not res:
        print("Failed to plan to pre-grasp pose")
    else:
        # EXECUTE
        print("Executing planned poses")
        planned_poses = planner.get_planned_poses()
        if not planned_poses:
            return

        # eef_name = _eef_name(env)
        # curr_pose = env.get_robot_eef_pose(eef_name, env_ids=[0])[0]
        # curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)
        # # curr_rot = PoseUtils.axis_angle_from_quat(PoseUtils.quat_from_matrix(curr_rot))
        # curr_pos[0] += 0.1
        # curr_pose = PoseUtils.make_pose(curr_pos, curr_rot)
        # curr_pose_wrt_base = base_pose.inverse() @ curr_pose 

        # pos_pg = curr_pose[:3, 3].detach().cpu()
        # quat_pg = math_utils.quat_from_matrix(curr_pose[:3, :3].unsqueeze(0))[0].detach().cpu()
        # # self._visualize_goal_pose(pos_pg, quat_pg)
        # goal_pose_visualizer.visualize(translations=pos_pg.unsqueeze(0), orientations=quat_pg.unsqueeze(0))
        _pause_simulation_until_play(env, "Paused before planning. Focus viewport and press R to continue...")
        for pose in planned_poses:
            print("pose: ", pose[:3, 3], )
            _drive_to_pose_until_reached(
                env=env,
                target_pose_wrt_base=pose,
                base_pose_w=base_pose,
                gripper_binary_action=GRIPPER_OPEN_CMD,
                env_id=0,
                max_steps=10,
            )
            print("robot eef pose: ", get_robot_eef_pose(env, _eef_name(env))[0][0, :3, 3])

        # print("target_pose_wrt_base: ", target_pose_wrt_base)
        # print("target_pose: ", target_pose[:3, 3])
        print("Executing finished")

    n_tries = 20
    grasp_infos = compute_box_grasp_thin_side_info(
        target_bbox, 
        ee_direction=target_shelf_direction,
        depth=0.02,
        n_grasps=n_tries,
        approaching_angles_range=[-np.pi/4, np.pi/4],
    )
    for n_try, grasp_info in enumerate(grasp_infos):
        grasp_closing, grasp_center, grasp_approaching = grasp_info["closing"], grasp_info["center"], grasp_info["approaching"]
        grasp_pose = build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)
        grasp_pose = torch.tensor(grasp_pose, dtype=torch.float32, device=target_rigid_object.device)

        grasp_pose_wrt_base = base_pose.inverse() @ grasp_pose
        # print(f"current pose: {target_pose}")
        # print(f"Pre-grasp pose: {target_pose_wrt_base}")
        if not headless:
            pos_pg = grasp_pose[:3, 3].detach().cpu()
            quat_pg = math_utils.quat_from_matrix(grasp_pose[:3, :3].unsqueeze(0))[0].detach().cpu()
            # self._visualize_goal_pose(pos_pg, quat_pg)
            goal_pose_visualizer.visualize(translations=pos_pg.unsqueeze(0), orientations=quat_pg.unsqueeze(0))

    # _pause_simulation_until_play(env, "Paused before planning. Focus viewport and press R to continue...")
        res = planner.update_world_and_plan_motion(grasp_pose_wrt_base @ gripper_offset)
        if not res:
            print(f"Failed to plan to pre-grasp pose! Try #{n_try + 1} of {n_tries}")
        else:
            print(f"Successed grasp!`")
            # EXECUTE
            print("Executing planned poses")
            planned_poses = planner.get_planned_poses()
            if not planned_poses:
                return

            _pause_simulation_until_play(env, "Paused before planning. Focus viewport and press R to continue...")
            for pose in planned_poses:
                print("pose: ", pose[:3, 3], )
                _drive_to_pose_until_reached(
                    env=env,
                    target_pose_wrt_base=pose,
                    base_pose_w=base_pose,
                    gripper_binary_action=GRIPPER_OPEN_CMD,
                    env_id=0,
                )
                print("robot eef pose: ", get_robot_eef_pose(env, _eef_name(env))[0][0, :3, 3])

            print("Executing finished")
            break

    
    while simulation_app.is_running():
        with torch.inference_mode():
            action = torch.zeros_like(env.action_manager.action)
            # action = env.action_manager.action
            # action[:, -2] = 10.0
            # action[:, -1] = 10.0
            # action[0, 0] =  0.0
            # action[0, 1] =  0.00
            # action[0, 2] =  0.0
            # action[0, 3] =  0.0
            # action[0, 4] =  0.0
            # action[0, 5] =  0.0
            # action[0, 6] = 0.0
# tensor([[-0.6635,  0.6894,  0.2776, -0.0860]], device='cuda:0')
            _ = env.step(action)
    # target_product_pose = 
    
    

if __name__ == "__main__":
    main()