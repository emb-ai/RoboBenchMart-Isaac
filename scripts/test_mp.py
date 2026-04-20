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

import omni.usd
from pxr import UsdGeom, Usd, Gf
from collections.abc import Sequence

import gymnasium as gym
import sys
sys.path.append('.')
from dsynth_isaac.tasks.manager_based.fetch_base_env_cfg import FetchBaseEnvCfg, FetchIKBaseEnvCfg
from dsynth_isaac.tasks.manager_based.panda_base_env_cfg import PandaBaseEnvCfg, PandaDsynthIKRelMimicEnvCfg
import dsynth_isaac.tasks.manager_based

GRIPPER_OPEN_CMD: float = 1.0
GRIPPER_CLOSE_CMD: float = -1.0

def _eef_name(env: ManagerBasedEnv) -> str:
    return list(env.cfg.subtask_configs.keys())[0]


def _action_from_pose(
    env: ManagerBasedEnv, target_pose: torch.Tensor, gripper_binary_action: float, env_id: int = 0
) -> torch.Tensor:
    eef = _eef_name(env)
    play_action = env.target_eef_pose_to_action(
        target_eef_pose_dict={eef: target_pose},
        gripper_action_dict={eef: torch.tensor([gripper_binary_action], device=env.device, dtype=torch.float32)},
        env_id=env_id,
    )
    if play_action.dim() == 1:
        play_action = play_action.unsqueeze(0)
    return play_action


def _env_step_with_action(env: ManagerBasedEnv, action: torch.Tensor) -> None:
    env.step(action)

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
        eef_pos = env.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = env.obs_buf["policy"]["eef_quat"][env_ids]
        # Quaternion format is w,x,y,z
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

class DsynthCuroboPlannerCfg(CuroboPlannerCfg):
    @classmethod
    def franka_dsynth_config(cls) -> "CuroboPlannerCfg":
        """Create configuration for Franka stacking a normal cube."""
        config = cls.franka_config()
        config.static_objects = ["table"]
        config.visualize_plan = False
        config.debug_planner = False
        config.motion_noise_scale = 0.02
        config.collision_activation_distance = 0.01
        config.approach_distance = 0.05
        config.retreat_distance = 0.05
        config.surface_sphere_radius = 0.01
        config.get_world_config = lambda: config._get_world_config_with_table_adjustment()
        return config

def _pose_from_xy_quat(xy: torch.Tensor, z: float, quat: torch.Tensor) -> torch.Tensor:
    """Build a 4×4 pose given xy (Tensor[2]), z, and quaternion."""
    device = xy.device
    dtype = xy.dtype
    pos = torch.cat([xy, torch.tensor([z], dtype=dtype, device=device)])
    rot = math_utils.matrix_from_quat(quat.to(device).unsqueeze(0))[0]
    return math_utils.make_pose(pos, rot)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    headless = False

    env_cfg = PandaDsynthIKRelMimicEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.dsynth_scene.spawn.config_dir_path = "generated_envs/isaac_sample"

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

    robot: Articulation = env.scene["robot"]
    planner_cfg = DsynthCuroboPlannerCfg.franka_dsynth_config()
    planner_cfg.visualize_plan = False
    planner_cfg.visualize_spheres = False
    planner_cfg.debug_planner = True
    planner_cfg.retreat_distance = 0.05
    planner_cfg.approach_distance = 0.05
    planner_cfg.time_dilation_factor = 1.0

    planner = CuroboPlanner(
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
    
    # stage = omni.usd.get_context().get_stage()

    # prim = stage.GetPrimAtPath("/World/envs/env_0/dsynth_scene/active_shelf_isaac_sample_ef1d9efe_0_mixed_shelf_0/food_dairy_products_milkCarton_0_1_0_0")
    # if prim.IsValid():
    #     xformable = UsdGeom.Xformable(prim)
    #     world_transform: Gf.Matrix4d = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    #     translation = world_transform.ExtractTranslation()   # Gf.Vec3d
    #     rotation_quat = world_transform.ExtractRotationQuat()  # Gf.Quatd  (real, i, j, k)
    #     print(f"Translation: {translation}")
    #     print(f"Rotation: {rotation_quat}")
    manager = env.scene.cfg.dsynth_scene.spawn.manager
    entities = manager.get_item_entities()['env_0']

    curr_pose = get_robot_eef_pose(env, _eef_name(env))[0]

    target_shift = torch.eye(4, device=curr_pose.device)
    target_shift[0, 3] = 0.01

    target_pose = curr_pose @ target_shift
    print(f"Pre-grasp pose: {target_pose}")
    if not headless:
        pos_pg = target_pose[:3, 3].detach().cpu()
        quat_pg = math_utils.quat_from_matrix(target_pose[:3, :3].unsqueeze(0))[0].detach().cpu()
        # self._visualize_goal_pose(pos_pg, quat_pg)
        goal_pose_visualizer.visualize(translations=pos_pg.unsqueeze(0), orientations=quat_pg.unsqueeze(0))

    assert planner.update_world_and_plan_motion(target_pose), "Failed to plan to pre-grasp pose"
    # EXECUTE
    planned_poses = planner.get_planned_poses()
    if not planned_poses:
        return
    for pose in planned_poses:
        action = _action_from_pose(env, pose, gripper_binary_action=GRIPPER_OPEN_CMD)
        _env_step_with_action(env, action)
    # target_product_pose = 
    
    

if __name__ == "__main__":
    main()