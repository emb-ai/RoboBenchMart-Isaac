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

import omni.usd
from pxr import UsdGeom, Usd, Gf

import gymnasium as gym
import sys
sys.path.append('.')
from dsynth_isaac.tasks.manager_based.fetch_base_env_cfg import FetchBaseEnvCfg, FetchIKBaseEnvCfg
from dsynth_isaac.tasks.manager_based.panda_base_env_cfg import PandaBaseEnvCfg
import dsynth_isaac.tasks.manager_based


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

    env_cfg = PandaBaseEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.dsynth_multi_scene.spawn.config_dir_path = "generated_envs/isaac_sample"

    for frame in env_cfg.scene.ee_frame.target_frames:
        if frame.name == "end_effector":
            print(f"Setting end effector offset from {frame.offset.pos} to (0.0, 0.0, 0.0) for SkillGen parity")
            frame.offset.pos = (0.0, 0.0, 0.0)
    
    env: ManagerBasedEnv = gym.make(
        "Isaac-Dsynth-Panda-Base-v0",
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
    
    stage = omni.usd.get_context().get_stage()

    prim = stage.GetPrimAtPath("/World/envs/env_0/dsynth_scene/active_shelf_isaac_sample_ef1d9efe_0_mixed_shelf_0/food_dairy_products_milkCarton_0_1_0_0")
    if prim.IsValid():
        xformable = UsdGeom.Xformable(prim)
        world_transform: Gf.Matrix4d = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = world_transform.ExtractTranslation()   # Gf.Vec3d
        rotation_quat = world_transform.ExtractRotationQuat()  # Gf.Quatd  (real, i, j, k)
        print(f"Translation: {translation}")
        print(f"Rotation: {rotation_quat}")

    curr_pose = env.get_robot_eef_pose(eef_name=eef, env_ids=[env_id])[0]
    
    

if __name__ == "__main__":
    main()