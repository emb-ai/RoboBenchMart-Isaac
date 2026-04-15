from dataclasses import MISSING

import os
from pathlib import Path
from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCollectionCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg

import isaaclab.envs.mdp as mdp

from dsynth_isaac.scene.spawners import DsynthMultiSceneCfg
from dsynth_isaac.scene.scene_loader import load_scene
@configclass
class DarkstoreSceneCfg(InteractiveSceneCfg):
    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    # ee_frame: FrameTransformerCfg = MISSING

    #==== RoboBenchMart scene ====
    
    # ground plane 
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )


    dsynth_multi_scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dsynth_scene",
        spawn=DsynthMultiSceneCfg(
            config_dir_path=MISSING
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    cube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/my_cube",
        spawn=sim_utils.CuboidCfg(size=(3, 1, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path="/home/kvsoshin/Work/RoboBenchMart-Isaac-Private/assets_usd/mdl/bricks3/bricks3.mdl",
                project_uvw=True,
                texture_scale=(1., 1.)
            ),
            # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 4)),
    )
    # cube2 = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/my_cube_2",
    #     spawn=sim_utils.CuboidCfg(size=(4, 5, 2),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.MdlFileCfg(
    #             mdl_path="/home/kvsoshin/Work/RoboBenchMart-Isaac-Private/assets_usd/mdl/bricks3/bricks3.mdl",
    #             project_uvw=True,
    #             texture_scale=(1., 1.)
    #         ),
    #         # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(-4.0, -4.0, 4)),
    # )

    # object: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Object",
    #     spawn=sim_utils.MultiAssetSpawnerCfg(
    #         assets_cfg=[
    #             sim_utils.ConeCfg(
    #                 radius=0.3,
    #                 height=0.6,
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
    #             ),
    #             sim_utils.CuboidCfg(
    #                 size=(0.3, 0.3, 0.3),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #             ),
    #             sim_utils.SphereCfg(
    #                 radius=0.3,
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
    #             ),
    #         ],
    #         random_choice=True,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=4, solver_velocity_iteration_count=0
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    # )

    # asset = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/asset",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path='assets_usd/preprocessed/monster_energy_drink_rotated/monster_energy_drink_rotated.usd',
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg()
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 3.0)),
    # )

    # object_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
    #     rigid_objects={
    #         "object_A": RigidObjectCfg(
    #             prim_path="/World/envs/env_.*/Object_A",
    #             spawn=sim_utils.SphereCfg(
    #                 radius=0.1,
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     solver_position_iteration_count=4, solver_velocity_iteration_count=0
    #                 ),
    #                 mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                 collision_props=sim_utils.CollisionPropertiesCfg(),
    #             ),
    #             init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.5, 2.0)),
    #         ),
    #         "object_B": RigidObjectCfg(
    #             prim_path="/World/envs/env_.*/Object_B",
    #             spawn=sim_utils.CuboidCfg(
    #                 size=(0.1, 0.1, 0.1),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     solver_position_iteration_count=4, solver_velocity_iteration_count=0
    #                 ),
    #                 mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                 collision_props=sim_utils.CollisionPropertiesCfg(),
    #             ),
    #             init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.5, 2.0)),
    #         ),
    #         "object_C": RigidObjectCfg(
    #             prim_path="/World/envs/env_.*/Object_C",
    #             spawn=sim_utils.ConeCfg(
    #                 radius=0.1,
    #                 height=0.3,
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     solver_position_iteration_count=4, solver_velocity_iteration_count=0
    #                 ),
    #                 mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                 collision_props=sim_utils.CollisionPropertiesCfg(),
    #             ),
    #             init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 2.0)),
    #         ),
    #     }
    # )



    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

def gen_dsynth_scene_cfg_class(config_dir_path: str, num_envs: int):
    @configclass
    class DarkstoreSceneCfg2(InteractiveSceneCfg):
        # robots: will be populated by agent env cfg
        robot: ArticulationCfg = MISSING
        # end-effector sensor: will be populated by agent env cfg
        # ee_frame: FrameTransformerCfg = MISSING

        #==== RoboBenchMart scene ====
        
        # ground plane 
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
        )


        dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )

        def __post_init__(self):
            load_scene(self, config_dir_path, num_envs)

    return DarkstoreSceneCfg2
@configclass
class EventCfg:
    # TODO: do scene rearrangement?
    """Configuration for events."""
    pass


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        # eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        # gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

def gen_dsynth_env_cfg_class(config_dir_path: str, num_envs: int):
    @configclass
    class DarkstoreBaseEnvCfg(ManagerBasedRLEnvCfg):
        """Configuration for the stacking environment."""
        # Scene settings
        # scene: DarkstoreSceneCfg = gen_dsynth_scene_cfg_class(config_dir_path, num_envs)(num_envs=num_envs, env_spacing=20.5, replicate_physics=False)
        scene: DarkstoreSceneCfg = DarkstoreSceneCfg(num_envs=num_envs, env_spacing=20.5, replicate_physics=False)
        # Basic settings
        observations: ObservationsCfg = ObservationsCfg()
        actions: ActionsCfg = ActionsCfg()
        # MDP settings
        terminations: TerminationsCfg = TerminationsCfg()
        # Set events
        events: EventCfg = MISSING

        # Unused managers
        commands = None
        rewards = None
        events = None
        curriculum = None
        
        def __post_init__(self):
            """Post initialization."""
            # general settings
            self.decimation = 2
            self.episode_length_s = 300.0
            # simulation settings
            self.sim.dt = 0.01  # 100Hz
            self.sim.render_interval = 2

    return DarkstoreBaseEnvCfg