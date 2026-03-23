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

import isaaclab.envs.mdp as mdp


@configclass
class DarkstoreSceneCfg(InteractiveSceneCfg):
    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    # ee_frame: FrameTransformerCfg = MISSING

    #==== RoboBenchMart scene ====
    # TODO: redo (how to import dsynth scene?)
    # config_dir_path: str = None
    
    # ground plane 
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # sphere_light = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/SphereLight",
    #     spawn=sim_utils.SphereLightCfg(color=(0.9, 0.9, 0.9), intensity=60000.0),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 2.5))
    # )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    object_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "object_a": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/object_a",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"assets_usd/preprocessed/monster_energy_drink_rotated/monster_energy_drink_rotated.usd",
                    # usd_path="shelf_metal_fake_1/shelf_metal_fake_1.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            solver_position_iteration_count=4, solver_velocity_iteration_count=0,
                            rigid_body_enabled=True,
                            kinematic_enabled=False
                        ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 2.0))
            ),
            "object_b": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/object_b",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"assets_usd/preprocessed/fantaNaranja1p35l/fantaNaranja1p35l.usd",
                    # usd_path="shelf_metal_fake_1/shelf_metal_fake_1.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            solver_position_iteration_count=4, solver_velocity_iteration_count=0,
                            rigid_body_enabled=True,
                            kinematic_enabled=False
                        ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 1.0, 2.0))
            )
        }
    )
    # custom_object = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/CustomObject",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path="assets/preprocessed/monster_energy_drink_rotated.glb", # Convert GLB to USD first
    #         # 1. Enable Rigid Body properties
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             rigid_body_enabled=True,
    #             kinematic_enabled=False,
    #             solver_position_iteration_count=4,
    #             solver_velocity_iteration_count=1,
    #         ),
    #         # 2. Enable Collision properties
    #         collision_props=sim_utils.CollisionPropertiesCfg(
    #             collision_enabled=True,
    #         ),
    #         # 3. Define the Collider Approximation (Essential for meshes)
    #         # 4. Optional: Set Mass
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    # )


    object_b = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_b",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.5, 2.0)),
    )
    # def __post_init__(self):
    #     if self.config_dir_path is not None:
    #         self.scene_config_paths = sorted(list(Path(self.config_dir_path).glob('*layout_data*.json')))
    #     else:
    #         self.scene_config_paths = []
    #     print(">>>>", self.scene_config_paths)

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

@configclass
class DarkstoreBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""
    # Scene settings
    scene: DarkstoreSceneCfg = DarkstoreSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
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
