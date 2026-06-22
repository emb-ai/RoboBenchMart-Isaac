from dataclasses import MISSING

import os
from pathlib import Path
from typing import Optional
from functools import partial

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab_tasks.manager_based.manipulation.stack.mdp import observations as stack_obs_mdp
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg

import isaaclab.envs.mdp as mdp

import dsynth_isaac.tasks.manager_based.mdp.ds_events as ds_events
import dsynth_isaac.tasks.manager_based.mdp.ds_observations as ds_obs
from dsynth_isaac.scene.env_subset_rigid_object_cfg import EnvSubsetRigidObjectCfg
from dsynth_isaac.scene.darkstore_scene_cfg import DarkstoreInteractiveSceneCfg
@configclass
class DarkstoreSceneCfg(DarkstoreInteractiveSceneCfg):
    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.35, 0.35, 0.35))
    )

@configclass
class DsEventCfg:
    """Configuration for events."""
    init_robot_base_pose = EventTerm(
        func=ds_events.set_default_base_pose,
        mode="reset",
        params={
            # "p": [-10, -10, 0],
            # "q": [1, 0, 0, 0],
        },
    )
    # reset_robot_base_pose = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {},
    #         "velocity_range": {},
    #     },
    # )

    reset_items = EventTerm(
        func=ds_events.reset_items_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
        },
    )

    wall_texture_randomizer_reset = EventTerm(
        func=ds_events.randomize_visual_texture_material,
        mode="reset",
        params={
            "prim_paths": [
                '/World/envs/env_.*/dsynth_scene/wall_0',
                '/World/envs/env_.*/dsynth_scene/wall_1',
                '/World/envs/env_.*/dsynth_scene/wall_2',
                '/World/envs/env_.*/dsynth_scene/wall_3'
            ],
            "texture_paths": [
                "assets_usd/textures/walls/apricot_texture.PNG",
                "assets_usd/textures/walls/Bricks058_1K-JPG_Color.jpg",
                "assets_usd/textures/walls/Bricks067_1K-JPG_Color.jpg",
                "assets_usd/textures/walls/brown_texture.PNG",
                "assets_usd/textures/walls/front-view-horizontal-white-copy-space-wall.jpg",
                "assets_usd/textures/walls/green_texture.PNG",
                "assets_usd/textures/walls/grey_texture.PNG",
                "assets_usd/textures/walls/option1.png",
                "assets_usd/textures/walls/red_bricks.png",
                "assets_usd/textures/walls/red_texture.PNG",
                "assets_usd/textures/walls/surface-yellow-solid-rough-stained.jpg",
                "assets_usd/textures/walls/white_bricks_2.png",
                "assets_usd/textures/walls/white_bricks_old.png",
                "assets_usd/textures/walls/white_bricks.png",
                "assets_usd/textures/walls/white-wall-textures.jpg",
                "assets_usd/textures/walls/yellow-plaster.png",
                "assets_usd/textures/walls/yellow_texture.png",
            ],
            "event_name": "wall_texture_randomizer_reset",
        },
    )

    ceiling_texture_randomizer_reset = EventTerm(
        func=ds_events.randomize_visual_texture_material,
        mode="reset",
        params={
            "prim_paths": '/World/envs/env_.*/dsynth_scene/ceiling',

            "texture_paths": [
                "assets_usd/textures/ceilings/147.jpg",
                "assets_usd/textures/ceilings/1626709619_7-kartinkin-com-p-armstrong-tekstura-besshovnaya-krasivo-7.png",
                "assets_usd/textures/ceilings/1672880537_grizly-club-p-tekstura-armstrong-28.jpg",
                "assets_usd/textures/ceilings/background-wall-concept-with-copy-space.jpg",
                "assets_usd/textures/ceilings/blacktile.jpeg",
                "assets_usd/textures/ceilings/ceilume-southland-2x2-latte-ceiling-tiles_face_540.jpg",
                "assets_usd/textures/ceilings/ec9fef5dfb43670b7da13a18b4b4a23c.jpg",
                "assets_usd/textures/ceilings/most-popular-suspended-nd-fissured-surf-ceiling-tiles-595mm-x-595mm-full-box.-629-p.jpg",
                "assets_usd/textures/ceilings/MPA_SkyBlue-e1720596032435.jpg",
                "assets_usd/textures/ceilings/OC1Y9B0.jpg",
                "assets_usd/textures/ceilings/old-squared-white-texture.jpg",
                "assets_usd/textures/ceilings/SQUARE-BK-1.png",
                "assets_usd/textures/ceilings/texturecan-plastic-0003-plane-1200.jpg",
                "assets_usd/textures/ceilings/textur-gas-kvas-com-8qct-p-teksturi-potolochnaya-plitka-16.jpg",
                "assets_usd/textures/ceilings/Tiles071_1K-PNG_Color.png"
            ],
            "event_name": "ceiling_texture_randomizer_reset",
        },
    )

    floor_texture_randomizer_reset = EventTerm(
        func=ds_events.randomize_visual_texture_material,
        mode="reset",
        params={
            "prim_paths": '/World/envs/env_.*/dsynth_scene/floor',

            "texture_paths": [
                "assets_usd/textures/floors/05f75518-7b70-4098-8983-aca459f33ddb.png",
                "assets_usd/textures/floors/144696-13369113.jpg",
                "assets_usd/textures/floors/144696-13370247.png",
                "assets_usd/textures/floors/41c3dbec-db54-4417-bb76-f51a4349943e.png",
                "assets_usd/textures/floors/bamboo.png",
                "assets_usd/textures/floors/black-stucco-wall.jpg",
                "assets_usd/textures/floors/checkers.png",
                "assets_usd/textures/floors/Concrete007_1K-JPG_Color.jpg",
                "assets_usd/textures/floors/concrete.png",
                "assets_usd/textures/floors/dark_wood_planks.png",
                "assets_usd/textures/floors/diamond_tiles_bak.png",
                "assets_usd/textures/floors/diamond_tiles.png",
                "assets_usd/textures/floors/gray-cement-wall.jpg",
                "assets_usd/textures/floors/gray_wood_planks.png",
                "assets_usd/textures/floors/grey-felt-texture.jpg",
                "assets_usd/textures/floors/grunge-wall-texture.jpg",
                "assets_usd/textures/floors/Hc518c5e6640145c5934d4b92f7e1c840Q.png",
                "assets_usd/textures/floors/light_wood_parquet.png",
                "assets_usd/textures/floors/light_wood_planks_2.png",
                "assets_usd/textures/floors/light_wood_planks_long.png",
                "assets_usd/textures/floors/old-grunge-concrete.jpg",
                "assets_usd/textures/floors/orange_square_tiles.png",
                "assets_usd/textures/floors/sq4.jpg",
                "assets_usd/textures/floors/wood_planks_2.png",
                "assets_usd/textures/floors/wood_planks_3.png",
                "assets_usd/textures/floors/wood_planks.png"
            ],
            "event_name": "floor_texture_randomizer_reset",
        },
    )

    lights_randomizer_reset = EventTerm(
        func=ds_events.randomize_lights,
        mode="reset",
        params={
        },
    )


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
        eef_pos = ObsTerm(func=stack_obs_mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=stack_obs_mdp.ee_frame_quat)
        eef_pos_base = ObsTerm(func=partial(stack_obs_mdp.ee_frame_pose_in_base_frame, return_key="pos"))
        eef_quat_base = ObsTerm(func=partial(stack_obs_mdp.ee_frame_pose_in_base_frame, return_key="quat"))
        base_pos = ObsTerm(func=ds_obs.base_pos)
        base_quat = ObsTerm(func=ds_obs.base_quat)
        gripper_pos = ObsTerm(func=stack_obs_mdp.gripper_pos)

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
    # scene: DarkstoreSceneCfg = gen_dsynth_scene_cfg_class(config_dir_path, num_envs)(num_envs=num_envs, env_spacing=20.5, replicate_physics=False)
    scene: DarkstoreSceneCfg = DarkstoreSceneCfg(env_spacing=20.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    # Set events
    events: DsEventCfg = MISSING

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