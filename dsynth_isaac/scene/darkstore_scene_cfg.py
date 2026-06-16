from dataclasses import MISSING
from typing import Optional

from isaaclab.utils.configclass import configclass
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg


import isaaclab.envs.mdp as mdp

import dsynth_isaac.tasks.manager_based.mdp.ds_events as ds_events
import dsynth_isaac.tasks.manager_based.mdp.ds_observations as ds_obs
from dsynth_isaac.scene.env_subset_rigid_object_cfg import EnvSubsetRigidObjectCfg

@configclass
class DarkstoreInteractiveSceneCfg(InteractiveSceneCfg):
    scene_config_dir_path: str = MISSING
    max_scenes: Optional[int] = None
    seed: int = 42

    # # robots: will be populated by agent env cfg
    # robot: ArticulationCfg = MISSING
    # # end-effector sensor: will be populated by agent env cfg
    # ee_frame: FrameTransformerCfg = MISSING

    # object: EnvSubsetRigidObjectCfg = EnvSubsetRigidObjectCfg(
    #     prim_path='/World/envs/env_[1,3]/object',
    #     env_ids=[1, 3],
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path='/home/soshin/repo/RoboBenchMart-Isaac-Private/assets_usd/preprocessed/cereal_-_honey_star/cereal_-_honey_star.usd',
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
    #         # collision_props=sim_utils.CollisionPropertiesCfg(),
    #     ),
    #     init_state=EnvSubsetRigidObjectCfg.InitialStateCfg(pos=[0, 0, 3]),
    # )

    # dome_light = AssetBaseCfg(
    #     prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.35, 0.35, 0.35))
    # )