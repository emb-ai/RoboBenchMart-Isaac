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

    # robot fingers contact sensor prim path. Set None to disable.
    robot_fingers_contact_prim_path: str = MISSING