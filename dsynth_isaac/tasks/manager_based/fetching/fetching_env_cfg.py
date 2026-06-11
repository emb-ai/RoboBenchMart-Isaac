from typing import Any, Sequence
from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg

from dsynth_isaac.tasks.manager_based.darkstore_manager_based_rl_env import DarkstoreManagerBasedRLEnv
from dsynth_isaac.tasks.manager_based.darkstore_base_cfg import DarkstoreBaseEnvCfg
from dsynth_isaac.tasks.manager_based.fetching.mdp.terminations import target_object_grasped

@configclass
class FetchAnyItemEnvTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(func=target_object_grasped,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
    )

@configclass
class FetchAnyItemEnvCfg(DarkstoreBaseEnvCfg):
    """Configuration for the fetching environment."""
    target_item_name_regex: str = MISSING

    def __post_init__(self):
        super().__post_init__()
        self.terminations = FetchAnyItemEnvTerminationsCfg()

    
    