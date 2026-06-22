from typing import Any, Sequence
from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg

from dsynth_isaac.tasks.manager_based.darkstore_manager_based_rl_env import DarkstoreManagerBasedRLEnv
from dsynth_isaac.tasks.manager_based.darkstore_base_cfg import DarkstoreBaseEnvCfg
import dsynth_isaac.tasks.manager_based.fetching.mdp.terminations as mdp_term

@configclass
class FetchItemEnvCfg(DarkstoreBaseEnvCfg):
    """Configuration for the fetching environment."""

    # regex for the human-readable name of the target item, e.g. "milk" or ".*Nivea*" etc.
    target_item_name_regex: str = MISSING
    language_instruction: str = MISSING


@configclass
class GraspAnyItemEnvTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(func=mdp_term.target_object_grasped,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "check_if_pulled": False,
        },
    )

@configclass
class GraspAnyItemEnvCfg(FetchItemEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.terminations = GraspAnyItemEnvTerminationsCfg()
        self.language_instruction = f"Grasp any {self.target_item_name_regex} item"


@configclass
class FetchAnyItemEnvTerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(func=mdp_term.target_object_grasped,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "check_if_pulled": True,
        },
    )

@configclass
class FetchAnyItemEnvCfg(FetchItemEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.terminations = FetchAnyItemEnvTerminationsCfg()
        self.language_instruction = f"Fetch any {self.target_item_name_regex}"