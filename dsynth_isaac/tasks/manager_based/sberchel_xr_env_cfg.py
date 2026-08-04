"""Free-floating open-kinematics Sberchel: zest legs + XR Pink IK upper body."""

import torch
import isaaclab.envs.mdp as mdp
from isaaclab.devices.openxr import XrCfg
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg

from dsynth_isaac.robots.sberchel_cfg import SBERCHEL_XR_LOCOMOTION_ROBOT
from dsynth_isaac.tasks.manager_based.darkstore_base_cfg import DarkstoreBaseEnvCfg
from dsynth_isaac.tasks.manager_based.sberchel_base_env_cfg import (
    POLICY_LEG_JOINT_NAMES,
    SberchelEventCfg,
    SberchelObservationsCfg,
)
from dsynth_isaac.tasks.manager_based.sberchel_upperbody_xr_env_cfg import (
    UpperbodySberchelXRActionsCfg,
    setup_sberchel_xr_pink_and_teleop,
)
from dsynth_isaac.scene.sberdemo_scene import SberDemoSceneCfg
from dsynth_isaac.tasks.manager_based.sberchel_base_env_cfg import SBERCHEL_STAND_HEIGHT
from dsynth_isaac.tasks.manager_based.darkstore_base_cfg import TerminationsCfg


@configclass
class SberchelXRActionsCfg:
    """Action layout: ``[legs(12), upper_body_ik(26)]`` (legs first)."""

    legs_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=POLICY_LEG_JOINT_NAMES,
        scale=1.0,
        preserve_order=True,
    )
    upper_body_ik = UpperbodySberchelXRActionsCfg().upper_body_ik


@configclass
class SberchelXRFetchAnyItemEnvCfg(DarkstoreBaseEnvCfg):
    """Open Sberchel: free pelvis, policy legs, XR upper-body teleop."""

    actions = SberchelXRActionsCfg()
    events = SberchelEventCfg()
    observations: SberchelObservationsCfg = SberchelObservationsCfg()

    target_item_name_regex: str = ".*"
    language_instruction: str = "XR teleop Sberchel with locomotion policy"

    xr: XrCfg = XrCfg(
        anchor_pos=(4.0, 5.0, 0.0),
        anchor_rot=(0.0, 0.0, 1.0, 0.0),
    )

    NUM_OPENXR_HAND_JOINTS = 26

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SBERCHEL_XR_LOCOMOTION_ROBOT.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot_fingers_contact_prim_path = None
        self.scene.ee_frame = None

        self.sim.dt = 1.0 / 150.0
        self.decimation = 1
        self.sim.render_interval = 1

        setup_sberchel_xr_pink_and_teleop(self)

@configclass
class SberchelXRDemoEnvCfg(ManagerBasedRLEnvCfg):
    scene: SberDemoSceneCfg = SberDemoSceneCfg(replicate_physics=False, env_spacing=20.5, filter_collisions=False)
    actions = SberchelXRActionsCfg()
    events: None
    observations: SberchelObservationsCfg = SberchelObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    xr: XrCfg = XrCfg(
        anchor_pos=(5.5, 5.0, 0.0),
        anchor_rot=(0.0, 0.0, 1.0, 0.0),
    )

    NUM_OPENXR_HAND_JOINTS = 26

    # Upper-body Pink idle only (26D). Replay pads legs from the robot after reset.
    idle_action = torch.tensor(
        [
            5.40,
            5.95,
            1.15,
            0.707,
            0.0,
            0.0,
            0.707,
            5.80,
            5.95,
            1.15,
            0.707,
            0.0,
            0.0,
            0.707,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SBERCHEL_XR_LOCOMOTION_ROBOT.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot.init_state.pos = (5.5, 5, SBERCHEL_STAND_HEIGHT)

        self.sim.dt = 1.0 / 150.0
        self.decimation = 1
        self.sim.render_interval = 1
        self.episode_length_s = 100000

        setup_sberchel_xr_pink_and_teleop(self)