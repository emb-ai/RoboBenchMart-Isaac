"""Free-floating closed-kinematics Sberchel (v3_0a): zest legs + XR Pink IK upper body."""

import torch
import isaaclab.envs.mdp as mdp
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg

import dsynth_isaac.tasks.manager_based.mdp.ds_events as ds_events

from dsynth_isaac.robots.sberchel_cfg import SBERCHEL_XR_LOCOMOTION_NECK_ROBOT
from dsynth_isaac.robots.sberchel_camera_cfg import add_sberchel_cameras
from dsynth_isaac.tasks.manager_based.darkstore_base_cfg import DarkstoreBaseEnvCfg
from dsynth_isaac.tasks.manager_based.sberchel_base_env_cfg import (
    NECK_JOINT_NAMES,
    POLICY_LEG_JOINT_NAMES,
    SberchelEventCfg,
    SberchelObservationsCfg,
    apply_green_challenge_physx,
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
    """Action layout: ``[legs(12), neck(2), upper_body_ik(26), torso_yaw(1)]``.

    ``torso_yaw_joint_pos`` is declared last on purpose: appending it keeps the
    first 40 dims identical to the layout every earlier dataset was recorded
    with, so legacy rows replay by zero-padding the tail (zero = default pose =
    exactly what an uncommanded torso_yaw did before the term existed).
    """

    legs_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=POLICY_LEG_JOINT_NAMES,
        scale=1.0,
        preserve_order=True,
    )
    neck_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=NECK_JOINT_NAMES,
        scale=1.0,
        preserve_order=True,
    )
    upper_body_ik = UpperbodySberchelXRActionsCfg().upper_body_ik
    # The walk policy commands torso_yaw as part of its gait (torso
    # counter-rotation); without this term that output had nowhere to go and the
    # joint was PD-held at its default.
    torso_yaw_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["torso_yaw_joint"],
        scale=1.0,
        preserve_order=True,
    )


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

        self.scene.robot = SBERCHEL_XR_LOCOMOTION_NECK_ROBOT.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot_fingers_contact_prim_path = None
        self.scene.ee_frame = None

        self.sim.dt = 1.0 / 150.0
        self.decimation = 1
        self.sim.render_interval = 1

        apply_green_challenge_physx(self)
        setup_sberchel_xr_pink_and_teleop(self)

@configclass
class SberDemoEventCfg:
    """Randomize the start pose on every reset so trajectories don't all begin identically."""

    # Without an explicit joint reset the articulation keeps its end-of-episode
    # joint state across env.reset() (only the root pose event below runs), so
    # every new episode would start with the previous episode's arm/hand/neck
    # configuration. Zero ranges = exactly the default joint state.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_robot_start_pose = EventTerm(
        func=ds_events.reset_root_pose_in_region_facing_target,
        mode="reset",
        params={
            # Free floor strip behind (west of) the packing table, which spans
            # x in [6.1, 6.9], y in [4.8, 7.2] in the small_store_aij layout.
            "pos_x_range": (4.7, 5.7),
            "pos_y_range": (4.6, 7.4),
            # Yaw aims at a random point on the shelf-row front (shelves sit at x ~ 9.65).
            "target_x_range": (9.4, 9.7),
            "target_y_range": (3.2, 8.8),
            "yaw_noise_range": (-0.1, 0.1),
        },
    )


@configclass
class SberchelXRDemoEnvCfg(ManagerBasedRLEnvCfg):
    scene: SberDemoSceneCfg = SberDemoSceneCfg(replicate_physics=False, env_spacing=20.5, filter_collisions=False)
    actions = SberchelXRActionsCfg()
    events: SberDemoEventCfg = SberDemoEventCfg()
    observations: SberchelObservationsCfg = SberchelObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    xr: XrCfg = XrCfg(
        anchor_pos=(5.5, 5.0, 0.0),
        # anchor_rot=(0.0, 0.0, 1.0, 0.0),
    )

    NUM_OPENXR_HAND_JOINTS = 26

    # Upper-body Pink idle only (26D). Replay pads legs/neck from the robot after reset.
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

        self.scene.robot = SBERCHEL_XR_LOCOMOTION_NECK_ROBOT.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot.init_state.pos = (5.5, 5, SBERCHEL_STAND_HEIGHT)
        # Every camera green_challenge's camera_mapping enables (left head plus
        # both wrists as of the _poly200 revision). Record/replay scripts strip
        # the ones not selected via --cameras.
        add_sberchel_cameras(self)

        self.sim.dt = 1.0 / 150.0
        self.decimation = 1
        self.sim.render_interval = 1
        self.episode_length_s = 100000

        apply_green_challenge_physx(self)
        setup_sberchel_xr_pink_and_teleop(self)
