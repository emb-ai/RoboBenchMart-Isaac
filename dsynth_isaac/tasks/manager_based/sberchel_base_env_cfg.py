"""Darkstore environment with the Sberchel v3.0a humanoid as the robot.

Same store/scene as the Fetch task (``DarkstoreBaseEnvCfg``); only the robot in
``self.scene.robot`` is swapped out.

Sberchel is a free-floating biped wired for the green_challenge locomotion policy
(see ``scripts/demo_sberchel.py``). Upper-body XR teleop lives in
``sberchel_upperbody_xr_env_cfg.UpperbodySberchelXRFetchAnyItemEnvCfg``.
Full-body XR + policy legs: ``sberchel_xr_env_cfg.SberchelXRFetchAnyItemEnvCfg``.
"""

import isaaclab.envs.mdp as mdp
import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from dsynth_isaac.tasks.manager_based.darkstore_base_cfg import DarkstoreBaseEnvCfg, DsEventCfg
from dsynth_isaac.robots.sberchel_cfg import SBERCHEL_CONFIG
import dsynth_isaac.tasks.manager_based.mdp.ds_events as ds_events

# Root (pelvis) height the robot is placed at so its feet start on the floor.
SBERCHEL_STAND_HEIGHT = 0.9


@configclass
class SberchelActionsCfg:
    """Position actions, mirroring green_challenge's ``ExampleActionsPosCfg``.

    Order of the concatenated action vector (41 dims):
      legs(12) + torso(13) + wrist(4) + fingers(12).
    """

    legs_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_crank_joint",
            "left_ankle_u_crank_joint",
            "left_ankle_b_crank_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_crank_joint",
            "right_ankle_u_crank_joint",
            "right_ankle_b_crank_joint",
        ],
        scale=1.0,
        preserve_order=True,
    )

    torso_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "torso_yaw_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_elbow_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_elbow_yaw_joint",
            "neck_yaw_joint",
            "neck_pitch_joint",
        ],
        scale=1.0,
        preserve_order=True,
    )

    wrist_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_wrist_crank_joint",
            "left_wrist_roll_joint",
            "right_wrist_crank_joint",
            "right_wrist_roll_joint",
        ],
        scale=1.0,
        use_default_offset=True,
        preserve_order=True,
    )

    finger_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_pinky_proximal_joint",
            "left_ring_proximal_joint",
            "left_middle_proximal_joint",
            "left_index_proximal_joint",
            "left_thumb_proximal_pitch_joint",
            "left_thumb_proximal_yaw_joint",
            "right_pinky_proximal_joint",
            "right_ring_proximal_joint",
            "right_middle_proximal_joint",
            "right_index_proximal_joint",
            "right_thumb_proximal_pitch_joint",
            "right_thumb_proximal_yaw_joint",
        ],
        scale=1.0,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class SberchelEventCfg(DsEventCfg):
    """Reuse the store's events, but lift the base to standing height."""

    init_robot_base_pose = EventTerm(
        func=ds_events.set_default_base_pose,
        mode="startup",
        params={"p": [0.65, 0.0, SBERCHEL_STAND_HEIGHT]},
    )


@configclass
class SberchelObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class SberchelBaseEnvCfg(DarkstoreBaseEnvCfg):
    observations: SberchelObservationsCfg = SberchelObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SBERCHEL_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot_fingers_contact_prim_path = None
        self.scene.ee_frame = None

        self.sim.dt = 1.0 / 150.0
        self.decimation = 1
        self.sim.render_interval = 1

        self.actions = SberchelActionsCfg()
        self.events = SberchelEventCfg()
