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
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab.envs import ManagerBasedRLEnvCfg

from dsynth_isaac.tasks.manager_based.darkstore_base_cfg import DarkstoreBaseEnvCfg, DsEventCfg
from dsynth_isaac.robots.sberchel_cfg import SBERCHEL_CONFIG, SBERCHEL_CONFIG_OPEN_KINEMATICS
import dsynth_isaac.tasks.manager_based.mdp.ds_events as ds_events
from dsynth_isaac.scene.sberdemo_scene import SberDemoSceneCfg
from dsynth_isaac.tasks.manager_based.darkstore_base_cfg import TerminationsCfg

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
        mode="reset",
        params={"p": [0.65, 0.0, SBERCHEL_STAND_HEIGHT]},
    )

    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
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

# Policy-native leg joint order (matches green_challenge ``model.joint_order`` legs slice).
POLICY_LEG_JOINT_NAMES = [
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
]


@configclass
class SberchelOpenKinematicsActionsCfg:
    """41D actions for hybrid open MJCF: policy crank legs + serial upper body."""

    legs_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=POLICY_LEG_JOINT_NAMES,
        scale=1.0,
        preserve_order=True,
    )

    torso_joint_pos = SberchelActionsCfg().torso_joint_pos
    wrist_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_wrist_pitch_joint",
            "left_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_roll_joint",
        ],
        scale=1.0,
        use_default_offset=True,
        preserve_order=True,
    )
    finger_joint_pos = SberchelActionsCfg().finger_joint_pos


LEGS_ACTION_DIM = len(POLICY_LEG_JOINT_NAMES)


def policy_leg_targets(net, vx: float, vy: float, wz: float) -> list[float]:
    """Run zest policy; return 12 leg joint targets in ``POLICY_LEG_JOINT_NAMES`` order."""
    net.set_command_detailed(vx, vy, wz)
    net.set_joint_overrides({
        "torso_joint_pos": [0.0] * 13,
        "wrist_joint_pos": [0.0] * 4,
        "finger_joint_pos": [0.0] * 12,
    })
    loco = net(net.obs())
    return list(loco["joint_pos"][:LEGS_ACTION_DIM])

@configclass
class SberchelOpenKinematicsFetchAnyItemEnvCfg(DarkstoreBaseEnvCfg):
    actions = SberchelOpenKinematicsActionsCfg()
    events = SberchelEventCfg()
    observations: SberchelObservationsCfg = SberchelObservationsCfg()

    target_item_name_regex: str = ".*"
    language_instruction: str = None

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SBERCHEL_CONFIG_OPEN_KINEMATICS.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot_fingers_contact_prim_path = None
        self.scene.ee_frame = None

        self.sim.dt = 1.0 / 150.0
        self.decimation = 1
        self.sim.render_interval = 1

@configclass
class SberchelOpenKinematicsDemoEnvCfg(ManagerBasedRLEnvCfg):
    scene: SberDemoSceneCfg = SberDemoSceneCfg(replicate_physics=False, env_spacing=20.5, filter_collisions=False)
    actions = SberchelOpenKinematicsActionsCfg()
    observations: SberchelObservationsCfg = SberchelObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    events: None

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SBERCHEL_CONFIG_OPEN_KINEMATICS.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot.init_state.pos = (5.5, 5, SBERCHEL_STAND_HEIGHT)
        # Pin pelvis_link (articulation root) in world space.
        # if self.scene.robot.spawn.articulation_props is not None:
        #     self.scene.robot.spawn.articulation_props = (
        #         self.scene.robot.spawn.articulation_props.replace(fix_root_link=True)
        #     )
        # general settings
        self.decimation = 2
        self.episode_length_s = 100000.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2