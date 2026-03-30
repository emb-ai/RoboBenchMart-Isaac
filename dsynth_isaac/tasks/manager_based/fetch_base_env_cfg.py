from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils import configclass
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg

from isaaclab.sensors import CameraCfg
import isaaclab.envs.mdp as mdp


from dsynth_isaac.tasks.manager_based.darkstore_base_cfg import DarkstoreBaseEnvCfg
from dsynth_isaac.robots.fetch_cfg import FETCH_CONFIG
import dsynth_isaac.tasks.manager_based.mdp.fetch_events as fetch_events
from dsynth_isaac.devices.mobile_se3_keyboard import MobileSe3KeyboardCfg

ARM_JOINT_NAMES = ["shoulder_pan_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "forearm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint"
]
GRIPPER_JOINT_NAMES = ["r_gripper_finger_joint", "l_gripper_finger_joint"]
BODY_JOINT_NAMES = ["torso_lift_joint"]
HEAD_JOINT_NAMES = ["head_pan_joint", "head_tilt_joint"]
BASE_JOINT_NAMES = ["l_wheel_joint", "r_wheel_joint"]

@configclass
class FetchActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_actions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=1.0)
    gripper_actions = mdp.BinaryJointPositionActionCfg(asset_name="robot", joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={"r_gripper_finger_joint": 0.04, "l_gripper_finger_joint": 0.04},
        close_command_expr={"r_gripper_finger_joint": 0.01, "l_gripper_finger_joint": 0.01},
        )
    body_actions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=BODY_JOINT_NAMES, scale=1.0)
    head_actions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=HEAD_JOINT_NAMES, scale=1.0)
    base_actions = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=BASE_JOINT_NAMES, scale=1.0)

@configclass
class FetchEventCfg:
    # TODO: do scene rearrangement?
    """Configuration for events."""
    init_fetch_pose = EventTerm(
        func=fetch_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
        },
    )
    reset_fetch_base_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
        },
    )

    init_fetch_base_pose = EventTerm(
        func=fetch_events.set_default_base_pose,
        mode="startup",
        params={
        },
    )


@configclass
class FetchBaseEnvCfg(DarkstoreBaseEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = FETCH_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot.init_state.pos = (0, 0, 0)

        self.scene.head_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/fetch/head_tilt_link/head_camera_link/head_camera_rgb_frame/head_camera_rgb_optical_frame/HeadCameraRGBFrame",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb",],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), convention="ros"),
        )

        self.observations.rgb_camera.head_img = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("head_camera")},
            # noise=Unoise(n_min=-0.1, n_max=0.1),
            # clip=(-1.0, 1.0),
        )

        self.actions = FetchActionsCfg()
        self.events = FetchEventCfg()


@configclass
class FetchIKBaseEnvCfg(FetchBaseEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.actions.arm_actions = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=ARM_JOINT_NAMES,
            body_name="gripper_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": MobileSe3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )