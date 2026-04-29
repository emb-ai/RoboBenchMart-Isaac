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


from dsynth_isaac.tasks.manager_based.darkstore_base_cfg import DarkstoreBaseEnvCfg, DsEventCfg
from dsynth_isaac.robots.worker_cfg import WORKER_CONFIG
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
import dsynth_isaac.tasks.manager_based.mdp.ds_events as ds_events


LEFT_ARM_JOINT_NAMES = ["arx5_left_joint.*"]
LEFT_GRIPPER_JOINT_NAMES = ["arx5_left_gripper_joint"]
RIGHT_ARM_JOINT_NAMES = ["arx5_right_joint.*"]
RIGHT_GRIPPER_JOINT_NAMES = ["arx5_right_gripper_joint"]
BODY_JOINT_NAMES = ["torso_drive_joint"]
CAMERA_JOINT_NAMES = ["femto_holder_drive_joint"]

@configclass
class WorkerActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    left_arm_actions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=LEFT_ARM_JOINT_NAMES, scale=1.0)
    left_gripper_actions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=LEFT_GRIPPER_JOINT_NAMES, scale=1.0)
    right_arm_actions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=RIGHT_ARM_JOINT_NAMES, scale=1.0)
    right_gripper_actions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES, scale=1.0)
    body_actions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=BODY_JOINT_NAMES, scale=1.0)
    camera_actions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=CAMERA_JOINT_NAMES, scale=1.0)

@configclass
class WorkerEventCfg(DsEventCfg):
    # TODO: do scene rearrangement?
    """Configuration for events."""
    init_robot_base_pose = EventTerm(
        func=ds_events.set_default_base_pose,
        mode="startup",
        params={
            "q": [0.7071068, 0, 0, -0.7071068]
        },
    )
    init_worker_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )


@configclass
class WorkerBaseEnvCfg(DarkstoreBaseEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = WORKER_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot.init_state.pos = (0, 0, 0)

        self.scene.head_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/femto_holder_link/camera_3_base_link/head_camera",
            height=480,
            width=640,
            data_types=["rgb",],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(0.5004, -0.5, 0.5, -0.5), convention="ros"),
        )

        self.observations.rgb_camera.head_img = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("head_camera")},
            # noise=Unoise(n_min=-0.1, n_max=0.1),
            # clip=(-1.0, 1.0),
        )

        self.actions = WorkerActionsCfg()
        self.events = WorkerEventCfg()
