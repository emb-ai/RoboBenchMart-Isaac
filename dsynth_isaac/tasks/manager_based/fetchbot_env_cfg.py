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
from isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env_cfg import FrankaCubeStackIKRelMimicEnvCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from isaaclab.sensors import CameraCfg
import isaaclab.envs.mdp as mdp


from dsynth_isaac.tasks.manager_based.darkstore_base_cfg import DarkstoreBaseEnvCfg, DsEventCfg
from dsynth_isaac.robots.fetch_cfg import FETCHBOT_CONFIG
import dsynth_isaac.tasks.manager_based.mdp.fetch_events as fetch_events
import dsynth_isaac.tasks.manager_based.mdp.ds_events as ds_events
from dsynth_isaac.devices.mobile_se3_keyboard import MobileSe3KeyboardCfg
from dsynth_isaac.tasks.manager_based.fetching.fetching_env_cfg import FetchAnyItemEnvCfg

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
class FetchEventCfg(DsEventCfg):
    # TODO: do scene rearrangement?
    """Configuration for events."""
    init_fetch_pose = EventTerm(
        func=fetch_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0, 0.0, 0.0, 0.0, 0.0, -1.28, 1.51, 0.35, 1.81, 0.0, 1.47, 0.0, 0.1, 0.1, ],
        },
    )

@configclass
class FetchbotAnyItemEnvCfg(FetchAnyItemEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = FETCHBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot.init_state.pos = (0, 0, 0)
        self.scene.robot_fingers_contact_prim_path = '{ENV_REGEX_NS}/Robot/fetch/*._gripper_finger_link'

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
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        self.gripper_joint_names = ['l_gripper_finger_joint', 'r_gripper_finger_joint']

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/fetch/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fetch/gripper_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),

            ],
        )


@configclass
class FetchbotIKRelAnyItemEnvCfg(FetchbotAnyItemEnvCfg):

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

# @configclass
# class FetchDsynthIKRelMimicEnvCfg(FetchIKBaseEnvCfg, FrankaCubeStackIKRelMimicEnvCfg):
#     """
#     Isaac Lab Mimic environment config class for Franka Cube Stack IK Rel env.
#     """

#     def __post_init__(self):
#         # post init of parents
#         super().__post_init__()
#         # # TODO: Figure out how we can move this to the MimicEnvCfg class
#         # # The __post_init__() above only calls the init for FrankaCubeStackEnvCfg and not MimicEnvCfg
#         # # https://stackoverflow.com/questions/59986413/achieving-multiple-inheritance-using-python-dataclasses

#         # Override the existing values
#         self.datagen_config.name = "demo_src_stack_isaac_lab_task_D0"
#         self.datagen_config.generation_guarantee = True
#         self.datagen_config.generation_keep_failed = True
#         self.datagen_config.generation_num_trials = 10
#         self.datagen_config.generation_select_src_per_subtask = True
#         self.datagen_config.generation_transform_first_robot_pose = False
#         self.datagen_config.generation_interpolate_from_last_target_pose = True
#         self.datagen_config.generation_relative = True
#         self.datagen_config.max_num_failures = 25
#         self.datagen_config.seed = 1

#         # The following are the subtask configurations for the stack task.
#         subtask_configs = []
#         subtask_configs.append(
#             SubTaskConfig(
#                 # Each subtask involves manipulation with respect to a single object frame.
#                 object_ref="cube_2",
#                 # This key corresponds to the binary indicator in "datagen_info" that signals
#                 # when this subtask is finished (e.g., on a 0 to 1 edge).
#                 subtask_term_signal="grasp_1",
#                 # Specifies time offsets for data generation when splitting a trajectory into
#                 # subtask segments. Random offsets are added to the termination boundary.
#                 subtask_term_offset_range=(10, 20),
#                 # Selection strategy for the source subtask segment during data generation
#                 selection_strategy="nearest_neighbor_object",
#                 # Optional parameters for the selection strategy function
#                 selection_strategy_kwargs={"nn_k": 3},
#                 # Amount of action noise to apply during this subtask
#                 action_noise=0.03,
#                 # Number of interpolation steps to bridge to this subtask segment
#                 num_interpolation_steps=5,
#                 # Additional fixed steps for the robot to reach the necessary pose
#                 num_fixed_steps=0,
#                 # If True, apply action noise during the interpolation phase and execution
#                 apply_noise_during_interpolation=False,
#                 description="Grasp red cube",
#                 next_subtask_description="Stack red cube on top of blue cube",
#             )
#         )
        # subtask_configs.append(
        #     SubTaskConfig(
        #         # Each subtask involves manipulation with respect to a single object frame.
        #         object_ref="cube_1",
        #         # Corresponding key for the binary indicator in "datagen_info" for completion
        #         subtask_term_signal="stack_1",
        #         # Time offsets for data generation when splitting a trajectory
        #         subtask_term_offset_range=(10, 20),
        #         # Selection strategy for source subtask segment
        #         selection_strategy="nearest_neighbor_object",
        #         # Optional parameters for the selection strategy function
        #         selection_strategy_kwargs={"nn_k": 3},
        #         # Amount of action noise to apply during this subtask
        #         action_noise=0.03,
        #         # Number of interpolation steps to bridge to this subtask segment
        #         num_interpolation_steps=5,
        #         # Additional fixed steps for the robot to reach the necessary pose
        #         num_fixed_steps=0,
        #         # If True, apply action noise during the interpolation phase and execution
        #         apply_noise_during_interpolation=False,
        #         next_subtask_description="Grasp green cube",
        #     )
        # )
        # subtask_configs.append(
        #     SubTaskConfig(
        #         # Each subtask involves manipulation with respect to a single object frame.
        #         object_ref="cube_3",
        #         # Corresponding key for the binary indicator in "datagen_info" for completion
        #         subtask_term_signal="grasp_2",
        #         # Time offsets for data generation when splitting a trajectory
        #         subtask_term_offset_range=(10, 20),
        #         # Selection strategy for source subtask segment
        #         selection_strategy="nearest_neighbor_object",
        #         # Optional parameters for the selection strategy function
        #         selection_strategy_kwargs={"nn_k": 3},
        #         # Amount of action noise to apply during this subtask
        #         action_noise=0.03,
        #         # Number of interpolation steps to bridge to this subtask segment
        #         num_interpolation_steps=5,
        #         # Additional fixed steps for the robot to reach the necessary pose
        #         num_fixed_steps=0,
        #         # If True, apply action noise during the interpolation phase and execution
        #         apply_noise_during_interpolation=False,
        #         next_subtask_description="Stack green cube on top of red cube",
        #     )
        # )
        # subtask_configs.append(
        #     SubTaskConfig(
        #         # Each subtask involves manipulation with respect to a single object frame.
        #         object_ref="cube_2",
        #         # End of final subtask does not need to be detected
        #         subtask_term_signal=None,
        #         # No time offsets for the final subtask
        #         subtask_term_offset_range=(0, 0),
        #         # Selection strategy for source subtask segment
        #         selection_strategy="nearest_neighbor_object",
        #         # Optional parameters for the selection strategy function
        #         selection_strategy_kwargs={"nn_k": 3},
        #         # Amount of action noise to apply during this subtask
        #         action_noise=0.03,
        #         # Number of interpolation steps to bridge to this subtask segment
        #         num_interpolation_steps=5,
        #         # Additional fixed steps for the robot to reach the necessary pose
        #         num_fixed_steps=0,
        #         # If True, apply action noise during the interpolation phase and execution
        #         apply_noise_during_interpolation=False,
        #     )
        # )
        # self.subtask_configs["fetch"] = subtask_configs
