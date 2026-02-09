from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg
import isaaclab.envs.mdp as mdp


from dsynth_isaac.tasks.manager_based.darkstore_base_cfg import DarkstoreBaseEnvCfg
from dsynth_isaac.robots.fetch_cfg import FETCH_CONFIG

@configclass
class FetchActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "forearm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
        ], scale=1.0)
    gripper_action = mdp.BinaryJointPositionActionCfg(asset_name="robot", joint_names=[
        "r_gripper_finger_joint",
        "l_gripper_finger_joint"
        ],
        open_command_expr={"r_gripper_finger_joint": 0.04, "l_gripper_finger_joint": 0.04},
        close_command_expr={"r_gripper_finger_joint": 0.01, "l_gripper_finger_joint": 0.01},
        )
    body_actions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[
        "torso_lift_joint"
    ], scale=1.0)
    head_actions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[
        "head_pan_joint", 
        "head_tilt_joint"
    ], scale=1.0)
    base_actions = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[
        "l_wheel_joint", 
        "r_wheel_joint"
        ], scale=1.0)


@configclass
class FetchBaseEnvCfg(DarkstoreBaseEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = FETCH_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot.init_state.pos = (0, 0, 0)

        self.scene.head_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/fetch/head_camera_link/head_camera_rgb_frame",
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


