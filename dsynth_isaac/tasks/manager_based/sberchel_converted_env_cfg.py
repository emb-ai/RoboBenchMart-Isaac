"""Darkstore envs using the USD-baked Sberchel (``SBERCHEL_CONVERTED_*``).

Uses the flattened USD robot only — not the MJCF ``SBERCHEL_CONFIG`` path.

Also registers converted-open upperbody XR teleop (VisionRobotDemo finger curl +
Isaac wrist offsets) without modifying MJCF ``UpperbodySberchelXR*``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from pink.tasks import DampingTask, FrameTask

from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import ManusViveCfg, OpenXRDeviceCfg, XrCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.utils import configclass

from dsynth_isaac.devices.sberchel_converted_retargeter import SberchelConvertedRetargeterCfg
from dsynth_isaac.robots.sberchel_converted_cfg import (
    SBERCHEL_CONVERTED_CONFIG,
    SBERCHEL_CONVERTED_FIXED_BASE_CONFIG,
    sberchel_converted_upperbody_xr_robot_cfg,
)
from dsynth_isaac.tasks.manager_based.darkstore_base_cfg import DarkstoreBaseEnvCfg
from dsynth_isaac.tasks.manager_based.sberchel_base_env_cfg import (
    SberchelBaseEnvCfg,
    SberchelEventCfg,
    SberchelObservationsCfg,
    _prepare_sberchel_pink_ik_urdf,
)

# Pink ``find_joints(..., preserve_order=False)`` applies hand targets in articulation
# DOF order. On the open converted USD, thumb pitch comes *after* digit proximals:
# L_yaw, L_index..L_pinky, R_yaw, R_index..R_pinky, L_thumb_pitch, R_thumb_pitch.
# Pack retargeter output in that same order (do not use MJCF SBERCHEL_HAND_JOINT_NAMES).
CONVERTED_SBERCHEL_HAND_JOINT_NAMES = [
    "left_thumb_proximal_yaw_joint",
    "left_index_proximal_joint",
    "left_middle_proximal_joint",
    "left_ring_proximal_joint",
    "left_pinky_proximal_joint",
    "right_thumb_proximal_yaw_joint",
    "right_index_proximal_joint",
    "right_middle_proximal_joint",
    "right_ring_proximal_joint",
    "right_pinky_proximal_joint",
    "left_thumb_proximal_pitch_joint",
    "right_thumb_proximal_pitch_joint",
]


class SberchelConvertedBaseEnvCfg(SberchelBaseEnvCfg):
    """Same as ``SberchelBaseEnvCfg``, but spawns the free-floating converted USD robot."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SBERCHEL_CONVERTED_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]


class SberchelConvertedFixedBaseEnvCfg(SberchelBaseEnvCfg):
    """Converted USD Sberchel with fixed pelvis (``fix_root_link=True``)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SBERCHEL_CONVERTED_FIXED_BASE_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]


_LEFT_HAND_BASE_FRAME = "left_hand_base_link"
_RIGHT_HAND_BASE_FRAME = "right_hand_base_link"


@configclass
class ConvertedUpperbodySberchelXRActionsCfg:
    """Pink IK upper body for converted-open USD — 26D wrists + proximal fingers."""

    upper_body_ik = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_elbow_yaw_joint",
            "left_wrist_pitch_joint",
            "left_wrist_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_elbow_yaw_joint",
            "right_wrist_pitch_joint",
            "right_wrist_roll_joint",
        ],
        hand_joint_names=CONVERTED_SBERCHEL_HAND_JOINT_NAMES,
        target_eef_link_names={
            "left_wrist": _LEFT_HAND_BASE_FRAME,
            "right_wrist": _RIGHT_HAND_BASE_FRAME,
        },
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="pelvis_link",
            num_hand_joints=len(CONVERTED_SBERCHEL_HAND_JOINT_NAMES),
            show_ik_warnings=True,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTask(
                    _LEFT_HAND_BASE_FRAME,
                    position_cost=8.0,
                    orientation_cost=2.0,
                    lm_damping=10.0,
                    gain=0.5,
                ),
                FrameTask(
                    _RIGHT_HAND_BASE_FRAME,
                    position_cost=8.0,
                    orientation_cost=2.0,
                    lm_damping=10.0,
                    gain=0.5,
                ),
                DampingTask(cost=0.5),
                NullSpacePostureTask(
                    cost=0.5,
                    lm_damping=1.0,
                    gain=0.3,
                    controlled_frames=[_LEFT_HAND_BASE_FRAME, _RIGHT_HAND_BASE_FRAME],
                    controlled_joints=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                    ],
                ),
            ],
            fixed_input_tasks=[],
            xr_enabled=True,
        ),
        enable_gravity_compensation=False,
    )


@configclass
class ConvertedUpperbodySberchelXRFetchAnyItemEnvCfg(DarkstoreBaseEnvCfg):
    """Fixed-base open converted USD + XR upper-body teleop for ``teleop_se3_agent``.

    Action space is Pink IK (26D): ``[left_wrist(7), right_wrist(7), fingers(12)]``.
    Fingers use ``SberchelConvertedRetargeter`` (VisionRobotDemo curl math).
    """

    actions = ConvertedUpperbodySberchelXRActionsCfg()
    events = SberchelEventCfg()
    observations: SberchelObservationsCfg = SberchelObservationsCfg()

    # Required by ``FetchItemEnv.setup_target_items`` (teleop scripts also overwrite).
    target_item_name_regex: str = ".*"
    language_instruction: str = "XR teleop converted Sberchel upper body"

    xr: XrCfg = XrCfg(
        anchor_pos=(4.0, 5.0, 0.0),
        anchor_rot=(0.0, 0.0, 1.0, 0.0),
    )

    NUM_OPENXR_HAND_JOINTS = 26
    temp_urdf_dir = tempfile.gettempdir()

    idle_action = torch.tensor(
        [
            3.90,
            5.95,
            1.15,
            0.707,
            0.0,
            0.0,
            0.707,
            4.30,
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

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = sberchel_converted_upperbody_xr_robot_cfg().replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        if self.scene.robot.spawn.rigid_props is not None:
            self.scene.robot.spawn.rigid_props.disable_gravity = True
        # Converted USD hierarchy does not match the MJCF ``/Robot/root/.*_thumb.*``
        # contact glob; disable grasp contact sensors for XR teleop (same as GR1T2).
        self.scene.robot_fingers_contact_prim_path = None
        self.scene.ee_frame = None

        self.decimation = 6
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 2

        open_usd = Path(self.scene.robot.spawn.usd_path)
        pink_urdf_path, pink_mesh_dir, raw_urdf_path = _prepare_sberchel_pink_ik_urdf(open_usd)
        self.actions.upper_body_ik.controller.urdf_path = pink_urdf_path
        self.actions.upper_body_ik.controller.mesh_path = pink_mesh_dir

        dex_cache_dir = str(Path(pink_urdf_path).parent.parent)
        idle = self.idle_action.detach().cpu().numpy()
        retargeter_kwargs = {
            "full_robot_urdf_path": raw_urdf_path,
            "dex_cache_dir": dex_cache_dir,
            "left_idle_wrist_pose": tuple(idle[0:7].tolist()),
            "right_idle_wrist_pose": tuple(idle[7:14].tolist()),
        }

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        SberchelConvertedRetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                            **retargeter_kwargs,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "manusvive": ManusViveCfg(
                    retargeters=[
                        SberchelConvertedRetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                            **retargeter_kwargs,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
