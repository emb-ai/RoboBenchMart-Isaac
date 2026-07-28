"""Fixed-base Sberchel upper-body OpenXR teleop (Pink IK, 26D action).

Driven by ``scripts/teleop_se3_agent.py`` or ``scripts/teleop_sberchel_upperbody_xr_compare.py``.
"""

from __future__ import annotations

import torch
from pink.tasks import DampingTask, FrameTask

from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import ManusViveCfg, OpenXRDeviceCfg, XrCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.utils import configclass

from dsynth_isaac.devices.sberchel_retargeter import SberchelRetargeterCfg
from dsynth_isaac.robots.sberchel_cfg import SBERCHEL_UPPERBODY_XR_ROBOT
from dsynth_isaac.tasks.manager_based.darkstore_base_cfg import DarkstoreBaseEnvCfg
from dsynth_isaac.tasks.manager_based.sberchel_base_env_cfg import (
    SberchelEventCfg,
    SberchelObservationsCfg,
)


def setup_sberchel_xr_pink_and_teleop(cfg) -> None:
    """Shared Pink URDF export + OpenXR teleop device wiring."""
    from dsynth_isaac.robots.sberchel_mjcf_usd import (
        prepare_sberchel_pink_ik_urdf,
        resolve_sberchel_spawn_usd_path,
    )

    spawn = cfg.scene.robot.spawn
    patched_usd_path = resolve_sberchel_spawn_usd_path(spawn)
    pink_urdf_path, _pink_mesh_dir, _raw_urdf_path = prepare_sberchel_pink_ik_urdf(patched_usd_path)
    cfg.actions.upper_body_ik.controller.urdf_path = pink_urdf_path
    cfg.actions.upper_body_ik.controller.mesh_path = None

    cfg.teleop_devices = DevicesCfg(
        devices={
            "handtracking": OpenXRDeviceCfg(
                retargeters=[
                    SberchelRetargeterCfg(
                        enable_visualization=True,
                        num_open_xr_hand_joints=2 * cfg.NUM_OPENXR_HAND_JOINTS,
                        sim_device=cfg.sim.device,
                        hand_joint_names=cfg.actions.upper_body_ik.hand_joint_names,
                    ),
                ],
                sim_device=cfg.sim.device,
                xr_cfg=cfg.xr,
            ),
            "manusvive": ManusViveCfg(
                retargeters=[
                    SberchelRetargeterCfg(
                        enable_visualization=True,
                        num_open_xr_hand_joints=2 * cfg.NUM_OPENXR_HAND_JOINTS,
                        sim_device=cfg.sim.device,
                        hand_joint_names=cfg.actions.upper_body_ik.hand_joint_names,
                    ),
                ],
                sim_device=cfg.sim.device,
                xr_cfg=cfg.xr,
            ),
        }
    )

# Pink ``find_joints`` applies hand targets in articulation DOF order (not MJCF
# name order). On open MJCF/USD, thumb pitch comes after digit proximals per hand.
SBERCHEL_HAND_JOINT_NAMES = [
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

_LEFT_HAND_BASE_FRAME = "left_hand_base_link"
_RIGHT_HAND_BASE_FRAME = "right_hand_base_link"


@configclass
class UpperbodySberchelXRActionsCfg:
    """Pink IK only: ``[left_wrist(7), right_wrist(7), fingers(12)]``."""

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
        hand_joint_names=SBERCHEL_HAND_JOINT_NAMES,
        target_eef_link_names={
            "left_wrist": _LEFT_HAND_BASE_FRAME,
            "right_wrist": _RIGHT_HAND_BASE_FRAME,
        },
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="pelvis_link",
            num_hand_joints=len(SBERCHEL_HAND_JOINT_NAMES),
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
class UpperbodySberchelXRFetchAnyItemEnvCfg(DarkstoreBaseEnvCfg):
    """Fixed-pelvis open-kinematics Sberchel + XR upper-body teleop."""

    actions = UpperbodySberchelXRActionsCfg()
    events = SberchelEventCfg()
    observations: SberchelObservationsCfg = SberchelObservationsCfg()

    target_item_name_regex: str = ".*"
    language_instruction: str = "XR teleop Sberchel upper body"

    xr: XrCfg = XrCfg(
        anchor_pos=(4.0, 5.0, 0.0),
        anchor_rot=(0.0, 0.0, 1.0, 0.0),
    )

    NUM_OPENXR_HAND_JOINTS = 26

    # [left_wrist(7), right_wrist(7), hand_joints(12)] — world-frame wxyz.
    # Wrist poses are near the XR anchor (see ``xr.anchor_pos``), for inactive teleop / replay.
    # Demo scripts should sample wrist targets from the live robot after reset instead.
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

        self.scene.robot = SBERCHEL_UPPERBODY_XR_ROBOT.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot_fingers_contact_prim_path = None
        self.scene.ee_frame = None

        self.decimation = 6
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 2

        setup_sberchel_xr_pink_and_teleop(self)
