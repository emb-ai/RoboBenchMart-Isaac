"""Sberchel v3.0a locomanipulator robot configuration.

Reuses robot definitions from ``green_challenge`` (serialized interface JSON +
``ConfigManager``). ``SBERCHEL_CONFIG`` is closed kinematics (``robot_v3_0a``).
``SBERCHEL_CONFIG_OPEN_KINEMATICS`` uses the ``robot_green_open_source`` package
(green_challenge open-kinematics robot) with spawn pointed at the serial open
MJCF file. ``SBERCHEL_UPPERBODY_XR_ROBOT`` adds fixed pelvis + ImplicitActuator
PD for upper-body XR teleop.
"""

import os
import sys
from pathlib import Path

# --- Make the sibling ``green_challenge`` package importable -----------------
_DEFAULT_GC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..", "green_challenge"))
_GREEN_CHALLENGE_ROOT = os.environ.get("GREEN_CHALLENGE_PATH", _DEFAULT_GC)
_parent = os.path.dirname(os.path.abspath(_GREEN_CHALLENGE_ROOT))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# NOTE: importing ConfigManager enables the MJCF extension after AppLauncher.
from green_challenge.assets.robot_v3_0a.utils.config_manager import ConfigManager
from green_challenge.assets.robot_green_open_source.utils.config_manager import (
    ConfigManager as ConfigManagerGreenOpenSource,
)
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_cm = ConfigManager.instance()
SBERCHEL_CONFIG = _cm.load_config("sberchel_v3_0_alpha_locomanipulation_full")

_cm_open = ConfigManagerGreenOpenSource.instance()
_base_open = _cm_open.load_config("sberchel_v3_0_alpha_locomanipulation_full")
_OPEN_MJCF_PATH = (
    Path(_GREEN_CHALLENGE_ROOT)
    / "assets/robot_v3_0a/dependencies/robots/v3_0_alpha_locomanipulator"
    / "v3_0_alpha_locomanipulator_open.xml"
)
_OPEN_MJCF_DIR = _OPEN_MJCF_PATH.parent
SBERCHEL_CONFIG_OPEN_KINEMATICS = _base_open.replace(
    spawn=_base_open.spawn.replace(
        asset_path=str(_OPEN_MJCF_PATH),
        usd_dir=str(_OPEN_MJCF_DIR / "usd_cache"),
        usd_file_name="v3_0_alpha_locomanipulator_open.usd",
    ),
)

# Closed-only init keys absent from the open MJCF model.
_OPEN_INIT_JOINT_DROP = (
    "left_knee_crank_joint",
    "left_knee_rod_joint",
    "left_ankle_u_crank_joint",
    "left_ankle_b_crank_joint",
    "right_knee_crank_joint",
    "right_knee_rod_joint",
    "right_ankle_u_crank_joint",
    "right_ankle_b_crank_joint",
)
_OPEN_INIT_JOINT_REMAP = {
    ".*_wrist_crank_joint": ".*_wrist_pitch_joint",
}


def _joint_dict_for_open_kinematics(joint_values: dict | None) -> dict:
    """Drop closed-only joints and rename wrist_crank → wrist_pitch for open MJCF."""
    if not joint_values:
        return {}
    drop = set(_OPEN_INIT_JOINT_DROP)
    out: dict = {}
    for key, value in joint_values.items():
        if key in drop:
            continue
        out[_OPEN_INIT_JOINT_REMAP.get(key, key)] = value
    return out


_XR_ARM_STIFFNESS = 3000.0
_XR_ARM_DAMPING = 100.0
_XR_FINGER_STIFFNESS = 400.0
_XR_FINGER_DAMPING = 40.0
_XR_LOCK_STIFFNESS = 10000.0
_XR_LOCK_DAMPING = 1000.0

_upperbody_joint_pos = _joint_dict_for_open_kinematics(
    dict(SBERCHEL_CONFIG_OPEN_KINEMATICS.init_state.joint_pos)
)
_upperbody_joint_pos[".*_elbow_pitch_joint"] = -1.0
# Defaults for serial leg joints (closed crank keys were dropped above).
_upperbody_joint_pos.setdefault("left_knee_pitch_joint", 0.299)
_upperbody_joint_pos.setdefault("right_knee_pitch_joint", 0.299)
_upperbody_joint_pos.setdefault("left_ankle_pitch_joint", -0.2)
_upperbody_joint_pos.setdefault("right_ankle_pitch_joint", -0.2)
_upperbody_joint_pos.setdefault("left_ankle_roll_joint", 0.0)
_upperbody_joint_pos.setdefault("right_ankle_roll_joint", 0.0)

_upperbody_spawn = SBERCHEL_CONFIG_OPEN_KINEMATICS.spawn.replace(
    rigid_props=SBERCHEL_CONFIG_OPEN_KINEMATICS.spawn.rigid_props.replace(disable_gravity=True)
    if SBERCHEL_CONFIG_OPEN_KINEMATICS.spawn.rigid_props is not None
    else None,
    articulation_props=SBERCHEL_CONFIG_OPEN_KINEMATICS.spawn.articulation_props.replace(
        fix_root_link=True,
        enabled_self_collisions=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    if SBERCHEL_CONFIG_OPEN_KINEMATICS.spawn.articulation_props is not None
    else None,
)

SBERCHEL_UPPERBODY_XR_ROBOT = ArticulationCfg(
    spawn=_upperbody_spawn,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=SBERCHEL_CONFIG_OPEN_KINEMATICS.init_state.pos,
        rot=SBERCHEL_CONFIG_OPEN_KINEMATICS.init_state.rot,
        joint_pos=_upperbody_joint_pos,
        joint_vel=SBERCHEL_CONFIG_OPEN_KINEMATICS.init_state.joint_vel,
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_pitch_joint",
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
                "torso_yaw_joint",
                "neck_yaw_joint",
                "neck_pitch_joint",
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
                ".*_thumb_proximal_yaw_joint",
                ".*_thumb_proximal_pitch_joint",
                ".*_index_proximal_joint",
                ".*_middle_proximal_joint",
                ".*_ring_proximal_joint",
                ".*_pinky_proximal_joint",
            ],
            stiffness={
                ".*_hip_.*": _XR_LOCK_STIFFNESS,
                ".*_knee_.*": _XR_LOCK_STIFFNESS,
                ".*_ankle_.*": _XR_LOCK_STIFFNESS,
                "torso_yaw_joint": _XR_LOCK_STIFFNESS,
                "neck_.*": _XR_LOCK_STIFFNESS,
                ".*_shoulder_.*": _XR_ARM_STIFFNESS,
                ".*_elbow_.*": _XR_ARM_STIFFNESS,
                ".*_wrist_.*": _XR_ARM_STIFFNESS,
                ".*_thumb_proximal_.*": _XR_FINGER_STIFFNESS,
                ".*_index_proximal_joint": _XR_FINGER_STIFFNESS,
                ".*_middle_proximal_joint": _XR_FINGER_STIFFNESS,
                ".*_ring_proximal_joint": _XR_FINGER_STIFFNESS,
                ".*_pinky_proximal_joint": _XR_FINGER_STIFFNESS,
            },
            damping={
                ".*_hip_.*": _XR_LOCK_DAMPING,
                ".*_knee_.*": _XR_LOCK_DAMPING,
                ".*_ankle_.*": _XR_LOCK_DAMPING,
                "torso_yaw_joint": _XR_LOCK_DAMPING,
                "neck_.*": _XR_LOCK_DAMPING,
                ".*_shoulder_.*": _XR_ARM_DAMPING,
                ".*_elbow_.*": _XR_ARM_DAMPING,
                ".*_wrist_.*": _XR_ARM_DAMPING,
                ".*_thumb_proximal_.*": _XR_FINGER_DAMPING,
                ".*_index_proximal_joint": _XR_FINGER_DAMPING,
                ".*_middle_proximal_joint": _XR_FINGER_DAMPING,
                ".*_ring_proximal_joint": _XR_FINGER_DAMPING,
                ".*_pinky_proximal_joint": _XR_FINGER_DAMPING,
            },
            effort_limit={
                ".*_hip_pitch_joint": 380.0,
                ".*_hip_roll_joint": 330.0,
                ".*_hip_yaw_joint": 150.0,
                ".*_knee_pitch_joint": 380.0,
                ".*_ankle_.*": 90.0,
                "torso_yaw_joint": 150.0,
                "neck_.*": 36.0,
                ".*_shoulder_pitch_joint": 90.0,
                ".*_shoulder_roll_joint": 90.0,
                ".*_shoulder_yaw_joint": 36.0,
                ".*_elbow_.*": 36.0,
                ".*_wrist_.*": 16.0,
                ".*_thumb_proximal_.*": 8.5,
                ".*_index_proximal_joint": 8.5,
                ".*_middle_proximal_joint": 8.5,
                ".*_ring_proximal_joint": 8.5,
                ".*_pinky_proximal_joint": 8.5,
            },
        ),
    },
    soft_joint_pos_limit_factor=SBERCHEL_CONFIG_OPEN_KINEMATICS.soft_joint_pos_limit_factor,
)
