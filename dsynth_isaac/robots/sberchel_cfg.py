"""Sberchel v3.0a locomanipulator robot configuration.

Reuses robot definitions from ``green_challenge`` (serialized interface JSON +
``ConfigManager``). ``SBERCHEL_CONFIG`` is closed kinematics (``robot_v3_0a``).
``SBERCHEL_CONFIG_OPEN_KINEMATICS`` uses the ``robot_green_open_source`` package
with spawn pointed at the hybrid open MJCF (serial upper body + policy-native
crank leg joints). ``SBERCHEL_UPPERBODY_XR_ROBOT`` adds fixed pelvis +
ImplicitActuator PD for upper-body XR teleop. ``SBERCHEL_XR_LOCOMOTION_ROBOT``
is free-floating open kinematics with policy leg actuators and XR arm PD.
``SBERCHEL_XR_LOCOMOTION_NECK_ROBOT`` adds a separate ImplicitActuator neck
group for headset look control.
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
_HYBRID_MJCF_PATH = (
    Path(_GREEN_CHALLENGE_ROOT)
    / "assets/robot_v3_0a/dependencies/robots/v3_0_alpha_locomanipulator"
    / "v3_0_alpha_locomanipulator_hybrid.xml"
)
_HYBRID_MJCF_DIR = _HYBRID_MJCF_PATH.parent

# green_open_source JSON uses wrist_crank; hybrid MJCF keeps serial wrist_pitch.
_HYBRID_JOINT_KEY_REMAP = {
    ".*_wrist_crank_joint": ".*_wrist_pitch_joint",
}


def _remap_hybrid_joint_keys(value):
    """Remap crank wrist keys to serial wrist_pitch for hybrid MJCF."""
    if isinstance(value, dict):
        return {_HYBRID_JOINT_KEY_REMAP.get(k, k): v for k, v in value.items()}
    if isinstance(value, list):
        return [_HYBRID_JOINT_KEY_REMAP.get(v, v) for v in value]
    return value


def _hybrid_motors_actuators():
    """Motors actuator group with wrist_pitch names for hybrid MJCF."""
    from dataclasses import fields

    motors = _base_open.actuators["motors"]
    overrides: dict = {}
    for field in fields(motors):
        if field.name in ("class_type",):
            continue
        attr = getattr(motors, field.name)
        if isinstance(attr, (dict, list)):
            overrides[field.name] = _remap_hybrid_joint_keys(attr)
    return {"motors": motors.replace(**overrides)}


_hybrid_init_joint_pos = _remap_hybrid_joint_keys(dict(_base_open.init_state.joint_pos))

SBERCHEL_CONFIG_OPEN_KINEMATICS = _base_open.replace(
    spawn=_base_open.spawn.replace(
        asset_path=str(_HYBRID_MJCF_PATH),
        usd_dir=str(_HYBRID_MJCF_DIR / "usd_cache"),
        usd_file_name="v3_0_alpha_locomanipulator_hybrid.usd",
    ),
    init_state=_base_open.init_state.replace(joint_pos=_hybrid_init_joint_pos),
    actuators=_hybrid_motors_actuators(),
)

_XR_ARM_STIFFNESS = 3000.0
_XR_ARM_DAMPING = 100.0
_XR_FINGER_STIFFNESS = 400.0
_XR_FINGER_DAMPING = 40.0
_XR_NECK_STIFFNESS = 200.0
_XR_NECK_DAMPING = 20.0
_XR_LOCK_STIFFNESS = 10000.0
_XR_LOCK_DAMPING = 1000.0

_upperbody_joint_pos = dict(SBERCHEL_CONFIG_OPEN_KINEMATICS.init_state.joint_pos)
_upperbody_joint_pos[".*_elbow_pitch_joint"] = -1.0

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
                ".*_knee_crank_joint",
                ".*_ankle_u_crank_joint",
                ".*_ankle_b_crank_joint",
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
                ".*_knee_crank_joint": 380.0,
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

# Upper-body / finger keys in motors cfg — not on the locomotion actuator group.
_LOCOMOTION_GAIN_SKIP = (
    "shoulder",
    "elbow",
    "wrist",
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky",
    "intermediate",
    "distal",
)


def _locomotion_gain_dict(mapping: dict | None, *, include_neck: bool = True) -> dict | None:
    """Keep leg/torso(/neck) gain keys only (for split ``locomotion`` actuators)."""
    if mapping is None:
        return None
    skip = _LOCOMOTION_GAIN_SKIP if include_neck else _LOCOMOTION_GAIN_SKIP + ("neck",)
    return {
        key: value
        for key, value in mapping.items()
        if not any(s in key for s in skip)
    }


def _xr_upper_actuator() -> ImplicitActuatorCfg:
    """ImplicitActuator PD for XR arm/finger teleop."""
    return ImplicitActuatorCfg(
        joint_names_expr=[
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
    )


def _open_locomotion_actuators(*, neck_on_locomotion: bool = True):
    """Frictional leg/torso actuators + ImplicitActuator arms for XR teleop.

    When ``neck_on_locomotion`` is False, neck joints are omitted from the
    frictional group so a separate ImplicitActuator can drive them.
    """
    from dataclasses import fields

    motors = _hybrid_motors_actuators()["motors"]
    leg_torso_expr = [
        ".*_hip_pitch_joint",
        ".*_hip_roll_joint",
        ".*_hip_yaw_joint",
        ".*_knee_crank_joint",
        ".*_ankle_u_crank_joint",
        ".*_ankle_b_crank_joint",
        "torso_yaw_joint",
    ]
    if neck_on_locomotion:
        leg_torso_expr.extend(["neck_yaw_joint", "neck_pitch_joint"])
    locomotion_overrides: dict = {"joint_names_expr": leg_torso_expr}
    for field in fields(motors):
        if field.name in ("class_type", "joint_names_expr"):
            continue
        value = getattr(motors, field.name)
        if isinstance(value, dict):
            locomotion_overrides[field.name] = _locomotion_gain_dict(
                value, include_neck=neck_on_locomotion
            )
    locomotion = motors.replace(**locomotion_overrides)
    return {"locomotion": locomotion, "xr_upper": _xr_upper_actuator()}


def _open_locomotion_actuators_with_neck_pd():
    """Legs/torso frictional + neck ImplicitActuator PD + XR arm PD."""
    actuators = _open_locomotion_actuators(neck_on_locomotion=False)
    actuators["neck"] = ImplicitActuatorCfg(
        joint_names_expr=["neck_yaw_joint", "neck_pitch_joint"],
        stiffness={"neck_.*": _XR_NECK_STIFFNESS},
        damping={"neck_.*": _XR_NECK_DAMPING},
        effort_limit={"neck_.*": 36.0},
    )
    return actuators


_xr_locomotion_joint_pos = dict(_upperbody_joint_pos)
_xr_locomotion_joint_pos[".*_elbow_pitch_joint"] = -1.0

SBERCHEL_XR_LOCOMOTION_ROBOT = ArticulationCfg(
    spawn=SBERCHEL_CONFIG_OPEN_KINEMATICS.spawn,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=SBERCHEL_CONFIG_OPEN_KINEMATICS.init_state.pos,
        rot=SBERCHEL_CONFIG_OPEN_KINEMATICS.init_state.rot,
        joint_pos=_xr_locomotion_joint_pos,
        joint_vel=SBERCHEL_CONFIG_OPEN_KINEMATICS.init_state.joint_vel,
    ),
    actuators=_open_locomotion_actuators(),
    soft_joint_pos_limit_factor=SBERCHEL_CONFIG_OPEN_KINEMATICS.soft_joint_pos_limit_factor,
)

# Free-floating XR loco robot with neck on ImplicitActuator for headset look control.
SBERCHEL_XR_LOCOMOTION_NECK_ROBOT = ArticulationCfg(
    spawn=SBERCHEL_CONFIG_OPEN_KINEMATICS.spawn,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=SBERCHEL_CONFIG_OPEN_KINEMATICS.init_state.pos,
        rot=SBERCHEL_CONFIG_OPEN_KINEMATICS.init_state.rot,
        joint_pos=_xr_locomotion_joint_pos,
        joint_vel=SBERCHEL_CONFIG_OPEN_KINEMATICS.init_state.joint_vel,
    ),
    actuators=_open_locomotion_actuators_with_neck_pd(),
    soft_joint_pos_limit_factor=SBERCHEL_CONFIG_OPEN_KINEMATICS.soft_joint_pos_limit_factor,
)


def apply_sberchel_xr_teleop_pd_gains(robot) -> None:
    """Raise arm/finger PD on ``xr_upper`` and neck PD when present."""
    if "xr_upper" in robot.actuators:
        actuator = robot.actuators["xr_upper"]
        for local_idx, j_name in enumerate(actuator.joint_names):
            if "shoulder" in j_name or "elbow" in j_name or "wrist" in j_name:
                actuator.stiffness[:, local_idx] = _XR_ARM_STIFFNESS
                actuator.damping[:, local_idx] = _XR_ARM_DAMPING
            elif "proximal" in j_name:
                actuator.stiffness[:, local_idx] = _XR_FINGER_STIFFNESS
                actuator.damping[:, local_idx] = _XR_FINGER_DAMPING
    if "neck" in robot.actuators:
        neck = robot.actuators["neck"]
        for local_idx, _j_name in enumerate(neck.joint_names):
            neck.stiffness[:, local_idx] = _XR_NECK_STIFFNESS
            neck.damping[:, local_idx] = _XR_NECK_DAMPING
