"""Sberchel v3.0a locomanipulator robot configuration.

Reuses robot definitions from ``green_challenge`` (serialized interface JSON +
``ConfigManager``). ``SBERCHEL_CONFIG`` is closed kinematics (``robot_v3_0a``).
``SBERCHEL_CONFIG_OPEN_KINEMATICS`` uses the ``robot_green_open_source`` package
with spawn pointed at the local hybrid MJCF under ``mjcf/`` (serial upper body +
policy-native crank leg joints; meshes still resolved from green_challenge).
``SBERCHEL_UPPERBODY_XR_ROBOT`` adds fixed pelvis + ImplicitActuator PD for
upper-body XR teleop. ``SBERCHEL_XR_LOCOMOTION_ROBOT`` is free-floating open
kinematics with policy leg actuators and XR arm PD.
``SBERCHEL_XR_LOCOMOTION_NECK_ROBOT`` adds a separate ImplicitActuator neck
group for headset look control.
"""

import os
import re
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

# Hybrid MJCF lives in this repo; meshes stay in green_challenge.
_HYBRID_MJCF_DIR = Path(__file__).resolve().parent / "mjcf"
_HYBRID_MJCF_SRC = _HYBRID_MJCF_DIR / "v3_0_alpha_locomanipulator_hybrid.xml"
_GC_MESH_DIR = (
    Path(_GREEN_CHALLENGE_ROOT)
    / "assets/robot_v3_0a/dependencies/robots/v3_0_alpha_locomanipulator"
    / "meshes"
)


def _resolve_hybrid_mjcf(src: Path, mesh_dir: Path) -> Path:
    """Write a sibling MJCF with ``meshdir`` pointing at green_challenge meshes.

    Isaac Sim's native MJCF importer mirrors ``meshdir`` under its own cache
    directory (e.g. ``~/.cache/dirol/mjcf/<meshdir>/<mesh>_tmp/``) without
    recursively creating missing parents, so it silently fails to write the
    per-mesh tmp USD when ``meshdir`` is a long absolute path. Keep it a
    shallow relative path instead, via a local symlink to the real mesh dir.
    """
    # IsaacLab's MjcfConverter does a naive `basename.split(".")` expecting
    # exactly one dot, so the resolved filename must not add another one.
    resolved = src.with_name(f"{src.stem}_resolved{src.suffix}")
    mesh_link = src.parent / "meshes"
    abs_mesh = mesh_dir.resolve()
    if mesh_link.is_symlink():
        if mesh_link.resolve() != abs_mesh:
            mesh_link.unlink()
            mesh_link.symlink_to(abs_mesh)
    elif mesh_link.exists():
        raise FileExistsError(f"{mesh_link} exists and is not the expected meshes symlink")
    else:
        mesh_link.symlink_to(abs_mesh)

    text = src.read_text()
    patched = re.sub(
        r'(<compiler\b[^>]*\bmeshdir=")[^"]*(")',
        r"\g<1>meshes/\g<2>",
        text,
        count=1,
    )
    if patched == text and 'meshdir="' not in text:
        raise ValueError(f"No meshdir attribute found in {src}")
    if not resolved.exists() or resolved.read_text() != patched:
        resolved.write_text(patched)
    return resolved


_HYBRID_MJCF_PATH = _resolve_hybrid_mjcf(_HYBRID_MJCF_SRC, _GC_MESH_DIR)

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
#_upperbody_joint_pos[".*_elbow_pitch_joint"] = -1.0

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
#_xr_locomotion_joint_pos[".*_elbow_pitch_joint"] = -1.0

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


# --- green_challenge-native upper-body PD ------------------------------------
#
# These are the gains the green_challenge evaluation env actually runs, so demos
# recorded with them are demos of a plant that exists. Two sources combine:
#
#   1. ``sberchel_v3_0_alpha_locomanipulation_full.json`` (the same interface JSON
#      this module loads at import time) gives the raw per-joint PD.
#   2. green_challenge routes *body* joints (legs, torso, arms, neck) through
#      ``jm_step`` -> ``scripts/pos2torque.py``, which converts position targets
#      to torque with ``kp_gain_scale = kd_gain_scale = 1.4``. That path is on by
#      default (``ENABLE_JOINT_MANAGER_TORQUE_CONTROL``, ``envs/manip_env.py:44``;
#      only an env var disables it). Wrists and fingers stay position-controlled
#      and are NOT scaled.
#
# So the effective gains below are JSON x 1.4 for arms/neck, JSON as-is for
# wrists/fingers. Matching *effective* values is the point: raw JSON values would
# leave us 1.4x soft on every arm joint.
#
# Per-joint detail matters and is easy to lose. green_challenge distinguishes
# shoulder_yaw from shoulder pitch/roll, and thumb from the other four digits.
# An earlier version of this table matched on substrings ("shoulder", "proximal")
# and silently collapsed both, leaving the thumb -- the opposing digit that
# carries most of the grip force -- 4x too soft. Match joints explicitly.
_GC_KP_SCALE = 1.4  # pos2torque.py kp_gain_scale (body joints only)

_GC_SHOULDER_PR_STIFFNESS = 30.0 * _GC_KP_SCALE  # 42.0  pitch / roll
_GC_SHOULDER_PR_DAMPING = 1.5 * _GC_KP_SCALE  # 2.1
_GC_SHOULDER_YAW_STIFFNESS = 20.0 * _GC_KP_SCALE  # 28.0
_GC_SHOULDER_YAW_DAMPING = 1.0 * _GC_KP_SCALE  # 1.4
_GC_ELBOW_STIFFNESS = 20.0 * _GC_KP_SCALE  # 28.0
_GC_ELBOW_DAMPING = 1.0 * _GC_KP_SCALE  # 1.4
_GC_NECK_STIFFNESS = 20.0 * _GC_KP_SCALE  # 28.0
_GC_NECK_DAMPING = 1.0 * _GC_KP_SCALE  # 1.4
# Position-controlled on their side: no 1.4 scale.
_GC_WRIST_STIFFNESS = 40.0
_GC_WRIST_DAMPING = 1.0
_GC_THUMB_STIFFNESS = 40.0
_GC_THUMB_DAMPING = 2.0
_GC_FINGER_STIFFNESS = 10.0
_GC_FINGER_DAMPING = 1.0


def _gc_upper_gains_for_joint(j_name: str) -> tuple[float, float] | None:
    """Return ``(stiffness, damping)`` for an ``xr_upper`` joint, or None.

    Order matters: thumb must be tested before the generic ``proximal`` branch,
    and shoulder yaw before the generic shoulder branch.
    """
    if "shoulder" in j_name:
        if "yaw" in j_name:
            return _GC_SHOULDER_YAW_STIFFNESS, _GC_SHOULDER_YAW_DAMPING
        return _GC_SHOULDER_PR_STIFFNESS, _GC_SHOULDER_PR_DAMPING
    if "elbow" in j_name:
        return _GC_ELBOW_STIFFNESS, _GC_ELBOW_DAMPING
    if "wrist" in j_name:
        return _GC_WRIST_STIFFNESS, _GC_WRIST_DAMPING
    if "thumb" in j_name:
        return _GC_THUMB_STIFFNESS, _GC_THUMB_DAMPING
    if "proximal" in j_name:
        return _GC_FINGER_STIFFNESS, _GC_FINGER_DAMPING
    return None


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


def apply_sberchel_loco_upper_pd_gains(robot, *, include_neck: bool = True) -> None:
    """Set ``xr_upper`` (and neck) PD to green_challenge's effective gains.

    This is the plant green_challenge evaluates on, so it is the plant demos
    intended for their VLA must be recorded on. Use it for the whole session --
    both WASD locomotion and static hand teleop -- not as one half of a runtime
    gain switch: switching plants mid-episode makes the dataset non-Markovian
    (identical observations, different action->outcome mapping) and produces a
    transition their env has no analogue for.

    ``include_neck=False`` keeps the stiff XR neck PD for headset look, at the
    cost of diverging from their config (neck sits in their ``torso_joint_pos``
    group and moves the head camera, so it does affect recorded images).
    """
    if "xr_upper" in robot.actuators:
        actuator = robot.actuators["xr_upper"]
        for local_idx, j_name in enumerate(actuator.joint_names):
            gains = _gc_upper_gains_for_joint(j_name)
            if gains is None:
                continue
            actuator.stiffness[:, local_idx] = gains[0]
            actuator.damping[:, local_idx] = gains[1]
    if include_neck and "neck" in robot.actuators:
        neck = robot.actuators["neck"]
        for local_idx, _j_name in enumerate(neck.joint_names):
            neck.stiffness[:, local_idx] = _GC_NECK_STIFFNESS
            neck.damping[:, local_idx] = _GC_NECK_DAMPING


def sberchel_upper_pd_gain_table(robot) -> dict[str, tuple[float, float]]:
    """Current ``(stiffness, damping)`` per joint for ``xr_upper`` + ``neck``.

    Used for the startup gain dump and for recording gain provenance into the
    demo dataset, so a converter never has to guess what plant produced a demo.
    """
    table: dict[str, tuple[float, float]] = {}
    for group in ("xr_upper", "neck"):
        actuator = robot.actuators.get(group)
        if actuator is None:
            continue
        for local_idx, j_name in enumerate(actuator.joint_names):
            table[j_name] = (
                float(actuator.stiffness[0, local_idx].item()),
                float(actuator.damping[0, local_idx].item()),
            )
    return table
