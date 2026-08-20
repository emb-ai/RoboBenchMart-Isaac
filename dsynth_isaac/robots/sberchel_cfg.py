"""Sberchel v3.0a locomanipulator robot configuration (closed kinematics).

Reuses the robot definition from ``green_challenge`` (serialized interface JSON +
``ConfigManager``) for the ``robot_v3_0a`` package -- the robot green_challenge
evaluates on. ``SBERCHEL_CONFIG`` spawns their closed-kinematics MJCF
(``v3_0_alpha_locomanipulator.xml``): the knee, ankle and wrist four-bar
linkages are present, and their ``connect`` equalities become USD spherical
joints inside green_challenge's own ``spawn_from_mjcf``. The actuated leg /
wrist dofs are therefore the *crank* joints, exactly as in their
``joint_groups.json`` action space; ``*_knee_pitch``, ``*_ankle_pitch``,
``*_ankle_roll`` and ``*_wrist_pitch`` are passive outputs driven through the
linkage.

``SBERCHEL_UPPERBODY_XR_ROBOT`` adds a fixed pelvis + ImplicitActuator PD for
upper-body XR teleop. ``SBERCHEL_XR_LOCOMOTION_NECK_ROBOT`` is free-floating
with policy leg actuators, XR arm PD and a separate ImplicitActuator neck group
for headset look control.
"""

from pathlib import Path

from dsynth_isaac.green_challenge_path import ensure_green_challenge_importable

# --- Make the ``green_challenge`` checkout importable ------------------------
_GREEN_CHALLENGE_ROOT = ensure_green_challenge_importable()

# NOTE: importing ConfigManager enables the MJCF extension after AppLauncher.
from green_challenge.assets.robot_v3_0a.utils.config_manager import ConfigManager
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_GC_ROBOT_DIR = (
    Path(_GREEN_CHALLENGE_ROOT)
    / "assets/robot_v3_0a/dependencies/robots/v3_0_alpha_locomanipulator"
)
# green_challenge's own closed-kinematics MJCF. Not the ``_open`` sibling: that
# one deletes the crank/rod linkages, renames the actuated leg dofs and flips
# ``right_wrist_roll_joint``'s axis, so demos recorded on it are demos of a
# plant their evaluation stack does not have.
_CLOSED_MJCF_PATH = _GC_ROBOT_DIR / "v3_0_alpha_locomanipulator.xml"
if not _CLOSED_MJCF_PATH.is_file():
    raise FileNotFoundError(
        f"green_challenge closed-kinematics MJCF not found: {_CLOSED_MJCF_PATH} "
        "(set GREEN_CHALLENGE_PATH to the checkout directory)."
    )

# MjcfConverter writes the converted USD here. It must not default into the
# green_challenge checkout, which is another user's read-only directory.
_USD_CACHE_DIR = Path(__file__).resolve().parent / "mjcf" / "usd_cache"

_cm = ConfigManager.instance()
_base = _cm.load_config("sberchel_v3_0_alpha_locomanipulation_full")

# The eight knee/ankle/wrist four-bars become USD spherical joints EXCLUDED from
# the articulation, so PhysX solves them as maximal-coordinate constraints and
# their rigidity depends on the solver. The serialized 4/4 iteration counts are
# kept as green_challenge ships them -- they are only adequate under the PGS
# solver, which green_challenge selects in ``envs/env_registration.make_env``
# and which ``apply_green_challenge_physx`` mirrors here. Under IsaacLab's
# default TGS solver the same 4/4 leaves the loops compliant and the zest policy
# limit-cycles: measured base path while standing at zero command over 15 s is
# 1.66 m on TGS 4/4 vs 0.044 m on PGS 4/4 (0.049 m on the retired open model).
# So do not raise these to paper over a solver mismatch -- set the solver.

# The serialized ``asset_path`` is a stale absolute path from the machine the
# config was captured on, and ``usd_dir`` is null; point both at this checkout.
SBERCHEL_CONFIG = _base.replace(
    spawn=_base.spawn.replace(
        asset_path=str(_CLOSED_MJCF_PATH),
        usd_dir=str(_USD_CACHE_DIR),
        usd_file_name="v3_0_alpha_locomanipulator.usd",
    ),
)

# Actuated wrist dof. The crank drives ``*_wrist_pitch_joint`` through a
# parallelogram four-bar (crank arm 0.02101 == output arm 0.02100, coupler
# 0.071251 == ground link 0.071251), so pitch angle == crank angle, 1:1 and
# same sign. Pink IK solves on the pitch joint and the target is written to the
# crank -- see ``sberchel_pink_actions.SberchelPinkIKAction``.
WRIST_MOTOR_JOINT_EXPR = ".*_wrist_crank_joint"
WRIST_OUTPUT_JOINT_EXPR = ".*_wrist_pitch_joint"

_XR_ARM_STIFFNESS = 3000.0
_XR_ARM_DAMPING = 100.0
_XR_FINGER_STIFFNESS = 400.0
_XR_FINGER_DAMPING = 40.0
_XR_NECK_STIFFNESS = 200.0
_XR_NECK_DAMPING = 20.0
_XR_LOCK_STIFFNESS = 10000.0
_XR_LOCK_DAMPING = 1000.0


def _passive_actuators() -> dict:
    """green_challenge's ``passive`` group for the linkage output joints.

    Covers ``.*_ankle_pitch``, ``.*_ankle_roll``, ``.*_knee_pitch`` and
    ``.*_wrist_pitch`` -- the dofs the four-bars drive. It carries no gains; it
    exists so Isaac Lab accounts for those joints instead of leaving them
    unclaimed. The remaining rod/inter dofs stay unclaimed, as on their side.
    """
    passive = SBERCHEL_CONFIG.actuators.get("passive")
    return {"passive": passive} if passive is not None else {}


_upperbody_joint_pos = dict(SBERCHEL_CONFIG.init_state.joint_pos)

_upperbody_spawn = SBERCHEL_CONFIG.spawn.replace(
    rigid_props=SBERCHEL_CONFIG.spawn.rigid_props.replace(disable_gravity=True)
    if SBERCHEL_CONFIG.spawn.rigid_props is not None
    else None,
    articulation_props=SBERCHEL_CONFIG.spawn.articulation_props.replace(
        fix_root_link=True,
        enabled_self_collisions=False,
    )
    if SBERCHEL_CONFIG.spawn.articulation_props is not None
    else None,
)

SBERCHEL_UPPERBODY_XR_ROBOT = ArticulationCfg(
    spawn=_upperbody_spawn,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=SBERCHEL_CONFIG.init_state.pos,
        rot=SBERCHEL_CONFIG.init_state.rot,
        joint_pos=_upperbody_joint_pos,
        joint_vel=SBERCHEL_CONFIG.init_state.joint_vel,
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
                WRIST_MOTOR_JOINT_EXPR,
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
        **_passive_actuators(),
    },
    soft_joint_pos_limit_factor=SBERCHEL_CONFIG.soft_joint_pos_limit_factor,
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
    """ImplicitActuator PD for XR arm/finger teleop.

    Drives ``*_wrist_crank_joint`` -- the motor dof of the wrist four-bar --
    not the passive ``*_wrist_pitch_joint`` output.
    """
    return ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_pitch_joint",
            ".*_elbow_yaw_joint",
            WRIST_MOTOR_JOINT_EXPR,
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


def _locomotion_actuators(*, neck_on_locomotion: bool = True):
    """Frictional leg/torso actuators + ImplicitActuator arms for XR teleop.

    When ``neck_on_locomotion`` is False, neck joints are omitted from the
    frictional group so a separate ImplicitActuator can drive them.
    """
    from dataclasses import fields

    motors = SBERCHEL_CONFIG.actuators["motors"]
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
    return {
        "locomotion": locomotion,
        "xr_upper": _xr_upper_actuator(),
        **_passive_actuators(),
    }


def _locomotion_actuators_with_neck_pd():
    """Legs/torso frictional + neck ImplicitActuator PD + XR arm PD."""
    actuators = _locomotion_actuators(neck_on_locomotion=False)
    actuators["neck"] = ImplicitActuatorCfg(
        joint_names_expr=["neck_yaw_joint", "neck_pitch_joint"],
        stiffness={"neck_.*": _XR_NECK_STIFFNESS},
        damping={"neck_.*": _XR_NECK_DAMPING},
        effort_limit={"neck_.*": 36.0},
    )
    return actuators


_xr_locomotion_joint_pos = dict(_upperbody_joint_pos)

# Free-floating XR loco robot with neck on ImplicitActuator for headset look control.
SBERCHEL_XR_LOCOMOTION_NECK_ROBOT = ArticulationCfg(
    spawn=SBERCHEL_CONFIG.spawn,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=SBERCHEL_CONFIG.init_state.pos,
        rot=SBERCHEL_CONFIG.init_state.rot,
        joint_pos=_xr_locomotion_joint_pos,
        joint_vel=SBERCHEL_CONFIG.init_state.joint_vel,
    ),
    actuators=_locomotion_actuators_with_neck_pd(),
    soft_joint_pos_limit_factor=SBERCHEL_CONFIG.soft_joint_pos_limit_factor,
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
    and shoulder yaw before the generic shoulder branch. The ``wrist`` branch
    covers both ``wrist_crank`` (pitch motor) and ``wrist_roll``.
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
