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

``SBERCHEL_UPPERBODY_XR_ROBOT`` adds a fixed pelvis for upper-body XR teleop.
``SBERCHEL_XR_LOCOMOTION_NECK_ROBOT`` is free-floating and is the robot demos are
recorded on: it splits green_challenge's single ``motors`` group into a
``locomotion`` half (legs + torso, driven by the RL policy) and an ``xr_upper``
half (arms, wrists, fingers, neck, driven by Pink IK and the headset). Both
halves keep their actuator class, gains, friction model, armature and limits --
the split is about who commands each joint, not about what the joint is.
"""

from pathlib import Path

from dsynth_isaac.green_challenge_path import ensure_green_challenge_importable

# --- Make the ``green_challenge`` checkout importable ------------------------
_GREEN_CHALLENGE_ROOT = ensure_green_challenge_importable()

# NOTE: importing ConfigManager enables the MJCF extension after AppLauncher.
from green_challenge.assets.robot_v3_0a.utils.config_manager import ConfigManager
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from dsynth_isaac.robots.sberchel_self_collision import with_mjcf_self_collision_filters

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
#
# ``func`` is green_challenge's ``spawn_from_mjcf`` with one post-spawn pass
# added: the importer drops every ``contype``/``conaffinity`` mask, so PhysX
# collides link pairs MuJoCo never does. Inside the hand that jams the thumb
# against the palm at the open-hand pose and makes both thumbs chatter at half
# the sim rate; see ``sberchel_self_collision``.
# green_challenge's unwrapped spawner, kept so a check script can spawn the
# masks-lost plant side by side with ours (scripts/sberchel_check_thumb_yaw.py).
GREEN_CHALLENGE_SPAWN_FUNC = _base.spawn.func

SBERCHEL_CONFIG = _base.replace(
    spawn=_base.spawn.replace(
        asset_path=str(_CLOSED_MJCF_PATH),
        usd_dir=str(_USD_CACHE_DIR),
        usd_file_name="v3_0_alpha_locomanipulator.usd",
        func=with_mjcf_self_collision_filters(GREEN_CHALLENGE_SPAWN_FUNC, _CLOSED_MJCF_PATH),
    ),
)

# Actuated wrist dof. The crank drives ``*_wrist_pitch_joint`` through a
# parallelogram four-bar (crank arm 0.02101 == output arm 0.02100, coupler
# 0.071251 == ground link 0.071251), so pitch angle == crank angle, 1:1 and
# same sign. Pink IK solves on the pitch joint and the target is written to the
# crank -- see ``sberchel_pink_actions.SberchelPinkIKAction``.
WRIST_MOTOR_JOINT_EXPR = ".*_wrist_crank_joint"
WRIST_OUTPUT_JOINT_EXPR = ".*_wrist_pitch_joint"

_XR_LOCK_STIFFNESS = 10000.0
_XR_LOCK_DAMPING = 1000.0

# --- green_challenge-native PD -----------------------------------------------
#
# Their live entry point (``run.sh``) passes ``--deactivate_torque``, so
# ``ENABLE_JOINT_MANAGER_TORQUE_CONTROL`` is False: ``jm_step`` never runs, the
# 1.4x position->torque path is off, and ``cfg.actions`` falls back to
# ``ExampleActionsPosCfg`` -- plain ``JointPositionAction`` on every group, the
# same way we drive the robot.
#
# So the plant is whatever sits in ``actuator.stiffness`` at run time, and that
# is set in two steps:
#
#   1. ``FrictionalActuator.__init__`` zeroes stiffness/damping wherever
#      ``force_control`` is true -- all 25 body joints. Wrists and fingers are
#      force_control False and keep the cfg values (40/40/10).
#   2. ``Network.__init__`` -> ``_restore_pd_gains_from_policy_cfg`` writes the
#      *policy* config's ``control.stiffness``/``damping`` for exactly those 25
#      body joints. That is the plant, and it follows whichever controller is
#      loaded (zest: 61.1/3.2 shoulders, 221.7 knee; walk: 30/1.5 and 350).
#
# Nothing here should therefore hardcode body-joint gains. Both groups below are
# green_challenge's ``motors`` group restricted to a joint subset, unchanged in
# every other respect, and the gains arrive with the policy.
#
# The fixed-base rig further down never loads a policy, so it needs literal
# numbers; those are zest's, spelled out at its definition.

# Joint patterns the ``xr_upper`` group owns: arms, wrists, fingers, neck. The
# legs and torso are the complement, in ``_locomotion_actuators`` below.
_UPPER_JOINT_EXPRS: list[str] = [
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
    "neck_yaw_joint",
    "neck_pitch_joint",
]
_UPPER_JOINT_EXPR_SET = frozenset(_UPPER_JOINT_EXPRS)

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
                # Legs/torso/neck stay locked above. This rig never constructs a
                # Network, so the arm gains cannot arrive with a policy: these are
                # zest's control.stiffness values written out literally.
                ".*_shoulder_pitch_joint": 61.1,
                ".*_shoulder_roll_joint": 61.1,
                ".*_shoulder_yaw_joint": 25.6,
                ".*_elbow_.*": 25.6,
                ".*_wrist_.*": 40.0,
                ".*_thumb_proximal_.*": 40.0,
                ".*_index_proximal_joint": 10.0,
                ".*_middle_proximal_joint": 10.0,
                ".*_ring_proximal_joint": 10.0,
                ".*_pinky_proximal_joint": 10.0,
            },
            damping={
                ".*_hip_.*": _XR_LOCK_DAMPING,
                ".*_knee_.*": _XR_LOCK_DAMPING,
                ".*_ankle_.*": _XR_LOCK_DAMPING,
                "torso_yaw_joint": _XR_LOCK_DAMPING,
                "neck_.*": _XR_LOCK_DAMPING,
                ".*_shoulder_pitch_joint": 3.2,
                ".*_shoulder_roll_joint": 3.2,
                ".*_shoulder_yaw_joint": 1.4,
                ".*_elbow_.*": 1.4,
                ".*_wrist_.*": 1.0,
                ".*_thumb_proximal_.*": 2.0,
                ".*_index_proximal_joint": 1.0,
                ".*_middle_proximal_joint": 1.0,
                ".*_ring_proximal_joint": 1.0,
                ".*_pinky_proximal_joint": 1.0,
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
    "neck",
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


def _locomotion_gain_dict(mapping: dict | None) -> dict | None:
    """Keep leg/torso gain keys only (for the split ``locomotion`` group)."""
    if mapping is None:
        return None
    return {
        key: value
        for key, value in mapping.items()
        if not any(s in key for s in _LOCOMOTION_GAIN_SKIP)
    }


def _upper_gain_dict(mapping: dict | None) -> dict | None:
    """Keep only the entries whose pattern is one of the upper-body joints."""
    if mapping is None:
        return None
    return {key: value for key, value in mapping.items() if key in _UPPER_JOINT_EXPR_SET}


def _xr_upper_actuator():
    """green_challenge's ``motors`` group, restricted to arms/wrists/fingers/neck.

    Same actuator class, armature, friction model, effort and velocity limits as
    their single ``motors`` group -- only the joint subset differs, because we
    split legs and torso into their own group. Two deliberate deviations, both
    forced by the fact that we command positions where they command torque:

    Nothing is overridden -- not the gains, not ``force_control``. The 25 body
    joints are force-controlled, so their cfg gains get zeroed at init and the
    real ones arrive when ``Network.__init__`` restores the loaded policy's
    table; wrists and fingers are position-controlled and keep the cfg's 40/40/10.
    That is exactly the sequence green_challenge runs.

    Drives ``*_wrist_crank_joint`` -- the motor dof of the wrist four-bar -- not
    the passive ``*_wrist_pitch_joint`` output.

    This group used to be an ``ImplicitActuatorCfg`` at 3000/100. Implicit gains
    reach PhysX exactly once, in ``Articulation._process_actuators_cfg``, and
    nothing re-reads ``actuator.stiffness`` afterwards -- so the runtime
    re-assert below could never move them, and every demo recorded before this
    change ran with arms ~70x stiffer than the plant it was meant to match.
    """
    from dataclasses import fields

    motors = SBERCHEL_CONFIG.actuators["motors"]
    overrides: dict = {"joint_names_expr": list(_UPPER_JOINT_EXPRS)}
    for field in fields(motors):
        if field.name == "class_type" or field.name in overrides:
            continue
        value = getattr(motors, field.name)
        if isinstance(value, dict):
            overrides[field.name] = _upper_gain_dict(value)
    return motors.replace(**overrides)


def _locomotion_actuators():
    """green_challenge's ``motors`` group, split into legs/torso and upper body.

    Their side has one group for the whole body. We split it only so the halves
    can be commanded differently -- legs from the RL policy, upper body from Pink
    IK -- not to change the plant: both keep the same actuator class, friction
    model, armature and limits. The neck sits with the upper body because the
    headset drives it as a position target.
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
    locomotion_overrides: dict = {"joint_names_expr": leg_torso_expr}
    for field in fields(motors):
        if field.name in ("class_type", "joint_names_expr"):
            continue
        value = getattr(motors, field.name)
        if isinstance(value, dict):
            locomotion_overrides[field.name] = _locomotion_gain_dict(value)
    locomotion = motors.replace(**locomotion_overrides)
    return {
        "locomotion": locomotion,
        "xr_upper": _xr_upper_actuator(),
        **_passive_actuators(),
    }


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
    actuators=_locomotion_actuators(),
    soft_joint_pos_limit_factor=SBERCHEL_CONFIG.soft_joint_pos_limit_factor,
)


def sberchel_upper_pd_gain_table(robot) -> dict[str, tuple[float, float]]:
    """Current ``(stiffness, damping)`` per joint of every actuated group.

    Used for the startup gain dump and for recording gain provenance into the
    demo dataset, so a converter never has to guess what plant produced a demo.
    """
    table: dict[str, tuple[float, float]] = {}
    for group in ("locomotion", "xr_upper"):
        actuator = robot.actuators.get(group)
        if actuator is None:
            continue
        for local_idx, j_name in enumerate(actuator.joint_names):
            table[j_name] = (
                float(actuator.stiffness[0, local_idx].item()),
                float(actuator.damping[0, local_idx].item()),
            )
    return table
