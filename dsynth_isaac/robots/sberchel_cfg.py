"""Sberchel v3.0a locomanipulator robot configuration.

This reuses the robot definition that already lives in the ``green_challenge``
repository instead of duplicating it here. ``green_challenge`` stores the full
Isaac Lab ``ArticulationCfg`` as a serialized JSON ("interface config") and
rebuilds it through its ``ConfigManager``. The rebuilt object is exactly the
same type as the other robots in this repo (e.g. ``FETCH_CONFIG``), so it can be
dropped straight into ``self.scene.robot``.

The config spawns the robot from a MuJoCo ``.xml`` (converted to USD on first
load) and uses green_challenge's custom frictional actuators.
"""

import os
import sys
from pathlib import Path

# --- Make the sibling ``green_challenge`` package importable -----------------
# It lives next to this repo (../green_challenge). The interface JSON references
# modules as ``green_challenge.assets.robot_v3_0a...``, so the *parent* of the
# green_challenge folder must be on sys.path. Override with GREEN_CHALLENGE_PATH.
# This file is <repo>/dsynth_isaac/robots/sberchel_cfg.py -> 3 dirnames to repo root.
_DEFAULT_GC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..", "green_challenge"))
_GREEN_CHALLENGE_ROOT = os.environ.get("GREEN_CHALLENGE_PATH", _DEFAULT_GC)
_parent = os.path.dirname(os.path.abspath(_GREEN_CHALLENGE_ROOT))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# NOTE: importing this enables the ``isaacsim.asset.importer.mjcf`` extension,
# so it must happen *after* the simulation app has launched. Because this module
# is only imported lazily by ``gym.make`` (via the env cfg entry point), that
# ordering is satisfied.
from green_challenge.assets.robot_v3_0a.utils.config_manager import ConfigManager

# Rebuild the full ArticulationCfg (MJCF spawn + frictional actuators) from the
# serialized "interface" config shipped in green_challenge.
_cm = ConfigManager.instance()
SBERCHEL_CONFIG = _cm.load_config("sberchel_v3_0_alpha_locomanipulation_full")

# Joints Pink IK drives during XR teleop (high PD required).
# Closed MJCF actuates wrist_crank (parallel pitch linkage). Open MJCF actuates
# wrist_pitch directly — see ``sberchel_upperbody_xr_robot_cfg``.
_SBERCHEL_XR_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "left_wrist_crank_joint",
    "left_wrist_roll_joint",
    "right_wrist_crank_joint",
    "right_wrist_roll_joint",
)

# Finger master joints commanded by dex-retargeting (need PD after FrictionalActuator spawn).
_SBERCHEL_XR_FINGER_JOINT_NAMES = (
    "left_thumb_proximal_yaw_joint",
    "left_thumb_proximal_pitch_joint",
    "left_index_proximal_joint",
    "left_middle_proximal_joint",
    "left_ring_proximal_joint",
    "left_pinky_proximal_joint",
    "right_thumb_proximal_yaw_joint",
    "right_thumb_proximal_pitch_joint",
    "right_index_proximal_joint",
    "right_middle_proximal_joint",
    "right_ring_proximal_joint",
    "right_pinky_proximal_joint",
)

# Torso/neck are not in the XR action vector; lock them for a stable IK chain.
_SBERCHEL_XR_LOCKED_JOINT_NAMES = (
    "torso_yaw_joint",
    "neck_yaw_joint",
    "neck_pitch_joint",
)

_XR_ARM_STIFFNESS = 3000.0
_XR_ARM_DAMPING = 100.0
_XR_FINGER_STIFFNESS = 400.0
_XR_FINGER_DAMPING = 40.0
_XR_LOCK_STIFFNESS = 10000.0
_XR_LOCK_DAMPING = 1000.0

# Exact init-state keys present on the closed model but absent from open MJCF.
_OPEN_MJCF_INIT_JOINT_DROP = (
    "left_knee_crank_joint",
    "left_knee_rod_joint",
    "left_ankle_u_crank_joint",
    "left_ankle_b_crank_joint",
    "right_knee_crank_joint",
    "right_knee_rod_joint",
    "right_ankle_u_crank_joint",
    "right_ankle_b_crank_joint",
)

_OPEN_MJCF_INIT_JOINT_REMAP = {
    ".*_wrist_crank_joint": ".*_wrist_pitch_joint",
}


def sberchel_xr_teleop_robot_cfg():
    """Sberchel preset for XR teleop: bent elbows, otherwise same as ``SBERCHEL_CONFIG``."""
    from isaaclab.assets import ArticulationCfg

    base = SBERCHEL_CONFIG
    joint_pos = dict(base.init_state.joint_pos)
    joint_pos[".*_elbow_pitch_joint"] = -1.0

    return base.replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=base.init_state.pos,
            rot=base.init_state.rot,
            joint_pos=joint_pos,
            joint_vel=base.init_state.joint_vel,
        ),
    )


def _joint_dict_for_open_mjcf(joint_values: dict | None) -> dict:
    """Drop closed-only joints and rename wrist_crank → wrist_pitch in init state."""
    if not joint_values:
        return {}
    out: dict = {}
    drop = set(_OPEN_MJCF_INIT_JOINT_DROP)
    for key, value in joint_values.items():
        if key in drop:
            continue
        out[_OPEN_MJCF_INIT_JOINT_REMAP.get(key, key)] = value
    return out


def sberchel_upperbody_xr_robot_cfg():
    """Fixed-base Sberchel for upper-body-only XR teleop (no locomotion policy).

    Uses the **open** MJCF (serial wrists/legs, no crank/rod equality loops) and
    **ImplicitActuator** PD (PhysX drives). FrictionalActuator is explicit and leaves
    PhysX stiffness at 0, which is a poor fit for Pink position tracking; it also
    fought finger mimic joints when intermediate DOFs were in the motor group.
    """
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    cfg = sberchel_xr_teleop_robot_cfg()

    # Serial open model — matches Pink URDF EEF chain (forearm → pitch → roll → hand).
    closed_mjcf = Path(cfg.spawn.asset_path)
    if not closed_mjcf.is_file():
        deps = Path(_GREEN_CHALLENGE_ROOT) / "assets" / "robot_v3_0a" / "dependencies"
        closed_mjcf = (deps / cfg.spawn.asset_path).resolve()
    open_mjcf = closed_mjcf.with_name("v3_0_alpha_locomanipulator_open.xml")
    if not open_mjcf.is_file():
        raise FileNotFoundError(f"Open Sberchel MJCF missing: {open_mjcf}")
    cfg.spawn.asset_path = str(open_mjcf)
    cfg.spawn.usd_file_name = f"{open_mjcf.stem}.usd"
    if hasattr(cfg.spawn, "usd_path"):
        cfg.spawn.usd_path = None

    if cfg.spawn.articulation_props is not None:
        cfg.spawn.articulation_props.fix_root_link = True
        cfg.spawn.articulation_props.enabled_self_collisions = False
        cfg.spawn.articulation_props.solver_position_iteration_count = 8
        cfg.spawn.articulation_props.solver_velocity_iteration_count = 4
    if hasattr(cfg.spawn, "fix_base"):
        cfg.spawn.fix_base = True
    if hasattr(cfg.spawn, "self_collision"):
        cfg.spawn.self_collision = False

    cfg.init_state = ArticulationCfg.InitialStateCfg(
        pos=cfg.init_state.pos,
        rot=cfg.init_state.rot,
        joint_pos=_joint_dict_for_open_mjcf(dict(cfg.init_state.joint_pos)),
        joint_vel=_joint_dict_for_open_mjcf(dict(cfg.init_state.joint_vel or {})),
    )

    # Implicit PhysX PD — actuate proximal finger masters only; MJCF equality
    # mimic constraints drive intermediate/distal slaves at runtime.
    cfg.actuators = {
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
    }

    return cfg


def apply_sberchel_xr_teleop_pd_gains(robot) -> None:
    """Raise arm/hand/torso PD gains after the locomotion policy restores policy defaults."""
    arm_joints = set(_SBERCHEL_XR_ARM_JOINT_NAMES)
    finger_joints = set(_SBERCHEL_XR_FINGER_JOINT_NAMES)
    locked_joints = set(_SBERCHEL_XR_LOCKED_JOINT_NAMES)

    for actuator in robot.actuators.values():
        for local_idx, joint_name in enumerate(actuator.joint_names):
            if joint_name in locked_joints:
                actuator.stiffness[:, local_idx] = _XR_LOCK_STIFFNESS
                actuator.damping[:, local_idx] = _XR_LOCK_DAMPING
            elif joint_name in arm_joints:
                actuator.stiffness[:, local_idx] = _XR_ARM_STIFFNESS
                actuator.damping[:, local_idx] = _XR_ARM_DAMPING
            elif joint_name in finger_joints:
                actuator.stiffness[:, local_idx] = _XR_FINGER_STIFFNESS
                actuator.damping[:, local_idx] = _XR_FINGER_DAMPING
