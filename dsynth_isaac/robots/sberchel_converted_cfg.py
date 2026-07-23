"""Sberchel v3.0a locomanipulator — USD spawn (converted from MJCF).

Additive alternative to ``SBERCHEL_CONFIG`` (MJCF + ConfigManager). Spawns from a
flattened, material-repaired USD under ``assets/robots/sberchel_converted/`` and
keeps green_challenge ``FrictionalActuatorCfg`` (motors / passive) from
``sberchel_v3_0_alpha_locomanipulation_full``.

Configs:
- ``SBERCHEL_CONVERTED_CONFIG`` — free-floating root (locomotion)
- ``SBERCHEL_CONVERTED_FIXED_BASE_CONFIG`` — pelvis fixed via ``fix_root_link``

Bake / refresh the USD with::

    /home/soshin/miniconda3/envs/env_isaaclab/bin/python scripts/bake_sberchel_usd.py --headless
"""

import os
import sys

from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

from dsynth_isaac import DSYNTH_ISAAC_PATH

# --- Make sibling green_challenge importable (FrictionalActuatorCfg) ----------
_DEFAULT_GC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..", "green_challenge"))
_GREEN_CHALLENGE_ROOT = os.environ.get("GREEN_CHALLENGE_PATH", _DEFAULT_GC)
_parent = os.path.dirname(os.path.abspath(_GREEN_CHALLENGE_ROOT))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from green_challenge.assets.robot_v3_0a.interface.actuators.actuator_cfg import (  # noqa: E402
    FrictionalActuatorCfg,
)

_USD_PATH = f"{DSYNTH_ISAAC_PATH}/assets/robots/sberchel_converted/v3_0_alpha_locomanipulator.usd"

SBERCHEL_CONVERTED_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            retain_accelerations=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-2.0, 0.0, 0.88),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={'left_hip_pitch_joint': -0.086,
            'left_hip_roll_joint': 0.017,
            'left_hip_yaw_joint': -0.044,
            'left_knee_crank_joint': 0.299,
            'left_knee_rod_joint': -0.299,
            'left_knee_pitch_joint': 0.299,
            'left_ankle_pitch_joint': -0.225,
            'left_ankle_roll_joint': -0.018,
            'left_ankle_u_crank_joint': -0.208,
            'left_ankle_b_crank_joint': -0.23,
            'right_hip_pitch_joint': -0.086,
            'right_hip_roll_joint': -0.017,
            'right_hip_yaw_joint': 0.044,
            'right_knee_crank_joint': 0.299,
            'right_knee_rod_joint': -0.299,
            'right_knee_pitch_joint': 0.299,
            'right_ankle_pitch_joint': -0.225,
            'right_ankle_roll_joint': 0.018,
            'right_ankle_u_crank_joint': -0.208,
            'right_ankle_b_crank_joint': -0.23,
            'torso_yaw_joint': 0.0,
            '.*_shoulder_pitch_joint': 0.0,
            '.*_shoulder_roll_joint': 0.0,
            '.*_shoulder_yaw_joint': 0.0,
            '.*_elbow_pitch_joint': 0.0,
            '.*_elbow_yaw_joint': 0.0,
            'neck_yaw_joint': 0.0,
            'neck_pitch_joint': 0.0,
            '.*_wrist_crank_joint': 0.0,
            '.*_wrist_roll_joint': 0.0,
            '.*_thumb_proximal_yaw_joint': -0.1,
            '.*_thumb_proximal_pitch_joint': 0.0,
            '.*_index_proximal_joint': 0.0,
            '.*_middle_proximal_joint': 0.0,
            '.*_ring_proximal_joint': 0.0,
            '.*_pinky_proximal_joint': 0.0},
            joint_vel={'.*': 0.0},
        ),
        actuators={
            "motors": FrictionalActuatorCfg(
                joint_names_expr=['.*_ankle_b_crank_joint',
                    '.*_ankle_u_crank_joint',
                    '.*_elbow_pitch_joint',
                    '.*_elbow_yaw_joint',
                    '.*_hip_pitch_joint',
                    '.*_hip_roll_joint',
                    '.*_hip_yaw_joint',
                    '.*_index_intermediate_joint',
                    '.*_index_proximal_joint',
                    '.*_knee_crank_joint',
                    '.*_middle_intermediate_joint',
                    '.*_middle_proximal_joint',
                    '.*_pinky_intermediate_joint',
                    '.*_pinky_proximal_joint',
                    '.*_ring_intermediate_joint',
                    '.*_ring_proximal_joint',
                    '.*_shoulder_pitch_joint',
                    '.*_shoulder_roll_joint',
                    '.*_shoulder_yaw_joint',
                    '.*_thumb_distal_joint',
                    '.*_thumb_intermediate_joint',
                    '.*_thumb_proximal_pitch_joint',
                    '.*_thumb_proximal_yaw_joint',
                    '.*_wrist_crank_joint',
                    '.*_wrist_roll_joint',
                    'neck_pitch_joint',
                    'neck_yaw_joint',
                    'torso_yaw_joint'],
                effort_limit={'.*_ankle_b_crank_joint': 90.0,
                    '.*_ankle_u_crank_joint': 90.0,
                    '.*_elbow_pitch_joint': 36.0,
                    '.*_elbow_yaw_joint': 36.0,
                    '.*_hip_pitch_joint': 380.0,
                    '.*_hip_roll_joint': 330.0,
                    '.*_hip_yaw_joint': 150.0,
                    '.*_index_intermediate_joint': 8.5,
                    '.*_index_proximal_joint': 8.5,
                    '.*_knee_crank_joint': 380.0,
                    '.*_middle_intermediate_joint': 8.5,
                    '.*_middle_proximal_joint': 8.5,
                    '.*_pinky_intermediate_joint': 8.5,
                    '.*_pinky_proximal_joint': 8.5,
                    '.*_ring_intermediate_joint': 8.5,
                    '.*_ring_proximal_joint': 8.5,
                    '.*_shoulder_pitch_joint': 90.0,
                    '.*_shoulder_roll_joint': 90.0,
                    '.*_shoulder_yaw_joint': 36.0,
                    '.*_thumb_distal_joint': 8.5,
                    '.*_thumb_intermediate_joint': 8.5,
                    '.*_thumb_proximal_pitch_joint': 8.5,
                    '.*_thumb_proximal_yaw_joint': 8.5,
                    '.*_wrist_crank_joint': 16.0,
                    '.*_wrist_roll_joint': 16.0,
                    'neck_pitch_joint': 36.0,
                    'neck_yaw_joint': 36.0,
                    'torso_yaw_joint': 150.0},
                velocity_limit={'.*_ankle_b_crank_joint': 145.0,
                    '.*_ankle_u_crank_joint': 145.0,
                    '.*_elbow_pitch_joint': 89.0,
                    '.*_elbow_yaw_joint': 89.0,
                    '.*_hip_pitch_joint': 130.0,
                    '.*_hip_roll_joint': 123.0,
                    '.*_hip_yaw_joint': 140.0,
                    '.*_index_intermediate_joint': 25,
                    '.*_index_proximal_joint': 25,
                    '.*_knee_crank_joint': 130.0,
                    '.*_middle_intermediate_joint': 25,
                    '.*_middle_proximal_joint': 25,
                    '.*_pinky_intermediate_joint': 25,
                    '.*_pinky_proximal_joint': 25,
                    '.*_ring_intermediate_joint': 25,
                    '.*_ring_proximal_joint': 25,
                    '.*_shoulder_pitch_joint': 145.0,
                    '.*_shoulder_roll_joint': 145.0,
                    '.*_shoulder_yaw_joint': 89.0,
                    '.*_thumb_distal_joint': 25,
                    '.*_thumb_intermediate_joint': 25,
                    '.*_thumb_proximal_pitch_joint': 25,
                    '.*_thumb_proximal_yaw_joint': 25,
                    '.*_wrist_crank_joint': 60.0,
                    '.*_wrist_roll_joint': 60.0,
                    'neck_pitch_joint': 89.0,
                    'neck_yaw_joint': 89.0,
                    'torso_yaw_joint': 140.0},
                stiffness={'.*_hip_roll_joint': 200.0,
                    '.*_hip_yaw_joint': 100.0,
                    '.*_hip_pitch_joint': 200.0,
                    '.*_knee_crank_joint': 350.0,
                    '.*_ankle_u_crank_joint': 70.0,
                    '.*_ankle_b_crank_joint': 70.0,
                    'torso_yaw_joint': 70.0,
                    '.*_shoulder_pitch_joint': 30.0,
                    '.*_shoulder_roll_joint': 30.0,
                    '.*_shoulder_yaw_joint': 20.0,
                    '.*_elbow_pitch_joint': 20.0,
                    '.*_elbow_yaw_joint': 20.0,
                    'neck_yaw_joint': 20.0,
                    'neck_pitch_joint': 20.0,
                    '.*_wrist_crank_joint': 40,
                    '.*_wrist_roll_joint': 40,
                    '.*_thumb_proximal_yaw_joint': 40,
                    '.*_thumb_proximal_pitch_joint': 40,
                    '.*_index_proximal_joint': 10,
                    '.*_middle_proximal_joint': 10,
                    '.*_ring_proximal_joint': 10,
                    '.*_pinky_proximal_joint': 10},
                damping={'.*_hip_roll_joint': 5.0,
                    '.*_hip_yaw_joint': 5.0,
                    '.*_hip_pitch_joint': 5.0,
                    '.*_knee_crank_joint': 5.0,
                    '.*_ankle_u_crank_joint': 4.0,
                    '.*_ankle_b_crank_joint': 4.0,
                    'torso_yaw_joint': 4.0,
                    '.*_shoulder_pitch_joint': 1.5,
                    '.*_shoulder_roll_joint': 1.5,
                    '.*_shoulder_yaw_joint': 1.0,
                    '.*_elbow_pitch_joint': 1.0,
                    '.*_elbow_yaw_joint': 1.0,
                    'neck_yaw_joint': 1.0,
                    'neck_pitch_joint': 1.0,
                    '.*_wrist_crank_joint': 1.0,
                    '.*_wrist_roll_joint': 1.0,
                    '.*_thumb_proximal_yaw_joint': 2.0,
                    '.*_thumb_proximal_pitch_joint': 2.0,
                    '.*_index_proximal_joint': 1.0,
                    '.*_middle_proximal_joint': 1.0,
                    '.*_ring_proximal_joint': 1.0,
                    '.*_pinky_proximal_joint': 1.0},
                armature={'.*_ankle_b_crank_joint': 0.043416,
                    '.*_ankle_u_crank_joint': 0.043416,
                    '.*_elbow_pitch_joint': 0.024624,
                    '.*_elbow_yaw_joint': 0.024624,
                    '.*_hip_pitch_joint': 0.155952,
                    '.*_hip_roll_joint': 0.32255999999999996,
                    '.*_hip_yaw_joint': 0.061488,
                    '.*_index_intermediate_joint': 0.005,
                    '.*_index_proximal_joint': 0.005,
                    '.*_knee_crank_joint': 0.155952,
                    '.*_middle_intermediate_joint': 0.005,
                    '.*_middle_proximal_joint': 0.005,
                    '.*_pinky_intermediate_joint': 0.005,
                    '.*_pinky_proximal_joint': 0.005,
                    '.*_ring_intermediate_joint': 0.005,
                    '.*_ring_proximal_joint': 0.005,
                    '.*_shoulder_pitch_joint': 0.043416,
                    '.*_shoulder_roll_joint': 0.043416,
                    '.*_shoulder_yaw_joint': 0.024624,
                    '.*_thumb_distal_joint': 0.005,
                    '.*_thumb_intermediate_joint': 0.005,
                    '.*_thumb_proximal_pitch_joint': 0.005,
                    '.*_thumb_proximal_yaw_joint': 0.005,
                    '.*_wrist_crank_joint': 0.005,
                    '.*_wrist_roll_joint': 0.005,
                    'neck_pitch_joint': 0.024624,
                    'neck_yaw_joint': 0.024624,
                    'torso_yaw_joint': 0.061488},
                 breakaway_friction={'.*_ankle_b_crank_joint': 0.876,
                    '.*_ankle_u_crank_joint': 0.876,
                    '.*_elbow_pitch_joint': 0.23,
                    '.*_elbow_yaw_joint': 0.23,
                    '.*_hip_pitch_joint': 1.481,
                    '.*_hip_roll_joint': 2.492,
                    '.*_hip_yaw_joint': 0.576,
                    '.*_index_intermediate_joint': 0.005,
                    '.*_index_proximal_joint': 0.005,
                    '.*_knee_crank_joint': 1.481,
                    '.*_middle_intermediate_joint': 0.005,
                    '.*_middle_proximal_joint': 0.005,
                    '.*_pinky_intermediate_joint': 0.005,
                    '.*_pinky_proximal_joint': 0.005,
                    '.*_ring_intermediate_joint': 0.005,
                    '.*_ring_proximal_joint': 0.005,
                    '.*_shoulder_pitch_joint': 0.876,
                    '.*_shoulder_roll_joint': 0.876,
                    '.*_shoulder_yaw_joint': 0.23,
                    '.*_thumb_distal_joint': 0.005,
                    '.*_thumb_intermediate_joint': 0.005,
                    '.*_thumb_proximal_pitch_joint': 0.005,
                    '.*_thumb_proximal_yaw_joint': 0.005,
                    '.*_wrist_crank_joint': 0.1227,
                    '.*_wrist_roll_joint': 0.1227,
                    'neck_pitch_joint': 0.23,
                    'neck_yaw_joint': 0.23,
                    'torso_yaw_joint': 0.576},
                coulomb_friction={'.*_ankle_b_crank_joint': 0.876,
                    '.*_ankle_u_crank_joint': 0.876,
                    '.*_elbow_pitch_joint': 0.23,
                    '.*_elbow_yaw_joint': 0.23,
                    '.*_hip_pitch_joint': 1.481,
                    '.*_hip_roll_joint': 2.492,
                    '.*_hip_yaw_joint': 0.576,
                    '.*_index_intermediate_joint': 0.005,
                    '.*_index_proximal_joint': 0.005,
                    '.*_knee_crank_joint': 1.481,
                    '.*_middle_intermediate_joint': 0.005,
                    '.*_middle_proximal_joint': 0.005,
                    '.*_pinky_intermediate_joint': 0.005,
                    '.*_pinky_proximal_joint': 0.005,
                    '.*_ring_intermediate_joint': 0.005,
                    '.*_ring_proximal_joint': 0.005,
                    '.*_shoulder_pitch_joint': 0.876,
                    '.*_shoulder_roll_joint': 0.876,
                    '.*_shoulder_yaw_joint': 0.23,
                    '.*_thumb_distal_joint': 0.005,
                    '.*_thumb_intermediate_joint': 0.005,
                    '.*_thumb_proximal_pitch_joint': 0.005,
                    '.*_thumb_proximal_yaw_joint': 0.005,
                    '.*_wrist_crank_joint': 0.1227,
                    '.*_wrist_roll_joint': 0.1227,
                    'neck_pitch_joint': 0.23,
                    'neck_yaw_joint': 0.23,
                    'torso_yaw_joint': 0.576},
                viscous_coeff={'.*_ankle_b_crank_joint': 0.028,
                    '.*_ankle_u_crank_joint': 0.028,
                    '.*_elbow_pitch_joint': 0.014,
                    '.*_elbow_yaw_joint': 0.014,
                    '.*_hip_pitch_joint': 0.338,
                    '.*_hip_roll_joint': 0.766,
                    '.*_hip_yaw_joint': 0.0069,
                    '.*_index_intermediate_joint': 0.08,
                    '.*_index_proximal_joint': 0.08,
                    '.*_knee_crank_joint': 0.338,
                    '.*_middle_intermediate_joint': 0.08,
                    '.*_middle_proximal_joint': 0.08,
                    '.*_pinky_intermediate_joint': 0.08,
                    '.*_pinky_proximal_joint': 0.08,
                    '.*_ring_intermediate_joint': 0.08,
                    '.*_ring_proximal_joint': 0.08,
                    '.*_shoulder_pitch_joint': 0.028,
                    '.*_shoulder_roll_joint': 0.028,
                    '.*_shoulder_yaw_joint': 0.014,
                    '.*_thumb_distal_joint': 0.08,
                    '.*_thumb_intermediate_joint': 0.08,
                    '.*_thumb_proximal_pitch_joint': 0.08,
                    '.*_thumb_proximal_yaw_joint': 0.08,
                    '.*_wrist_crank_joint': 0.2726,
                    '.*_wrist_roll_joint': 0.2726,
                    'neck_pitch_joint': 0.014,
                    'neck_yaw_joint': 0.014,
                    'torso_yaw_joint': 0.0069},
                breakaway_velocity=0.2,
                force_control={'.*_ankle_b_crank_joint': True,
                    '.*_ankle_u_crank_joint': True,
                    '.*_elbow_pitch_joint': True,
                    '.*_elbow_yaw_joint': True,
                    '.*_hip_pitch_joint': True,
                    '.*_hip_roll_joint': True,
                    '.*_hip_yaw_joint': True,
                    '.*_index_intermediate_joint': False,
                    '.*_index_proximal_joint': False,
                    '.*_knee_crank_joint': True,
                    '.*_middle_intermediate_joint': False,
                    '.*_middle_proximal_joint': False,
                    '.*_pinky_intermediate_joint': False,
                    '.*_pinky_proximal_joint': False,
                    '.*_ring_intermediate_joint': False,
                    '.*_ring_proximal_joint': False,
                    '.*_shoulder_pitch_joint': True,
                    '.*_shoulder_roll_joint': True,
                    '.*_shoulder_yaw_joint': True,
                    '.*_thumb_distal_joint': False,
                    '.*_thumb_intermediate_joint': False,
                    '.*_thumb_proximal_pitch_joint': False,
                    '.*_thumb_proximal_yaw_joint': False,
                    '.*_wrist_crank_joint': False,
                    '.*_wrist_roll_joint': False,
                    'neck_pitch_joint': True,
                    'neck_yaw_joint': True,
                    'torso_yaw_joint': True},
            ),
            "passive": FrictionalActuatorCfg(
                joint_names_expr=['.*_ankle_pitch_joint', '.*_ankle_roll_joint', '.*_knee_pitch_joint', '.*_wrist_pitch_joint'],
                velocity_limit={},
                stiffness={},
                damping={},
                armature={},
                breakaway_friction={},
                coulomb_friction={},
                viscous_coeff={},
                breakaway_velocity=0.2,
                force_control={},
            ),
        },
    soft_joint_pos_limit_factor=0.9,
)

# Fixed-pelvis variant: same USD + frictional actuators, root link welded in place.
SBERCHEL_CONVERTED_FIXED_BASE_CONFIG = SBERCHEL_CONVERTED_CONFIG.replace(
    spawn=SBERCHEL_CONVERTED_CONFIG.spawn.replace(
        articulation_props=SBERCHEL_CONVERTED_CONFIG.spawn.articulation_props.replace(
            fix_root_link=True,
        ),
    ),
)

# --- Open converted USD + XR teleop preset (additive; does not change closed) ---

_OPEN_USD_PATH = f"{DSYNTH_ISAAC_PATH}/assets/robots/sberchel_converted/v3_0_alpha_locomanipulator_open.usd"

_XR_ARM_STIFFNESS = 3000.0
_XR_ARM_DAMPING = 100.0
_XR_FINGER_STIFFNESS = 400.0
_XR_FINGER_DAMPING = 40.0
_XR_LOCK_STIFFNESS = 10000.0
_XR_LOCK_DAMPING = 1000.0

# Closed-only init keys absent from the open model.
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


def _joint_dict_for_open_usd(joint_values: dict | None) -> dict:
    """Drop closed-only joints and rename wrist_crank → wrist_pitch."""
    if not joint_values:
        return {}
    drop = set(_OPEN_INIT_JOINT_DROP)
    out: dict = {}
    for key, value in joint_values.items():
        if key in drop:
            continue
        out[_OPEN_INIT_JOINT_REMAP.get(key, key)] = value
    return out


def sberchel_converted_upperbody_xr_robot_cfg() -> ArticulationCfg:
    """Fixed-pelvis open converted USD for upper-body XR teleop.

    Mirrors ``sberchel_upperbody_xr_robot_cfg`` (open kinematics + ImplicitActuator
    + proximal finger masters only) but spawns the baked open USD instead of MJCF.
    """
    from isaaclab.actuators import ImplicitActuatorCfg

    if not os.path.isfile(_OPEN_USD_PATH):
        raise FileNotFoundError(
            f"Open converted Sberchel USD missing: {_OPEN_USD_PATH}\n"
            "Bake it with:\n"
            "  /home/soshin/miniconda3/envs/env_isaaclab/bin/python "
            "scripts/bake_sberchel_usd.py --open --headless"
        )

    base = SBERCHEL_CONVERTED_CONFIG
    joint_pos = _joint_dict_for_open_usd(dict(base.init_state.joint_pos))
    joint_pos[".*_elbow_pitch_joint"] = -1.0
    joint_vel = _joint_dict_for_open_usd(dict(base.init_state.joint_vel or {".*": 0.0}))

    spawn = base.spawn.replace(
        usd_path=_OPEN_USD_PATH,
        rigid_props=base.spawn.rigid_props.replace(disable_gravity=True)
        if base.spawn.rigid_props is not None
        else None,
        articulation_props=base.spawn.articulation_props.replace(
            fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        )
        if base.spawn.articulation_props is not None
        else None,
    )

    return ArticulationCfg(
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=base.init_state.pos,
            rot=base.init_state.rot,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
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
        soft_joint_pos_limit_factor=0.9,
    )

