from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

from dsynth_isaac import DSYNTH_ISAAC_PATH

# https://github.com/jih189/isaaclab_fetch_project/blob/main/exts/fetch_project/fetch_project/robots/fetch.py
WORKER_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"/home/kvsoshin/Work/fetch-isaac-sim/assets/fetch/fetch_new.usd",
        usd_path=f"{DSYNTH_ISAAC_PATH}/assets/robots/worker/worker.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # body
            "torso_drive_joint": 0.0,
            # left arm
            "arx5_left_joint1": 0.0,
            "arx5_left_joint2": 0.0,
            "arx5_left_joint3": 0.0,
            "arx5_left_joint4": 0.0,
            "arx5_left_joint5": 0.0,
            "arx5_left_joint6": 0.0,
            "arx5_left_gripper_joint": 0.0,
            # right arm
            "arx5_right_joint1": 0.0,
            "arx5_right_joint2": 0.0,
            "arx5_right_joint3": 0.0,
            "arx5_right_joint4": 0.0,
            "arx5_right_joint5": 0.0,
            "arx5_right_joint6": 0.0,
            "arx5_right_gripper_joint": 0.0,
            # femto holder
            "femto_holder_drive_joint": 0.0,
        },
    ),
    actuators={
        "torso_drive_joint": ImplicitActuatorCfg(
            joint_names_expr=["torso_drive_joint"],
            effort_limit=10000.0,
            velocity_limit=0.1,
            stiffness=100000.0,
            damping=100000.0,
        ),
        "arx5_left_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "arx5_left_joint.*",
            ],
            effort_limit=10000.0,
            velocity_limit=1.0,
            stiffness=100000.0,
            damping=100.0,
        ),
        "arx5_left_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "arx5_left_gripper_joint",
            ],
            effort_limit=10000.0,
            velocity_limit=1.0,
            stiffness=100000.0,
            damping=100.0,
        ),
        "arx5_right_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "arx5_right_joint.*",
            ],
            effort_limit=10000.0,
            velocity_limit=1.0,
            stiffness=100000.0,
            damping=100.0,
        ),
        "arx5_left_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "arx5_right_gripper_joint",
            ],
            effort_limit=10000.0,
            velocity_limit=1.0,
            stiffness=100000.0,
            damping=100.0,
        ),
        "camera_holder": ImplicitActuatorCfg(
            joint_names_expr=["femto_holder_drive_joint"],
            effort_limit=1.0,
            velocity_limit=1.57,
            stiffness=100,
            damping=10,
        ),
        
    },
    soft_joint_pos_limit_factor=1.0,
)