"""Darkstore environment with the Sberchel v3.0a humanoid as the robot.

Same store/scene as the Fetch task (``DarkstoreBaseEnvCfg``); only the robot in
``self.scene.robot`` is swapped out.

Sberchel is a free-floating biped. Instead of pinning it, this env is wired so
the ONNX RL locomotion policy from green_challenge can balance it and keep it
standing (see ``scripts/demo_sberchel.py``). To make the policy behave the same
way it does in green_challenge's own ``run.sh``, three things are replicated:

  1. the exact 4-group position-action layout green_challenge drives
     (``ExampleActionsPosCfg``): legs(12) + torso(13) + wrist(4) + fingers(12),
     in that order, with the same per-group ``use_default_offset`` flags;
  2. the same robot config (reused verbatim, so default joint poses / actuator
     friction match what the policy expects);
  3. the same control timing (sim.dt = 1/150, decimation = 1).

The base is left free by default so the policy actually carries the weight. The
demo can still pin it (``--pin``) for a no-policy sanity check.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from dsynth_isaac.tasks.manager_based.darkstore_base_cfg import DarkstoreBaseEnvCfg, DsEventCfg
from dsynth_isaac.robots.sberchel_cfg import SBERCHEL_CONFIG
import dsynth_isaac.tasks.manager_based.mdp.ds_events as ds_events

# Root (pelvis) height the robot is placed at so its feet start on the floor.
SBERCHEL_STAND_HEIGHT = 0.9


@configclass
class SberchelActionsCfg:
    """Position actions, mirroring green_challenge's ``ExampleActionsPosCfg``.

    Order of the concatenated action vector (41 dims):
      legs(12) + torso(13) + wrist(4) + fingers(12).
    ``Network`` produces leg targets in exactly this leg order; the other groups
    are held at default (zero action) for plain standing.

    Offsets match green_challenge exactly: legs/torso send absolute-style targets
    (no default offset — ``Network`` already removes the default), while
    wrist/fingers use the default offset so a zero action holds their rest pose.
    """

    legs_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_crank_joint",
            "left_ankle_u_crank_joint",
            "left_ankle_b_crank_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_crank_joint",
            "right_ankle_u_crank_joint",
            "right_ankle_b_crank_joint",
        ],
        scale=1.0,
        preserve_order=True,
    )

    torso_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "torso_yaw_joint",
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
            "neck_yaw_joint",
            "neck_pitch_joint",
        ],
        scale=1.0,
        preserve_order=True,
    )

    wrist_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_wrist_crank_joint",
            "left_wrist_roll_joint",
            "right_wrist_crank_joint",
            "right_wrist_roll_joint",
        ],
        scale=1.0,
        use_default_offset=True,
        preserve_order=True,
    )

    finger_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_pinky_proximal_joint",
            "left_ring_proximal_joint",
            "left_middle_proximal_joint",
            "left_index_proximal_joint",
            "left_thumb_proximal_pitch_joint",
            "left_thumb_proximal_yaw_joint",
            "right_pinky_proximal_joint",
            "right_ring_proximal_joint",
            "right_middle_proximal_joint",
            "right_index_proximal_joint",
            "right_thumb_proximal_pitch_joint",
            "right_thumb_proximal_yaw_joint",
        ],
        scale=1.0,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class SberchelEventCfg(DsEventCfg):
    """Reuse the store's events, but lift the base to standing height."""

    init_robot_base_pose = EventTerm(
        func=ds_events.set_default_base_pose,
        mode="startup",
        params={"p": [0.0, 0.0, SBERCHEL_STAND_HEIGHT]},
    )


@configclass
class SberchelBaseEnvCfg(DarkstoreBaseEnvCfg):
    def __post_init__(self):
        # build the shared darkstore scene first
        super().__post_init__()

        # --- swap in the sberchel robot (base left FREE for the policy) ---
        self.scene.robot = SBERCHEL_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # --- match green_challenge's control timing so the policy is stable ---
        # (its run.sh default: sim.dt = 1/150, decimation = 1, render_interval = 1)
        self.sim.dt = 1.0 / 150.0
        self.decimation = 1
        self.sim.render_interval = 1

        self.actions = SberchelActionsCfg()
        self.events = SberchelEventCfg()
