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

``SberchelXRFetchAnyItemEnvCfg`` adds OpenXR Pink IK teleop for the upper body
while keeping leg actions for the locomotion policy (see ``scripts/teleop_sberchel.py``).

``UpperbodySberchelXRFetchAnyItemEnvCfg`` pins the root and exposes only the 26D
Pink IK action so it can be driven by ``scripts/teleop_se3_agent.py`` like GR1T2.
"""

import re
import tempfile
from pathlib import Path

import torch
from pink.tasks import DampingTask, FrameTask

import isaaclab.envs.mdp as mdp
import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import ManusViveCfg, OpenXRDeviceCfg, XrCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from dsynth_isaac.devices.sberchel_retargeter import SberchelRetargeterCfg
from dsynth_isaac.tasks.manager_based.darkstore_base_cfg import DarkstoreBaseEnvCfg, DsEventCfg
from dsynth_isaac.tasks.manager_based.fetching.fetching_env_cfg import FetchAnyItemEnvCfg
from dsynth_isaac.robots.sberchel_cfg import (
    SBERCHEL_CONFIG,
    sberchel_upperbody_xr_robot_cfg,
    sberchel_xr_teleop_robot_cfg,
)
from dsynth_isaac.robots.sberchel_mjcf_usd import convert_sberchel_usd_to_urdf, resolve_sberchel_mjcf_usd
import dsynth_isaac.tasks.manager_based.mdp.ds_events as ds_events
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as pick_place_mdp

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
        params={"p": [0.65, 0.0, SBERCHEL_STAND_HEIGHT]},
    )

@configclass
class SberchelObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class SberchelBaseEnvCfg(DarkstoreBaseEnvCfg):
    observations: SberchelObservationsCfg = SberchelObservationsCfg()

    def __post_init__(self):
        # build the shared darkstore scene first
        super().__post_init__()

        # --- swap in the sberchel robot (base left FREE for the policy) ---
        self.scene.robot = SBERCHEL_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot_fingers_contact_prim_path = None
        self.scene.ee_frame = None

        # --- match green_challenge's control timing so the policy is stable ---
        # (its run.sh default: sim.dt = 1/150, decimation = 1, render_interval = 1)
        self.sim.dt = 1.0 / 150.0
        self.decimation = 1
        self.sim.render_interval = 1

        self.actions = SberchelActionsCfg()
        self.events = SberchelEventCfg()


# --- XR teleop preset (upper body via Pink IK, legs via locomotion policy) --------

# Pink hand action layout (12D) — proximal masters only. Open MJCF drives
# intermediate/distal via equality mimic constraints; commanding those slaves
# with PD fights the constraint and locks the finger chain (only thumb_yaw escapes).
#
# Order MUST match articulation DOF order: PinkInverseKinematicsAction calls
# ``find_joints(..., preserve_order=False)``, which returns joints in robot
# joint order (not this list's written order if they disagree). Mismatch swaps
# fingers across digits and can look like cross-hand coupling.
SBERCHEL_HAND_JOINT_NAMES = [
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
]

_LEFT_HAND_BASE_FRAME = "left_hand_base_link"
_RIGHT_HAND_BASE_FRAME = "right_hand_base_link"

_TELEOP_ARM_STIFFNESS = {
    ".*_shoulder_pitch_joint": 300.0,
    ".*_shoulder_roll_joint": 300.0,
    ".*_shoulder_yaw_joint": 200.0,
    ".*_elbow_pitch_joint": 200.0,
    ".*_elbow_yaw_joint": 200.0,
    ".*_wrist_crank_joint": 150.0,
    ".*_wrist_roll_joint": 150.0,
    "torso_yaw_joint": 200.0,
    "neck_yaw_joint": 80.0,
    "neck_pitch_joint": 80.0,
    ".*_thumb_proximal_yaw_joint": 80.0,
    ".*_thumb_proximal_pitch_joint": 80.0,
    ".*_index_proximal_joint": 40.0,
    ".*_middle_proximal_joint": 40.0,
    ".*_ring_proximal_joint": 40.0,
    ".*_pinky_proximal_joint": 40.0,
}

_TELEOP_ARM_DAMPING = {
    ".*_shoulder_pitch_joint": 30.0,
    ".*_shoulder_roll_joint": 30.0,
    ".*_shoulder_yaw_joint": 20.0,
    ".*_elbow_pitch_joint": 20.0,
    ".*_elbow_yaw_joint": 20.0,
    ".*_wrist_crank_joint": 15.0,
    ".*_wrist_roll_joint": 15.0,
    "torso_yaw_joint": 15.0,
    "neck_yaw_joint": 8.0,
    "neck_pitch_joint": 8.0,
    ".*_thumb_proximal_yaw_joint": 8.0,
    ".*_thumb_proximal_pitch_joint": 8.0,
    ".*_index_proximal_joint": 4.0,
    ".*_middle_proximal_joint": 4.0,
    ".*_ring_proximal_joint": 4.0,
    ".*_pinky_proximal_joint": 4.0,
}


def _strip_urdf_collision_geometry(urdf_path: Path, output_path: Path) -> Path:
    """Remove collision geometry from URDF so Pinocchio skips unreliable collision meshes."""
    with urdf_path.open() as file:
        content = file.read()
    content = re.sub(r"<collision>.*?</collision>\s*", "", content, flags=re.DOTALL)

    visual_pattern = re.compile(r"<visual>.*?</visual>\s*", flags=re.DOTALL)

    def _drop_collision_visuals(match: re.Match[str]) -> str:
        block = match.group(0)
        if "collisions" in block and ".obj" in block:
            return ""
        return block

    content = visual_pattern.sub(_drop_collision_visuals, content)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    return output_path


def _triangulate_obj_meshes(mesh_dir: Path) -> None:
    """Triangulate quad faces in exported OBJ meshes for Pinocchio compatibility."""
    for obj_path in mesh_dir.glob("*.obj"):
        triangulated_lines: list[str] = []
        for line in obj_path.read_text().splitlines():
            if line.startswith("f "):
                face_vertices = line.split()[1:]
                if len(face_vertices) == 4:
                    triangulated_lines.append(f"f {face_vertices[0]} {face_vertices[1]} {face_vertices[2]}")
                    triangulated_lines.append(f"f {face_vertices[0]} {face_vertices[2]} {face_vertices[3]}")
                    continue
            triangulated_lines.append(line)
        obj_path.write_text("\n".join(triangulated_lines) + "\n")


def _resolve_sberchel_spawn_usd_path(spawn_cfg) -> str:
    """Resolve a patched Sberchel MJCF USD path for Pink IK / URDF export."""
    usd_path, _ = resolve_sberchel_mjcf_usd(spawn_cfg)
    return usd_path


def _prepare_sberchel_pink_ik_urdf(usd_path: str | Path, force_conversion: bool = False) -> tuple[str, str, str]:
    """Convert the robot USD to cached URDF/meshes usable by Pink IK."""
    from dsynth_isaac.robots.sberchel_mjcf_usd import PINK_URDF_CACHE_VERSION

    usd_path = Path(usd_path)
    cache_dir = usd_path.parent / "pink_ik_cache"
    version_path = cache_dir / ".cache_version"
    raw_urdf_path = cache_dir / "urdf" / f"{usd_path.stem}.urdf"
    pink_urdf_path = cache_dir / "urdf" / f"{usd_path.stem}_pink.urdf"
    mesh_dir = cache_dir / "meshes"

    cache_is_current = version_path.exists() and version_path.read_text().strip() == PINK_URDF_CACHE_VERSION
    if pink_urdf_path.exists() and mesh_dir.exists() and cache_is_current and not force_conversion:
        return str(pink_urdf_path), str(mesh_dir), str(raw_urdf_path)

    if force_conversion or not cache_is_current or not raw_urdf_path.exists() or not mesh_dir.exists():
        raw_urdf_path, mesh_dir = convert_sberchel_usd_to_urdf(usd_path, cache_dir)
        raw_urdf_path = Path(raw_urdf_path)
        mesh_dir = Path(mesh_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        version_path.write_text(PINK_URDF_CACHE_VERSION)

    _triangulate_obj_meshes(mesh_dir)
    pink_urdf_path = _strip_urdf_collision_geometry(raw_urdf_path, pink_urdf_path)
    return str(pink_urdf_path), str(mesh_dir), str(raw_urdf_path)


def _apply_sberchel_teleop_actuator_gains(robot_cfg: ArticulationCfg) -> ArticulationCfg:
    """Raise arm/hand PD gains so Pink IK and XR teleop can move the upper body."""
    motors = robot_cfg.actuators["motors"]
    for pattern, value in _TELEOP_ARM_STIFFNESS.items():
        motors.stiffness[pattern] = value
    for pattern, value in _TELEOP_ARM_DAMPING.items():
        motors.damping[pattern] = value
    return robot_cfg


# @configclass
# class SberchelXRActionsCfg:
#     """Legs from the locomotion policy + upper body from Pink IK / XR teleop."""

#     legs_joint_pos = mdp.JointPositionActionCfg(
#         asset_name="robot",
#         joint_names=[
#             "left_hip_pitch_joint",
#             "left_hip_roll_joint",
#             "left_hip_yaw_joint",
#             "left_knee_crank_joint",
#             "left_ankle_u_crank_joint",
#             "left_ankle_b_crank_joint",
#             "right_hip_pitch_joint",
#             "right_hip_roll_joint",
#             "right_hip_yaw_joint",
#             "right_knee_crank_joint",
#             "right_ankle_u_crank_joint",
#             "right_ankle_b_crank_joint",
#         ],
#         scale=1.0,
#         preserve_order=True,
#     )

#     upper_body_ik = PinkInverseKinematicsActionCfg(
#         # Do not command wrist_crank on the closed MJCF: it is disconnected from the
#         # Pink EEF chain and fights the wrist equality constraint (idle shaking).
#         pink_controlled_joint_names=[
#             "left_shoulder_pitch_joint",
#             "left_shoulder_roll_joint",
#             "left_shoulder_yaw_joint",
#             "left_elbow_pitch_joint",
#             "left_elbow_yaw_joint",
#             "left_wrist_roll_joint",
#             "right_shoulder_pitch_joint",
#             "right_shoulder_roll_joint",
#             "right_shoulder_yaw_joint",
#             "right_elbow_pitch_joint",
#             "right_elbow_yaw_joint",
#             "right_wrist_roll_joint",
#         ],
#         hand_joint_names=SBERCHEL_HAND_JOINT_NAMES,
#         target_eef_link_names={
#             "left_wrist": _LEFT_HAND_BASE_FRAME,
#             "right_wrist": _RIGHT_HAND_BASE_FRAME,
#         },
#         asset_name="robot",
#         controller=PinkIKControllerCfg(
#             articulation_name="robot",
#             base_link_name="pelvis_link",
#             num_hand_joints=len(SBERCHEL_HAND_JOINT_NAMES),
#             show_ik_warnings=True,
#             fail_on_joint_limit_violation=False,
#             variable_input_tasks=[
#                 FrameTask(
#                     _LEFT_HAND_BASE_FRAME,
#                     position_cost=8.0,
#                     orientation_cost=2.0,
#                     lm_damping=10.0,
#                     gain=0.5,
#                 ),
#                 FrameTask(
#                     _RIGHT_HAND_BASE_FRAME,
#                     position_cost=8.0,
#                     orientation_cost=2.0,
#                     lm_damping=10.0,
#                     gain=0.5,
#                 ),
#                 DampingTask(cost=0.5),
#                 NullSpacePostureTask(
#                     cost=0.5,
#                     lm_damping=1.0,
#                     gain=0.3,
#                     controlled_frames=[_LEFT_HAND_BASE_FRAME, _RIGHT_HAND_BASE_FRAME],
#                     controlled_joints=[
#                         "left_shoulder_pitch_joint",
#                         "left_shoulder_roll_joint",
#                         "left_shoulder_yaw_joint",
#                         "right_shoulder_pitch_joint",
#                         "right_shoulder_roll_joint",
#                         "right_shoulder_yaw_joint",
#                     ],
#                 ),
#             ],
#             fixed_input_tasks=[],
#             xr_enabled=True,
#         ),
#         enable_gravity_compensation=True,
#     )


# @configclass
# class SberchelXRObservationsCfg:
#     """Minimal observations for Sberchel XR teleop."""

#     @configclass
#     class PolicyCfg(ObsGroup):
#         actions = ObsTerm(func=pick_place_mdp.last_action)

#         def __post_init__(self):
#             self.enable_corruption = False
#             self.concatenate_terms = False

#     policy: PolicyCfg = PolicyCfg()


# @configclass
# class SberchelXRFetchAnyItemEnvCfg(DarkstoreBaseEnvCfg):
#     """Sberchel fetching env with XR upper-body teleop and policy-controlled legs."""

#     actions = SberchelXRActionsCfg()
#     events = SberchelEventCfg()
#     observations: SberchelObservationsCfg = SberchelObservationsCfg()

#     # Static world-frame XR origin (same as GR1/H12 darkstore).
#     xr: XrCfg = XrCfg(
#         anchor_pos=(4.0, 5.0, 0.0),
#         anchor_rot=(0.0, 0.0, 1.0, 0.0),
#     )

#     NUM_OPENXR_HAND_JOINTS = 26
#     temp_urdf_dir = tempfile.gettempdir()

#     # [left_wrist(7), right_wrist(7), hand_joints(12)] — world-frame wxyz.
#     idle_action = torch.tensor(
#         [
#             3.90,
#             5.95,
#             1.15,
#             0.707,
#             0.0,
#             0.0,
#             0.707,
#             4.30,
#             5.95,
#             1.15,
#             0.707,
#             0.0,
#             0.0,
#             0.707,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#         ]
#     )

#     def __post_init__(self):
#         super().__post_init__()

#         self.scene.robot = _apply_sberchel_teleop_actuator_gains(
#             sberchel_xr_teleop_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
#         )
#         self.scene.robot.spawn.semantic_tags = [("class", "robot")]
#         self.scene.robot_fingers_contact_prim_path = (
#             "{ENV_REGEX_NS}/Robot/root/.*_(index|middle|ring|pinky|thumb).*"
#         )
#         self.scene.ee_frame = None

#         # Keep policy timing from the base Sberchel env.
#         self.sim.dt = 1.0 / 150.0
#         self.decimation = 1
#         self.sim.render_interval = 1
#         self.episode_length_s = 20.0

#         pink_urdf_path, pink_mesh_dir, raw_urdf_path = _prepare_sberchel_pink_ik_urdf(
#             _resolve_sberchel_spawn_usd_path(self.scene.robot.spawn)
#         )
#         self.actions.upper_body_ik.controller.urdf_path = pink_urdf_path
#         self.actions.upper_body_ik.controller.mesh_path = pink_mesh_dir

#         dex_cache_dir = str(Path(pink_urdf_path).parent.parent)
#         idle = self.idle_action.detach().cpu().numpy()
#         retargeter_kwargs = {
#             "full_robot_urdf_path": raw_urdf_path,
#             "dex_cache_dir": dex_cache_dir,
#             "left_idle_wrist_pose": tuple(idle[0:7].tolist()),
#             "right_idle_wrist_pose": tuple(idle[7:14].tolist()),
#         }

#         self.teleop_devices = DevicesCfg(
#             devices={
#                 "handtracking": OpenXRDeviceCfg(
#                     retargeters=[
#                         SberchelRetargeterCfg(
#                             enable_visualization=True,
#                             num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
#                             sim_device=self.sim.device,
#                             hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
#                             **retargeter_kwargs,
#                         ),
#                     ],
#                     sim_device=self.sim.device,
#                     xr_cfg=self.xr,
#                 ),
#                 "manusvive": ManusViveCfg(
#                     retargeters=[
#                         SberchelRetargeterCfg(
#                             enable_visualization=True,
#                             num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
#                             sim_device=self.sim.device,
#                             hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
#                             **retargeter_kwargs,
#                         ),
#                     ],
#                     sim_device=self.sim.device,
#                     xr_cfg=self.xr,
#                 ),
#             }
#         )


@configclass
class UpperbodySberchelXRActionsCfg:
    """Pink IK upper body only — ``[left_wrist(7), right_wrist(7), fingers(12)]`` (26D).

    Wrist orientation offsets (OpenXR → ``*_hand_base_link``) live in
    ``SberchelRetargeter``, not here.
    """

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
    """Fixed-base Sberchel with XR upper-body teleop for ``scripts/teleop_se3_agent.py``.

    Action space is only Pink IK (26D): ``[left_wrist(7), right_wrist(7), fingers(12)]``.
    Root is pinned; legs/torso/neck hold default pose via high PD (not in the action vector).
    """

    actions = UpperbodySberchelXRActionsCfg()
    events = SberchelEventCfg()
    observations: SberchelObservationsCfg = SberchelObservationsCfg()

    xr: XrCfg = XrCfg(
        anchor_pos=(4.0, 5.0, 0.0),
        anchor_rot=(0.0, 0.0, 1.0, 0.0),
    )

    NUM_OPENXR_HAND_JOINTS = 26
    temp_urdf_dir = tempfile.gettempdir()

    # [left_wrist(7), right_wrist(7), hand_joints(12)] world-frame wxyz near darkstore spawn.
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

        self.scene.robot = sberchel_upperbody_xr_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        # Easier Pink IK tracking (same trick as H12 Inspire / G1 teleop presets).
        if self.scene.robot.spawn.rigid_props is not None:
            self.scene.robot.spawn.rigid_props.disable_gravity = True
        self.scene.robot_fingers_contact_prim_path = (
            "{ENV_REGEX_NS}/Robot/root/.*_(index|middle|ring|pinky|thumb).*"
        )
        self.scene.ee_frame = None

        # Match GR1T2 darkstore teleop timing.
        self.decimation = 6
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 2

        pink_urdf_path, pink_mesh_dir, raw_urdf_path = _prepare_sberchel_pink_ik_urdf(
            _resolve_sberchel_spawn_usd_path(self.scene.robot.spawn)
        )
        self.actions.upper_body_ik.controller.urdf_path = pink_urdf_path
        self.actions.upper_body_ik.controller.mesh_path = pink_mesh_dir

        dex_cache_dir = str(Path(pink_urdf_path).parent.parent)
        idle = self.idle_action.detach().cpu().numpy()
        retargeter_kwargs = {
            "full_robot_urdf_path": raw_urdf_path,
            "dex_cache_dir": dex_cache_dir,
            "left_idle_wrist_pose": tuple(idle[0:7].tolist()),
            "right_idle_wrist_pose": tuple(idle[7:14].tolist()),
        }

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        SberchelRetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                            **retargeter_kwargs,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "manusvive": ManusViveCfg(
                    retargeters=[
                        SberchelRetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                            **retargeter_kwargs,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
