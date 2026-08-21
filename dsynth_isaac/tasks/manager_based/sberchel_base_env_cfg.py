"""Darkstore environment with the Sberchel v3.0a humanoid as the robot.

Same store/scene as the Fetch task (``DarkstoreBaseEnvCfg``); only the robot in
``self.scene.robot`` is swapped out.

Sberchel is a free-floating biped wired for the green_challenge locomotion policy
(see ``scripts/demo_sberchel.py``). Upper-body XR teleop lives in
``sberchel_upperbody_xr_env_cfg.UpperbodySberchelXRFetchAnyItemEnvCfg``.
Full-body XR + policy legs: ``sberchel_xr_env_cfg.SberchelXRFetchAnyItemEnvCfg``.
"""

from typing import TYPE_CHECKING, Literal

import isaaclab.envs.mdp as mdp
import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab.envs import ManagerBasedRLEnvCfg

from dsynth_isaac.tasks.manager_based.darkstore_base_cfg import DarkstoreBaseEnvCfg, DsEventCfg
from dsynth_isaac.robots.sberchel_cfg import SBERCHEL_CONFIG
from dsynth_isaac.robots.sberchel_camera_cfg import make_sberchel_left_head_camera_cfg
import dsynth_isaac.tasks.manager_based.mdp.ds_events as ds_events
from dsynth_isaac.scene.sberdemo_scene import SberDemoSceneCfg
from dsynth_isaac.tasks.manager_based.darkstore_base_cfg import TerminationsCfg

if TYPE_CHECKING:  # green_challenge is resolved at runtime, not at import time
    from green_challenge.policy.network import Network

# Root (pelvis) height the robot is placed at so its feet start on the floor.
SBERCHEL_STAND_HEIGHT = 0.9


def apply_green_challenge_physx(cfg) -> None:
    """Mirror the PhysX scene settings green_challenge evaluates on.

    From their ``envs/env_registration.make_env``. Only the settings that are
    real IsaacLab fields are copied -- their ``num_position_iterations``,
    ``num_velocity_iterations``, ``sleep_threshold``, ``stabilization_threshold``,
    ``num_threads``, ``contact_collection_mode``, ``use_gpu`` and ``use_flatcache``
    assignments name attributes that do not exist on IsaacLab 2.3's cfg classes,
    so they are silently inert on their side too.

    ``solver_type`` is the one that matters and is easy to miss. The robot's
    eight loop closures are solved as maximal-coordinate constraints, and PGS
    holds them at the 4/4 iteration counts the robot config ships with, while
    IsaacLab's default TGS does not: on TGS the loops stay compliant and the zest
    policy drops into a ~1-2 Hz limit cycle, so the robot shuffles instead of
    standing. Measured base path standing at zero command for 15 s: 0.044 m on
    PGS 4/4, 1.66 m on TGS 4/4.

    Not mirrored: ``sim.device = "cpu"``. green_challenge runs physics on CPU;
    we run on GPU, which is a real remaining difference in the plant.
    """
    cfg.sim.physx.solver_type = 0  # PGS, as green_challenge selects
    cfg.sim.physx.enable_stabilization = True
    cfg.sim.physx.bounce_threshold_velocity = 0.5
    cfg.sim.render.enable_translucency = True


@configclass
class SberchelActionsCfg:
    """Position actions, mirroring green_challenge's ``ExampleActionsPosCfg``.

    Order of the concatenated action vector (41 dims):
      legs(12) + torso(13) + wrist(4) + fingers(12).
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
        mode="reset",
        params={"p": [0.65, 0.0, SBERCHEL_STAND_HEIGHT]},
    )

    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
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

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()


@configclass
class SberchelBaseEnvCfg(DarkstoreBaseEnvCfg):
    observations: SberchelObservationsCfg = SberchelObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SBERCHEL_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot_fingers_contact_prim_path = None
        self.scene.ee_frame = None

        self.sim.dt = 1.0 / 150.0
        self.decimation = 1
        self.sim.render_interval = 1
        apply_green_challenge_physx(self)

        self.actions = SberchelActionsCfg()
        self.events = SberchelEventCfg()

# Policy-native leg joint order (matches green_challenge ``model.joint_order``
# legs slice). These are the crank motor dofs of the closed leg linkages.
POLICY_LEG_JOINT_NAMES = [
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
]


LEGS_ACTION_DIM = len(POLICY_LEG_JOINT_NAMES)

# Neck joints for XR headset look (MJCF ranges: yaw ±0.785, pitch ±0.524).
NECK_JOINT_NAMES = [
    "neck_yaw_joint",
    "neck_pitch_joint",
]
NECK_ACTION_DIM = len(NECK_JOINT_NAMES)
NECK_YAW_LIMIT = 0.785
NECK_PITCH_LIMIT = 0.524


# green_challenge ``model.joint_order`` past the legs: torso_yaw, both arms, neck.
# ``Network.__call__`` returns all 25 in this order, so the arm block is a plain slice.
POLICY_TORSO_MODEL_INDEX = LEGS_ACTION_DIM
POLICY_MODEL_ARM_SLICE = slice(13, 23)
POLICY_ARM_JOINT_NAMES = [
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
]
ARM_ACTION_DIM = len(POLICY_ARM_JOINT_NAMES)


def policy_full_targets(net, vx: float, vy: float, wz: float) -> list[float]:
    """Run ``net``; return all 25 joint targets in green_challenge model order.

    Values have the USD default pose already subtracted, because the leg/neck
    ``JointPositionActionCfg`` terms re-add it (``use_default_offset=True``).
    Anything writing *absolute* targets (``PinkJointHold``) must add it back.

    Both policy types are driven through here on purpose. ``set_command_detailed``
    does not filter by the commander's ``cmd_names`` -- it works for either type
    only because ``lin_vel_x/y`` and ``ang_vel_z`` exist in both. The
    ``set_joint_overrides`` call is what zest needs to hold its upper-body
    reference at zero; the walk policy's commander advertises only
    ``lin_vel_x/y``, ``ang_vel_z`` and ``height_offset``, so every key below is
    filtered out and the call is a no-op for it.

    Never share an observation between two ``Network`` instances: the policy
    joint order differs per type, and the controllers mutate the obs dict in
    place (tensor coercion, default subtraction, zest's motor->joint mapping).
    ``net(net.obs())`` keeps each instance self-contained.
    """
    net.set_command_detailed(vx, vy, wz)
    net.set_joint_overrides({
        "torso_joint_pos": [0.0] * 13,
        "wrist_joint_pos": [0.0] * 4,
        "finger_joint_pos": [0.0] * 12,
    })
    loco = net(net.obs())
    return list(loco["joint_pos"])


def policy_leg_targets(net, vx: float, vy: float, wz: float) -> list[float]:
    """Run the locomotion policy; return 12 leg targets in ``POLICY_LEG_JOINT_NAMES`` order."""
    return policy_full_targets(net, vx, vy, wz)[:LEGS_ACTION_DIM]


def apply_policy_pd_gains(net) -> None:
    """Write ``net``'s policy PD gains into the articulation.

    This is the plant, not a tweak. green_challenge runs with
    ``--deactivate_torque``, so every joint is position-controlled and the gains
    that matter are the ones sitting in ``actuator.stiffness`` -- which
    ``FrictionalActuator.__init__`` zeroed for the 25 force-controlled body
    joints and which ``Network.__init__`` then fills in from the loaded policy's
    ``control`` block. Swapping policy mid-session swaps those gains on their
    side too, so re-assert them on every swap.
    """
    restore = getattr(net, "_restore_pd_gains_from_policy_cfg", None)
    if restore is None:
        raise RuntimeError(
            "green_challenge Network no longer exposes "
            "_restore_pd_gains_from_policy_cfg; without it the force-controlled "
            "body joints keep the zeroed gains and the robot goes limp."
        )
    restore(net.env, net.cfg)


@configclass
class SberchelFetchAnyItemEnvCfg(DarkstoreBaseEnvCfg):
    actions = SberchelActionsCfg()
    events = SberchelEventCfg()
    observations: SberchelObservationsCfg = SberchelObservationsCfg()

    target_item_name_regex: str = ".*"
    language_instruction: str = None

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SBERCHEL_CONFIG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot_fingers_contact_prim_path = None
        self.scene.ee_frame = None

        self.sim.dt = 1.0 / 150.0
        self.decimation = 1
        self.sim.render_interval = 1
        apply_green_challenge_physx(self)

@configclass
class SberchelDemoEnvCfg(ManagerBasedRLEnvCfg):
    scene: SberDemoSceneCfg = SberDemoSceneCfg(replicate_physics=False, env_spacing=20.5, filter_collisions=False)
    actions = SberchelActionsCfg()
    observations: SberchelObservationsCfg = SberchelObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    events: None

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SBERCHEL_CONFIG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot.init_state.pos = (5.5, 5, SBERCHEL_STAND_HEIGHT)
        self.scene.left_head_camera = make_sberchel_left_head_camera_cfg()
        # Raw RGB for HDF5 demo video (default ``normalize=True`` mean-centers floats).
        self.observations.rgb_camera.head_img = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("left_head_camera"), "normalize": False},
        )
        # Pin pelvis_link (articulation root) in world space.
        # if self.scene.robot.spawn.articulation_props is not None:
        #     self.scene.robot.spawn.articulation_props = (
        #         self.scene.robot.spawn.articulation_props.replace(fix_root_link=True)
        #     )
        # general settings
        self.decimation = 2
        self.episode_length_s = 100000.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2
        apply_green_challenge_physx(self)


# Which half of the C toggle a step belongs to. Recorded per step as
# ``teleop_state[:, 0]`` (1.0 = locomotion), which is what lets replay pick the
# same policy the recorder used.
LocoMode = Literal["locomotion", "static"]


class LocoPolicySet:
    """The locomotion policies for a session, plus the PD regime each one owns.

    Two things make a runtime policy swap more than "call the other net":

    * ``Network.__init__`` writes the policy's PD gains into the articulation and
      ``Network.reset()`` does not, so constructing a second ``Network`` silently
      leaves *its* leg gains resident. Every swap has to re-assert them.
    * Those gains are the plant. Both groups are ``FrictionalActuator``s whose
      ``compute()`` reads ``stiffness`` every step, and the cfg values for the 25
      body joints were zeroed at init because they are force-controlled -- so the
      policy's table is the only thing holding the robot up. green_challenge
      behaves the same way, which is why a swap re-asserts rather than pins.

    Holding both nets and the swap rule in one object means there is exactly one
    binding to thread through the script, and one place that defines what replay
    has to reproduce.
    """

    def __init__(self, env, mode_map: dict[LocoMode, str], *,
                 decimation: dict[str, int] | None = None, device: str = "cpu") -> None:
        # Imported here, not at module scope: this module is imported to build env
        # configs in contexts where the green_challenge checkout is not on the path.
        from green_challenge.policy.network import Network

        self._mode_map = dict(mode_map)
        self._decimation = dict(decimation or {})
        # dict.fromkeys keeps construction order deterministic, and builds one
        # net per distinct type: the single-policy case constructs exactly one.
        self._nets: dict[str, "Network"] = {
            policy_type: Network(env, type=policy_type, device=device, closed_kinematics=True)
            for policy_type in dict.fromkeys(self._mode_map.values())
        }
        self._active_type: str | None = None
        self._phase: dict[str, int] = {t: 0 for t in self._nets}
        self._held: dict[str, list[float] | None] = {t: None for t in self._nets}
        # Each Network constructor wrote its own policy's gains, so the last one
        # built is the one left resident; activate() puts the right one back.

    @property
    def multi_policy(self) -> bool:
        return len(self._nets) > 1

    @property
    def active(self) -> "Network":
        if self._active_type is None:
            raise RuntimeError("LocoPolicySet.activate() has not been called yet")
        return self._nets[self._active_type]

    @property
    def active_type(self) -> str | None:
        return self._active_type

    def type_for(self, mode: LocoMode) -> str:
        return self._mode_map[mode]

    def net_for(self, mode: LocoMode) -> "Network":
        return self._nets[self._mode_map[mode]]

    def activate(self, mode: LocoMode) -> "Network":
        """Make ``mode``'s policy the live one, PD gains included.

        Compares net *identity*, not mode: when both modes map to the same policy
        (the default) every C press must be a total no-op -- no reset, no
        discontinuity -- so the single-policy path stays exactly what it was
        before per-mode policies existed.
        """
        policy_type = self._mode_map[mode]
        net = self._nets[policy_type]
        if policy_type == self._active_type:
            return net
        # This policy's gains *are* the plant, and the other net's constructor may
        # have left its own resident, so re-assert on every swap.
        apply_policy_pd_gains(net)
        # Neither policy expects to resume mid-segment: zest's previous action
        # and walk's 15-frame observation history would both be a segment stale.
        # Reset is what each gets at episode start anyway, and unlike keeping the
        # idle net warm it costs nothing per step and is trivially reproducible.
        net.reset()
        self._phase[policy_type] = 0
        self._held[policy_type] = None
        self._active_type = policy_type
        return net

    def reassert(self, mode: LocoMode) -> "Network":
        """Run the full activate sequence even if the net is already active."""
        self._active_type = None
        return self.activate(mode)

    def reset_all(self) -> None:
        """Reset every net (after ``env.reset()``); gains are untouched.

        Gains survive ``env.reset()`` -- ``FrictionalActuator.reset`` is a no-op --
        so nothing re-asserts them for us and nothing undoes them either.
        """
        for policy_type, net in self._nets.items():
            net.reset()
            self._phase[policy_type] = 0
            self._held[policy_type] = None

    def step(self, mode: LocoMode, vx: float, vy: float, wz: float) -> list[float]:
        """All 25 joint targets from ``mode``'s policy, honouring its decimation.

        Call once per *recorded* step. The phase counter advances only on those
        steps and is zeroed on every activation, so replay reconstructs it from
        the recorded mode transitions alone -- nothing extra has to be stored.
        """
        net = self.activate(mode)
        policy_type = self._mode_map[mode]
        decim = max(1, int(self._decimation.get(policy_type, 1)))
        held = self._held[policy_type]
        if decim > 1 and held is not None and (self._phase[policy_type] % decim) != 0:
            self._phase[policy_type] += 1
            return held
        targets = policy_full_targets(net, vx, vy, wz)
        self._held[policy_type] = targets
        self._phase[policy_type] += 1
        return targets

    def capture(self) -> dict:
        """Snapshot every net's cross-step state for a checkpoint.

        zest's only carry-over is ``controller.actions`` (it runs a single
        observation frame), but walk keeps a 15-frame history in ``obs_buf``:
        restoring one without the other leaves the policy on a timeline the
        scene no longer matches, and the rollback stops being bit-exact.
        """
        snapshot = {"active_type": self._active_type, "nets": {}}
        for policy_type, net in self._nets.items():
            controller = net.controller
            obs_buf = getattr(controller.observation_packer, "obs_buf", None)
            snapshot["nets"][policy_type] = {
                "actions": controller.actions.clone(),
                "obs_buf": None if obs_buf is None else obs_buf.clone(),
                "commands": controller.commander._commands.clone(),
                "phase": self._phase[policy_type],
                "held": None if self._held[policy_type] is None else list(self._held[policy_type]),
            }
        return snapshot

    def restore(self, snapshot: dict) -> None:
        """Roll every net back to ``capture``'s instant.

        Must run *after* whichever ``enter_*_mode`` the restore lands in, or that
        mode's ``activate`` would reset away what was just restored.
        """
        for policy_type, state in snapshot["nets"].items():
            net = self._nets.get(policy_type)
            if net is None:
                continue
            controller = net.controller
            controller.actions.copy_(state["actions"])
            controller.commander._commands.copy_(state["commands"])
            saved_obs = state["obs_buf"]
            if saved_obs is None:
                controller.observation_packer.obs_buf = None
            else:
                controller.observation_packer.obs_buf = saved_obs.clone()
            self._phase[policy_type] = state["phase"]
            self._held[policy_type] = None if state["held"] is None else list(state["held"])

    def weights_by_type(self) -> dict[str, str]:
        """Each policy's weights directory name, for provenance.

        The checkpoint that produced a dataset is the one fact a reader cannot
        recover from anything else, so record the directory rather than just the
        type alias that points at it.
        """
        from green_challenge.policy.network import _POLICY_NAME_BY_TYPE

        return {t: _POLICY_NAME_BY_TYPE.get(t, t) for t in self._nets}

    def command_ranges(self, mode: LocoMode) -> dict[str, list[float]]:
        """Velocity clip ranges the policy for ``mode`` applies to commands."""
        commander = self.net_for(mode).controller.commander
        ranges = commander.cfg.get("ranges", {})
        return {
            name: [float(v) for v in ranges[name]]
            for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
            if name in ranges
        }
