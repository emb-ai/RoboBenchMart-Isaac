"""Sberchel XR demo recorders for Isaac Lab RecorderManager."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.managers.recorder_manager import RecorderTermCfg
from isaaclab.utils import configclass

LOCO_COMMAND_ATTR = "_sberchel_loco_command"
TELEOP_STATE_ATTR = "_sberchel_teleop_state"


def stash_locomotion_command(env, vx: float, vy: float, wz: float) -> None:
    """Store per-env locomotion velocity commands for the next ``env.step`` recorder tick."""
    setattr(
        env,
        LOCO_COMMAND_ATTR,
        torch.tensor([[vx, vy, wz]], dtype=torch.float32, device=env.device).repeat(env.num_envs, 1),
    )


def stash_teleop_state(env, *, loco_mode: str, squeeze: float, legacy_gains: bool) -> None:
    """Store operator/plant state for the next ``env.step`` recorder tick.

    ``loco_mode`` and ``legacy_gains`` are what let a downstream consumer tell
    which plant produced a segment; ``squeeze`` is the grasp-intent channel that
    the finger targets alone do not disambiguate (a finger blocked by an object
    reads the same whether it is squeezing hard or merely resting).
    """
    setattr(
        env,
        TELEOP_STATE_ATTR,
        torch.tensor(
            [[1.0 if loco_mode == "locomotion" else 0.0, float(squeeze), float(legacy_gains)]],
            dtype=torch.float32,
            device=env.device,
        ).repeat(env.num_envs, 1),
    )


class LocomotionCommandRecorder(RecorderTerm):
    """Record ``(vx, vy, wz)`` stashed on the env before each step."""

    def record_pre_step(self):
        command = getattr(self._env, LOCO_COMMAND_ATTR, None)
        if command is None:
            command = torch.zeros(self._env.num_envs, 3, device=self._env.device)
        return "loco_commands", command


@configclass
class LocomotionCommandRecorderCfg(RecorderTermCfg):
    """Configuration for the locomotion command recorder term."""

    class_type: type[RecorderTerm] = LocomotionCommandRecorder


class PreStepRgbCameraObservationsRecorder(RecorderTerm):
    """Record ``observations.rgb_camera`` terms under flat ``obs/<term>`` keys.

    ``ActionStateRecorderManagerCfg`` only writes ``obs_buf["policy"]``. Sberchel puts
    left-head RGB under ``rgb_camera``, so demos need this term or HDF5 exports omit
    video even when ``--enable_cameras`` is set.

    Terms are flattened (``obs/head_img``, not ``obs/rgb_camera/head_img``) so
    Isaac Lab's ``hdf5_to_mp4.py --input_keys head_img`` works without nested paths.
    """

    def record_pre_step(self):
        rgb_obs = self._env.obs_buf.get("rgb_camera")
        if rgb_obs is None or (isinstance(rgb_obs, dict) and len(rgb_obs) == 0):
            return None, None
        # Flat under ``obs/`` (policy recorder also uses ``obs``; keys merge).
        return "obs", rgb_obs


@configclass
class PreStepRgbCameraObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the RGB camera observation recorder term."""

    class_type: type[RecorderTerm] = PreStepRgbCameraObservationsRecorder


class TeleopStateRecorder(RecorderTerm):
    """Record ``(is_locomotion, squeeze, legacy_gains)`` stashed before each step."""

    def record_pre_step(self):
        state = getattr(self._env, TELEOP_STATE_ATTR, None)
        if state is None:
            state = torch.zeros(self._env.num_envs, 3, device=self._env.device)
        return "teleop_state", state


@configclass
class TeleopStateRecorderCfg(RecorderTermCfg):
    """Configuration for the teleop state recorder term."""

    class_type: type[RecorderTerm] = TeleopStateRecorder


class PostStepJointTargetsRecorder(RecorderTerm):
    """Record full-articulation commanded and achieved joint positions.

    ``processed_actions`` only covers joints owned by an action term, and the
    upper-body term reports Cartesian-derived IK output. green_challenge's VLA
    interface is joint-space throughout (``legs_joint_pos`` / ``torso_joint_pos``
    / ``finger_joint_pos``), so recording the raw articulation vectors -- plus
    the joint-name order in the episode metadata -- is what makes conversion to
    their layout mechanical rather than guesswork.

    Both are needed: the *target* carries grip intent (a blocked finger's
    achieved angle says nothing about how hard it is squeezing), while the
    *achieved* value is the plant-independent physical fact that survives
    relabelling onto a different controller.
    """

    def record_post_step(self):
        robot = self._env.scene["robot"]
        return "joint_targets", {
            "joint_pos_target": robot.data.joint_pos_target.clone(),
            "joint_pos": robot.data.joint_pos.clone(),
            "joint_vel": robot.data.joint_vel.clone(),
        }


@configclass
class PostStepJointTargetsRecorderCfg(RecorderTermCfg):
    """Configuration for the joint target/state recorder term."""

    class_type: type[RecorderTerm] = PostStepJointTargetsRecorder


@configclass
class SberchelXRRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Action/state recorder plus locomotion commands, teleop state, joint targets and RGB."""

    record_pre_step_loco_commands = LocomotionCommandRecorderCfg()
    record_pre_step_teleop_state = TeleopStateRecorderCfg()
    record_post_step_joint_targets = PostStepJointTargetsRecorderCfg()
    record_pre_step_rgb_camera_observations = PreStepRgbCameraObservationsRecorderCfg()
