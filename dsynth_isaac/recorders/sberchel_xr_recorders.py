"""Sberchel XR demo recorders for Isaac Lab RecorderManager."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.managers.recorder_manager import RecorderTermCfg
from isaaclab.utils import configclass

LOCO_COMMAND_ATTR = "_sberchel_loco_command"


def stash_locomotion_command(env, vx: float, vy: float, wz: float) -> None:
    """Store per-env locomotion velocity commands for the next ``env.step`` recorder tick."""
    setattr(
        env,
        LOCO_COMMAND_ATTR,
        torch.tensor([[vx, vy, wz]], dtype=torch.float32, device=env.device).repeat(env.num_envs, 1),
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


@configclass
class SberchelXRRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Action/state recorder plus locomotion commands and head-camera RGB."""

    record_pre_step_loco_commands = LocomotionCommandRecorderCfg()
    record_pre_step_rgb_camera_observations = PreStepRgbCameraObservationsRecorderCfg()
