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


@configclass
class SberchelXRRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Action/state recorder plus Sberchel XR locomotion velocity commands."""

    record_pre_step_loco_commands = LocomotionCommandRecorderCfg()
