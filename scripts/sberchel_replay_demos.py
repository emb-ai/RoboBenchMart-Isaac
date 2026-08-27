# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Replay Sberchel XR demonstrations recorded by ``sberchel_kb_record_demos.py`` / record scripts.

Supports the 41D action layout ``[legs(12), neck(2), upper_body_ik(26), torso_yaw(1)]``
with headset neck targets. Legacy 40D demos (no torso_yaw term) are padded with a zero
torso tail, legacy 38D demos additionally with zero neck — zero action = default pose,
exactly what those joints did before their terms existed.

With ``--output_dataset_file`` and ``--enable_cameras``, each replayed episode is copied
into a new HDF5 with live robot RGB for every camera selected via ``--cameras``:
``obs/head_img`` (left head), ``obs/right_head_img``, ``obs/left_wrist_img`` and
``obs/right_wrist_img``. Defaults target the LeRobot sample layout
(``cam_head``/``cam_left_wrist``/``cam_right_wrist``): the left-head + wrists trio,
one frame per ``--frame_stride`` env steps (150/5 = 30 fps), processed through the
green_challenge runtime pipeline (native 960x600 render -> fisheye mask -> full-frame
squash to 448x448; see ``--camera_pipeline``/``--camera_size``). Stride/fps/resolution/
pipeline go into the output's ``env_args`` for the conversion.

With ``--video``, also records viewport MP4s (one per replayed episode) via Gymnasium
``RecordVideo``, following the robot camera like ``demo_random_actions.py``.

With ``--third_person``, the camera export additionally renders a static third-person
scene camera framing the packing table, the shelf rows and the robot, stored as
``obs/third_person_img`` in ``--output_dataset_file`` on the same ``--frame_stride``
grid as the robot cameras. ``sberchel_hdf5_to_video.py`` then tiles it as a fourth
panel next to head/left-wrist/right-wrist. The key is not in the LeRobot converter's
``CAMERA_MAP``, so it never reaches the LeRobot dataset.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from dsynth_isaac.green_challenge_path import ensure_green_challenge_importable  # noqa: E402
from dsynth_isaac.sim_defaults import apply_sberchel_cli_defaults  # noqa: E402

ensure_green_challenge_importable()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay Sberchel XR demonstrations in Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Demo-Sberchel-XR-v0",
    help="Task used for replay (should match the recording task).",
)
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="Episode indices to replay. Empty replays all episodes in the dataset.",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="datasets/sberchel_xr_demos.hdf5",
    help="Dataset file to replay.",
)
parser.add_argument(
    "--output_dataset_file",
    type=str,
    default="",
    help=(
        "If set, write a new HDF5 that copies each replayed episode and adds live "
        "robot RGB for the cameras selected via --cameras (obs/head_img, "
        "obs/right_head_img, obs/left_wrist_img, obs/right_wrist_img). "
        "Requires --enable_cameras."
    ),
)
parser.add_argument(
    "--cameras",
    type=str,
    default="left_head,left_wrist,right_wrist",
    help=(
        "Comma-separated robot cameras to render/export with --output_dataset_file: "
        "left_head,right_head,left_wrist,right_wrist (or 'all'). The default trio "
        "matches the LeRobot sample layout (cam_head/cam_left_wrist/cam_right_wrist). "
        "Cameras not listed are stripped from the scene to save render time."
    ),
)
parser.add_argument(
    "--camera_size",
    type=str,
    default="448x448",
    help=(
        "Output image size: 'WxH', a single square size, or 'native' (960x600). "
        "Default 448x448 matches the LeRobot sample layout."
    ),
)
parser.add_argument(
    "--camera_pipeline",
    type=str,
    choices=("green_challenge", "center_crop"),
    default="green_challenge",
    help=(
        "How frames become --camera_size. 'green_challenge' (default) reproduces the "
        "challenge runtime exactly: render native 960x600, zero pixels outside the "
        "per-camera fisheye mask, then squash-resize the full frame (keeps the full "
        "horizontal FoV). 'center_crop' renders directly at --camera_size with "
        "center-crop-resize rescaled intrinsics (no mask, narrower FoV)."
    ),
)
parser.add_argument(
    "--frame_stride",
    type=int,
    default=5,
    help=(
        "Store every Nth env step's camera frames. With the 150 Hz env step the "
        "default 5 gives 30 fps, the LeRobot sample rate. The stride is written to "
        "the output file's env_args so the conversion can align frames with the "
        "full-rate states/actions (frame k <-> step k*stride)."
    ),
)
parser.add_argument(
    "--fast_export",
    action="store_true",
    default=False,
    help=(
        "Camera-export fast path: skip physics stepping and per-step state sync, "
        "injecting recorded states and rendering only at --frame_stride capture "
        "steps. Frames are rendered from the same recorded states as the normal "
        "path. Requires --output_dataset_file with processed_joint_targets state "
        "sync; incompatible with --video and the validation flags."
    ),
)
parser.add_argument(
    "--per_step_render",
    action="store_true",
    default=False,
    help=(
        "Render every physics step even during headless camera export. By default "
        "camera export skips the per-step render and relies on the explicit render "
        "done at each --frame_stride capture, which is much faster."
    ),
)
parser.add_argument(
    "--validate_states",
    action="store_true",
    default=False,
    help="Validate recorded states against runtime (single env only).",
)
parser.add_argument(
    "--validate_success_rate",
    action="store_true",
    default=False,
    help="Validate replay success rate using task termination criteria.",
)
parser.add_argument(
    "--validate_report",
    type=str,
    default="/tmp/sberchel_replay_validate_report.txt",
    help="When --validate_states is set, write validation output to this file instead of stdout.",
)
parser.add_argument(
    "--replay_source",
    type=str,
    choices=("policy_loco", "processed_joint_targets", "actions"),
    default="policy_loco",
    help=(
        "Replay command source. 'policy_loco' (default) re-runs the recorded leg "
        "policy from recorded loco_commands and applies recorded upper-body joint "
        "targets. 'processed_joint_targets' applies recorded post-IK joint targets "
        "directly. 'actions' feeds raw Pink+leg+neck+torso actions through the env "
        "(re-solves IK); it is rejected for datasets whose arms were driven by the "
        "walk policy, because the raw upper action still holds the frozen-snapshot "
        "IK targets rather than what actually ran."
    ),
)
parser.add_argument(
    "--loco_policy",
    type=str,
    choices=("auto", "zest", "walk"),
    default="auto",
    help=(
        "Which policy drives the legs during LOCOMOTION segments. 'auto' (default) "
        "reads sberchel_loco_policy_by_mode from the dataset's env_args, so a replay "
        "reproduces whatever the recording used; datasets written before that key "
        "existed are single-policy zest. Override only to A/B a policy against a "
        "recording it did not produce."
    ),
)
parser.add_argument(
    "--no_state_sync",
    action="store_true",
    default=False,
    help=(
        "Disable per-step scene state sync from recorded post-step states. By default, "
        "processed_joint_targets replay re-applies the previous recorded scene state "
        "before each step to prevent open-loop drift on the floating humanoid."
    ),
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio (required for Pink IK replay).",
)
parser.add_argument(
    "--enhanced_determinism",
    action="store_true",
    default=False,
    help=(
        "Force PhysX enhanced determinism (PhysxCfg.enable_enhanced_determinism) on for "
        "replay. Normally not needed: replay auto-matches the setting stored in the "
        "dataset's provenance (sberchel_enhanced_determinism), since replaying with a "
        "different value than the recording changes the physics and causes divergence."
    ),
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record viewport MP4s via Gymnasium RecordVideo (one file per replayed episode).",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default="",
    help="Directory for viewport MP4s. Default: <dataset_dir>/videos.",
)
parser.add_argument(
    "--third_person",
    action="store_true",
    default=False,
    help=(
        "Also store obs/third_person_img in --output_dataset_file: a static "
        "third-person scene camera framing the packing table, the shelf rows and "
        "the robot, captured on the same --frame_stride grid as the robot "
        "cameras. Visualization-only: the key is outside the LeRobot converter's "
        "CAMERA_MAP, so it never reaches the LeRobot dataset. Requires "
        "--output_dataset_file."
    ),
)
parser.add_argument(
    "--third_person_pose",
    type=str,
    default="4.0,1.0,2.5,9.5,6.0,0.0",
    help=(
        "Static camera pose 'eye_x,eye_y,eye_z,look_x,look_y,look_z' in env-local "
        "meters. The default is the replay viewer vantage for the small_store_aij "
        "layout: packing table left, shelf rows center/right, robot in between."
    ),
)
parser.add_argument(
    "--third_person_size",
    type=str,
    default="640x360",
    help=(
        "Third-person render size 'WxH' (default 640x360 — kept modest since the "
        "frames are stored uncompressed in the output HDF5)."
    ),
)

AppLauncher.add_app_launcher_args(parser)
# Sberchel runs on green_challenge's CPU physics by default; --device overrides.
apply_sberchel_cli_defaults(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher_args = vars(args_cli)
# Viewport rgb_array / RecordVideo and the third-person scene camera need camera
# rendering enabled.
if args_cli.video or args_cli.third_person:
    app_launcher_args["enable_cameras"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import contextlib

import cv2
import gymnasium as gym
import h5py
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class _PlainProgress:
    """Line-based progress fallback when tqdm is unavailable."""

    def __init__(self, total: int):
        self.total = max(1, total)
        self.n = 0
        self._last_pct = -1

    def update(self, n: int = 1) -> None:
        self.n += n
        pct = int(100 * self.n / self.total)
        if pct != self._last_pct and pct % 2 == 0:
            print(f"[replay] {self.n}/{self.total} steps ({pct}%)", flush=True)
            self._last_pct = pct

    def close(self) -> None:
        pass


def make_step_progress(total_steps: int):
    if tqdm is not None:
        return tqdm(total=total_steps, desc="replay", unit="step", dynamic_ncols=True)
    return _PlainProgress(total_steps)

import isaaclab.sim as sim_utils
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

import dsynth_isaac.tasks.manager_based  # noqa: F401

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from dsynth_isaac.recorders.sberchel_xr_recorders import (
    _write_episode_with_extra_attrs,
    stash_episode_extra_attrs,
)
from dsynth_isaac.tasks.manager_based.sberchel_base_env_cfg import (
    LEGS_ACTION_DIM,
    NECK_ACTION_DIM,
    LocoPolicySet,
)
from dsynth_isaac.robots.sberchel_camera_cfg import (
    SBERCHEL_CAMERA_OBS_KEYS,
    apply_sberchel_camera_size,
    load_sberchel_camera_masks,
    parse_camera_names,
    parse_camera_size,
    strip_camera_configs,
)


def make_green_challenge_frame_postprocess(mask: np.ndarray, out_size: tuple[int, int] | None):
    """Mask at native resolution then squash-resize the full frame.

    Mirrors green_challenge's ``LatestFrameHandler``: ``rgb_image[mask == 0] = 0``
    followed by ``cv2.resize`` (default INTER_LINEAR) of the whole 960x600 frame,
    keeping the full horizontal fisheye FoV.
    """
    invalid = mask == 0

    def _apply(frame: torch.Tensor) -> torch.Tensor:
        img = frame.numpy()
        if img.shape[:2] != invalid.shape:
            raise ValueError(
                f"Frame {img.shape[:2]} does not match camera mask {invalid.shape}; "
                "the green_challenge pipeline requires native-resolution rendering."
            )
        img = img.copy()
        img[invalid] = 0
        if out_size is not None and out_size != (img.shape[1], img.shape[0]):
            img = cv2.resize(img, out_size)
        return torch.from_numpy(img)

    return _apply


# --third_person camera optics: ~72 deg horizontal FOV keeps the packing
# table, the shelf rows and the walking robot in frame from the default vantage
# without fisheye distortion.
_THIRD_PERSON_FOCAL_LENGTH = 14.4
_THIRD_PERSON_HORIZONTAL_APERTURE = 20.955
# obs key for the exported frames. Deliberately NOT a SBERCHEL_CAMERA_OBS_KEYS /
# lerobot CAMERA_MAP key: hdf5_to_video auto-detects it as an extra panel while
# hdf5_to_lerobot ignores it.
THIRD_PERSON_IMG_KEY = "third_person_img"


def parse_third_person_pose(spec: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Parse ``--third_person_pose`` into env-local ``(eye, lookat)`` triples."""
    try:
        values = [float(item) for item in spec.replace(" ", "").split(",") if item]
    except ValueError as exc:
        raise ValueError(f"Invalid --third_person_pose {spec!r}: {exc}") from exc
    if len(values) != 6:
        raise ValueError(
            "--third_person_pose needs 6 comma-separated floats "
            f"(eye_x,eye_y,eye_z,look_x,look_y,look_z), got {len(values)}."
        )
    return tuple(values[:3]), tuple(values[3:])


def make_third_person_camera_cfg(size: tuple[int, int]) -> CameraCfg:
    """Free-standing pinhole RGB camera (no parent body, so it never moves).

    The pose is applied after env creation via ``set_world_poses_from_view``;
    the cfg only spawns the prim and render product.
    """
    width, height = size
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/third_person_camera",
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_THIRD_PERSON_FOCAL_LENGTH,
            horizontal_aperture=_THIRD_PERSON_HORIZONTAL_APERTURE,
            clipping_range=(0.1, 1.0e6),
        ),
    )


is_paused = False

# Action layout: [legs(12), neck(2), upper_body_ik(26), torso_yaw(1)] = 41D.
# torso_yaw is appended last so legacy 40D/38D rows stay prefix-compatible.
_UPPER_ACTION_OFFSET = LEGS_ACTION_DIM + NECK_ACTION_DIM
_UPPER_ACTION_DIM = 26  # Pink idle / raw upper slice size


def play_cb() -> None:
    global is_paused
    is_paused = False


def pause_cb() -> None:
    global is_paused
    is_paused = True


def _flatten_state_tensor(value: torch.Tensor) -> torch.Tensor:
    """Normalize scene-state tensors for element-wise comparison."""
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.reshape(-1).detach().cpu()


def compare_states(state_from_dataset, runtime_state, runtime_env_index) -> tuple[bool, str, float]:
    states_matched = True
    output_log = ""
    max_abs_error = 0.0
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = _flatten_state_tensor(
                    runtime_state[asset_type][asset_name][state_name][runtime_env_index]
                )
                dataset_asset_state = _flatten_state_tensor(
                    state_from_dataset[asset_type][asset_name][state_name]
                )
                if dataset_asset_state.numel() != runtime_asset_state.numel():
                    raise ValueError(
                        f"State shape of {state_name} for asset {asset_name} don't match: "
                        f"dataset {tuple(dataset_asset_state.shape)} vs runtime {tuple(runtime_asset_state.shape)}"
                    )
                for i in range(dataset_asset_state.numel()):
                    abs_error = abs(dataset_asset_state[i] - runtime_asset_state[i]).item()
                    max_abs_error = max(max_abs_error, abs_error)
                    if abs_error > 0.01:
                        states_matched = False
                        output_log += f'\tState ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] don\'t match\r\n'
                        output_log += f"\t  Dataset:\t{dataset_asset_state[i]}\r\n"
                        output_log += f"\t  Runtime: \t{runtime_asset_state[i]}\r\n"
    return states_matched, output_log, max_abs_error


def build_replay_idle_action(env, env_cfg, num_envs: int) -> torch.Tensor:
    """Build a full action-dim idle tensor for multi-env filler rows.

    Used only for envs that have no command this step (Isaac Lab pattern). Legs/neck
    stay zero so ``JointPositionAction`` ``use_default_offset=True`` holds default
    pose; upper body uses ``env_cfg.idle_action`` Pink targets when present.
    """
    action_dim = env.action_manager.total_action_dim
    idle = torch.zeros(num_envs, action_dim, device=env.device)

    if hasattr(env_cfg, "idle_action"):
        cfg_idle = env_cfg.idle_action.to(env.device).reshape(-1)
        if cfg_idle.numel() == action_dim:
            idle[:] = cfg_idle.unsqueeze(0).repeat(num_envs, 1)
            return idle
        # 26D Pink-only idle (current Demo/Fetch XR cfg).
        if cfg_idle.numel() == _UPPER_ACTION_DIM:
            idle[:, _UPPER_ACTION_OFFSET : _UPPER_ACTION_OFFSET + _UPPER_ACTION_DIM] = (
                cfg_idle.unsqueeze(0).repeat(num_envs, 1)
            )
        # Legacy padding: neck+upper after legs (28D) if someone stores that.
        elif cfg_idle.numel() == action_dim - LEGS_ACTION_DIM:
            idle[:, LEGS_ACTION_DIM:] = cfg_idle.unsqueeze(0).repeat(num_envs, 1)
    return idle


def prepare_sberchel_xr_episode_replay(env, env_cfg, num_envs: int) -> torch.Tensor:
    """Re-apply teleop PD gains and rebuild idle actions after ``reset`` / ``reset_to``."""
    return build_replay_idle_action(env, env_cfg, num_envs)


def get_next_replay_command(episode_data: EpisodeData, replay_source: str) -> torch.Tensor | None:
    """Return the next replay command tensor, advancing the episode cursor."""
    if replay_source == "processed_joint_targets":
        processed = episode_data.data.get("processed_actions")
        if processed is None:
            if episode_data.is_empty():
                return None
            raise ValueError(
                "Dataset episode is missing 'processed_actions' required for processed_joint_targets replay."
            )
        index = episode_data.next_action_index
        if index >= len(processed):
            return None
        episode_data.next_action_index = index + 1
        return processed[index]
    if replay_source == "policy_loco":
        return None
    return episode_data.get_next_action()


class PolicyLocoStep(NamedTuple):
    """One recorded step's inputs for ``policy_loco`` replay.

    A named tuple rather than a bare one: this is unpacked and stored at four
    sites, and a silent arity change is exactly how a per-step misalignment gets
    introduced.
    """

    loco_command: torch.Tensor
    processed_actions: torch.Tensor
    is_locomotion: bool
    # --zest_upper datasets only: the 13-value absolute commander reference
    # (torso_yaw + arms + neck) the recorder fed to zest on this step. None on
    # legacy datasets, which replays as the standing-pose reference — exactly
    # what those sessions ran.
    zest_upper_ref: torch.Tensor | None = None


def get_next_policy_loco_step(episode_data: EpisodeData) -> PolicyLocoStep | None:
    """Return the recorded command, joint targets and mode for the next step."""
    if episode_data.is_empty():
        return None
    loco_commands = episode_data.data.get("loco_commands")
    processed_actions = episode_data.data.get("processed_actions")
    teleop_state = episode_data.data.get("teleop_state")
    zest_upper_refs = episode_data.data.get("zest_upper_ref")
    if loco_commands is None:
        raise ValueError(
            "Dataset episode is missing 'loco_commands' required for policy_loco replay. "
            "Re-record demos with the updated sberchel_record_demos.py."
        )
    if processed_actions is None:
        raise ValueError(
            "Dataset episode is missing 'processed_actions' required for policy_loco replay."
        )
    index = episode_data.next_action_index
    if index >= len(loco_commands):
        return None
    episode_data.next_action_index = index + 1
    # teleop_state[:, 0] is the recorder's locomotion flag, i.e. the per-step
    # selector between the two policies. Datasets written before it existed are
    # single-policy anyway, so the default never picks the wrong net.
    is_locomotion = True
    if teleop_state is not None and index < len(teleop_state):
        is_locomotion = bool(float(teleop_state[index].reshape(-1)[0]) > 0.5)
    zest_upper_ref = None
    if zest_upper_refs is not None and index < len(zest_upper_refs):
        zest_upper_ref = zest_upper_refs[index]
    return PolicyLocoStep(
        loco_commands[index], processed_actions[index], is_locomotion, zest_upper_ref
    )


def _processed_term_dims(env) -> tuple[int, int, int, int]:
    """Return ``(legs_dim, neck_dim, upper_dim, torso_dim)`` from live action terms."""
    legs_dim = env.action_manager.get_term("legs_joint_pos")._processed_actions.shape[-1]
    upper_dim = env.action_manager.get_term("upper_body_ik")._processed_actions.shape[-1]
    if "neck_joint_pos" in env.action_manager.active_terms:
        neck_dim = env.action_manager.get_term("neck_joint_pos")._processed_actions.shape[-1]
    else:
        neck_dim = 0
    if "torso_yaw_joint_pos" in env.action_manager.active_terms:
        torso_dim = env.action_manager.get_term("torso_yaw_joint_pos")._processed_actions.shape[-1]
    else:
        torso_dim = 0
    return legs_dim, neck_dim, upper_dim, torso_dim


def _split_processed_joint_targets(
    processed: torch.Tensor,
    env,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split processed targets into ``(legs, neck, upper, torso)``.

    Uses live action-term processed widths. Legacy recordings get zero targets
    for the slices they predate: 40D rows lack the torso_yaw tail, 38D rows also
    lack the neck slice (zero action = default pose for those offset terms).
    """
    flat = processed.reshape(-1)
    legs_dim, neck_dim, upper_dim, torso_dim = _processed_term_dims(env)

    def _zero_torso() -> torch.Tensor:
        return torch.zeros(max(torso_dim, 1), device=flat.device, dtype=flat.dtype)[
            : torso_dim if torso_dim > 0 else 1
        ]

    # Current layout: legs + neck + upper + torso.
    if (
        neck_dim > 0
        and torso_dim > 0
        and flat.numel() == legs_dim + neck_dim + upper_dim + torso_dim
    ):
        return (
            flat[:legs_dim],
            flat[legs_dim : legs_dim + neck_dim],
            flat[legs_dim + neck_dim : legs_dim + neck_dim + upper_dim],
            flat[legs_dim + neck_dim + upper_dim :],
        )

    # Legacy 40D: legs + neck + upper (recorded before the torso_yaw term).
    if neck_dim > 0 and flat.numel() == legs_dim + neck_dim + upper_dim:
        return (
            flat[:legs_dim],
            flat[legs_dim : legs_dim + neck_dim],
            flat[legs_dim + neck_dim :],
            _zero_torso(),
        )

    # Legacy 38D: legs + upper.
    if flat.numel() == legs_dim + upper_dim:
        neck = torch.zeros(max(neck_dim, NECK_ACTION_DIM), device=flat.device, dtype=flat.dtype)
        if neck_dim > 0:
            neck = neck[:neck_dim]
        return flat[:legs_dim], neck, flat[legs_dim:], _zero_torso()

    # Best-effort fallback for mismatched recordings.
    legs = flat[:legs_dim]
    if neck_dim > 0 and flat.numel() > legs_dim + neck_dim:
        neck = flat[legs_dim : legs_dim + neck_dim]
        upper = flat[legs_dim + neck_dim :]
    else:
        neck = torch.zeros(max(neck_dim, 1) if neck_dim else NECK_ACTION_DIM, device=flat.device, dtype=flat.dtype)
        if neck_dim > 0:
            neck = neck[:neck_dim]
        upper = flat[legs_dim:]
    if torso_dim > 0 and upper.numel() > upper_dim:
        torso = upper[upper_dim : upper_dim + torso_dim]
        upper = upper[:upper_dim]
    else:
        torso = _zero_torso()
    return legs, neck, upper, torso


def assert_dataset_matches_robot(dataset_env_args: dict, robot) -> None:
    """Abort unless the dataset was recorded on the robot now being spawned.

    Demos collected before the switch to green_challenge's closed-kinematics
    ``robot_v3_0a`` MJCF were recorded on the retired hybrid/open model: 53 dofs
    with ``*_wrist_pitch_joint`` actuated, and serial leg joints wearing crank
    names. The closed model has 73 dofs and drives ``*_wrist_crank_joint``.
    Recorded joint targets therefore land on different joints, which is silent
    -- the replay runs, on a plant that did not produce the demo.
    """
    recorded = list(dataset_env_args.get("sberchel_joint_names") or [])
    if not recorded:
        print(
            "Warning: dataset carries no 'sberchel_joint_names'; cannot verify it was "
            "recorded on this robot model. Replay outputs drop the key, so this is "
            "expected when re-replaying an augmented file.",
            flush=True,
        )
        return
    live = list(robot.data.joint_names)
    if recorded == live:
        return
    only_recorded = sorted(set(recorded) - set(live))
    only_live = sorted(set(live) - set(recorded))
    detail = "joint order differs (same names)"
    if only_recorded or only_live:
        detail = (
            f"joints only in dataset: {only_recorded or '[]'}; "
            f"joints only in sim: {only_live or '[]'}"
        )
    raise SystemExit(
        "Refusing to replay: this dataset was recorded on a different robot model.\n"
        f"  dataset joints: {len(recorded)}\n"
        f"  sim joints    : {len(live)}\n"
        f"  {detail}\n"
        "Datasets recorded before the switch to green_challenge's closed-kinematics "
        "robot_v3_0a MJCF cannot be replayed on it: their joint targets would be "
        "applied to different dofs. Re-record them, or replay them with the commit "
        "they were recorded on."
    )


def step_recorded_joint_targets(
    env,
    legs: torch.Tensor,
    upper: torch.Tensor,
    neck: torch.Tensor | None = None,
    torso: torch.Tensor | None = None,
) -> None:
    """Step physics with separate leg/neck/upper/torso joint targets, bypassing Pink IK re-solve."""
    robot = env.scene["robot"]
    legs_term = env.action_manager.get_term("legs_joint_pos")
    upper_term = env.action_manager.get_term("upper_body_ik")
    neck_term = None
    if "neck_joint_pos" in env.action_manager.active_terms:
        neck_term = env.action_manager.get_term("neck_joint_pos")
    torso_term = None
    if "torso_yaw_joint_pos" in env.action_manager.active_terms:
        torso_term = env.action_manager.get_term("torso_yaw_joint_pos")

    legs_term._processed_actions[:] = legs
    upper_term._processed_actions[:] = upper
    if neck_term is not None:
        if neck is None:
            neck = torch.zeros(
                legs.shape[0],
                neck_term._processed_actions.shape[-1],
                device=legs.device,
                dtype=legs.dtype,
            )
        neck_term._processed_actions[:] = neck
    if torso_term is not None:
        if torso is None:
            torso = torch.zeros(
                legs.shape[0],
                torso_term._processed_actions.shape[-1],
                device=legs.device,
                dtype=legs.dtype,
            )
        torso_term._processed_actions[:] = torso.reshape(
            legs.shape[0], torso_term._processed_actions.shape[-1]
        )

    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        legs_term.apply_actions()
        if neck_term is not None:
            neck_term.apply_actions()
        if torso_term is not None:
            torso_term.apply_actions()
        robot.set_joint_position_target(upper, joint_ids=upper_term._controlled_joint_ids)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
            env.sim.render()
        env.scene.update(dt=env.physics_dt)

    env.episode_length_buf += 1
    env.common_step_counter += 1


def step_processed_joint_targets(env, processed_actions: torch.Tensor) -> None:
    """Step physics using recorded joint targets, bypassing Pink IK re-solve."""
    legs_list = []
    neck_list = []
    upper_list = []
    torso_list = []
    for env_id in range(processed_actions.shape[0]):
        legs, neck, upper, torso = _split_processed_joint_targets(processed_actions[env_id], env)
        legs_list.append(legs)
        neck_list.append(neck)
        upper_list.append(upper)
        torso_list.append(torso)
    step_recorded_joint_targets(
        env,
        torch.stack(legs_list, dim=0),
        torch.stack(upper_list, dim=0),
        torch.stack(neck_list, dim=0),
        torch.stack(torso_list, dim=0),
    )


def normalize_processed_to_env(processed: torch.Tensor, env) -> torch.Tensor:
    """Pad legacy processed targets (zero neck slice / zero torso tail) when needed."""
    flat = processed.reshape(-1).to(dtype=torch.float32)
    legs_dim, neck_dim, upper_dim, torso_dim = _processed_term_dims(env)
    expected = legs_dim + neck_dim + upper_dim + torso_dim
    if flat.numel() == expected:
        return flat
    # Legacy 38D rows lack the neck slice in the middle; insert it, then fall
    # through to the tail pad (which supplies the missing torso zeros).
    if neck_dim > 0 and flat.numel() in (legs_dim + upper_dim, legs_dim + upper_dim + torso_dim):
        flat = torch.cat(
            [
                flat[:legs_dim],
                torch.zeros(neck_dim, device=flat.device, dtype=flat.dtype),
                flat[legs_dim:],
            ],
            dim=0,
        )
        if flat.numel() == expected:
            return flat
    if flat.numel() < expected:
        out = torch.zeros(expected, device=flat.device, dtype=flat.dtype)
        out[: flat.numel()] = flat
        return out
    return flat[:expected]


def normalize_raw_action_to_env(action: torch.Tensor, action_dim: int) -> torch.Tensor:
    """Pad legacy raw actions (zero neck slice / zero torso tail) to ``action_dim``.

    Legacy 40D rows only need the trailing torso zero; legacy 38D rows also get
    the neck slice inserted after the legs before the tail pad.
    """
    flat = action.reshape(-1).to(dtype=torch.float32)
    if flat.numel() == action_dim:
        return flat
    # Legacy [legs(12), upper(26)] -> insert zero neck; the tail pad below
    # supplies the torso zero(s) if the env carries the torso_yaw term.
    if (
        flat.numel() == LEGS_ACTION_DIM + _UPPER_ACTION_DIM
        and action_dim >= LEGS_ACTION_DIM + NECK_ACTION_DIM + _UPPER_ACTION_DIM
    ):
        flat = torch.cat(
            [
                flat[:LEGS_ACTION_DIM],
                torch.zeros(NECK_ACTION_DIM, device=flat.device, dtype=flat.dtype),
                flat[LEGS_ACTION_DIM:],
            ],
            dim=0,
        )
        if flat.numel() == action_dim:
            return flat
    if flat.numel() < action_dim:
        out = torch.zeros(action_dim, device=flat.device, dtype=flat.dtype)
        out[: flat.numel()] = flat
        return out
    return flat[:action_dim]


def _write_action_row(actions: torch.Tensor, env_id: int, command: torch.Tensor, env, replay_source: str) -> None:
    """Write one env's fetched replay command into the batched action tensor."""
    if replay_source == "actions":
        normalized = normalize_raw_action_to_env(command, env.action_manager.total_action_dim)
        actions[env_id] = normalized.to(device=env.device, dtype=actions.dtype)
        return
    if replay_source == "processed_joint_targets":
        normalized = normalize_processed_to_env(command, env)
        if normalized.numel() != actions.shape[1]:
            if normalized.numel() > actions.shape[1]:
                raise ValueError(
                    f"Processed action dim {normalized.numel()} exceeds env action dim {actions.shape[1]}."
                )
            row = torch.zeros(actions.shape[1], device=env.device, dtype=actions.dtype)
            row[: normalized.numel()] = normalized.to(device=env.device, dtype=actions.dtype)
            actions[env_id] = row
        else:
            actions[env_id] = normalized.to(device=env.device, dtype=actions.dtype)


def stop_viewport_episode_record(video_env: gym.Env | None) -> None:
    """Finalize the in-progress viewport MP4, if any."""
    if isinstance(video_env, gym.wrappers.RecordVideo) and video_env.recording:
        video_env.stop_recording()


def begin_viewport_episode_record(video_env: gym.Env | None) -> None:
    """Start a new viewport MP4 for the next replayed episode."""
    if not isinstance(video_env, gym.wrappers.RecordVideo):
        return
    stop_viewport_episode_record(video_env)
    video_env.episode_id += 1
    if video_env.episode_trigger(video_env.episode_id):
        file_name = (
            f"{video_env.name_prefix}-episode-{video_env.episode_id}"
            if video_env.name_prefix
            else f"rl-video-episode-{video_env.episode_id}"
        )
        video_env.start_recording(file_name)


def capture_viewport_frame(video_env: gym.Env | None, base_env) -> None:
    """Append one viewport frame after a custom physics step (bypasses ``env.step``)."""
    if not isinstance(video_env, gym.wrappers.RecordVideo) or not video_env.recording:
        return
    # Render from the unwrapped Isaac Lab env: custom replay steps do not call the
    # wrapped env.step/reset, so OrderEnforcer on the Gymnasium stack would reject
    # video_env.env.render().
    frame = base_env.render(recompute=True)
    if frame is None:
        return
    video_env.recorded_frames.append(frame)
    video_env.step_id += 1
    if video_env.video_length > 0 and video_env.step_id >= video_env.video_length:
        video_env.stop_recording()


def _zero_state_velocities(state):
    """Copy a scene-state tree with every ``*velocity*`` tensor zeroed."""
    if isinstance(state, dict):
        return {
            key: (
                torch.zeros_like(value)
                if "velocity" in key and isinstance(value, torch.Tensor)
                else _zero_state_velocities(value)
            )
            for key, value in state.items()
        }
    return state


def sync_scene_to_recorded_state(env, recorded_state: dict, env_id: int, zero_velocities: bool = False) -> None:
    """Inject a recorded post-step scene state without running env ``reset_to`` managers.

    ``zero_velocities`` injects the recorded poses with zeroed root/joint
    velocities (and zero velocity targets), so a following settle ``sim.step``
    holds the pose still instead of integrating it one physics dt forward.
    """
    env_ids = torch.tensor([env_id], device=env.device)
    if zero_velocities:
        recorded_state = _zero_state_velocities(recorded_state)
    env.scene.reset_to(recorded_state, env_ids=env_ids, is_relative=True)
    env.scene.write_data_to_sim()
    env.sim.forward()


_warned_no_store_products = False
# Indices into the live store snapshot for each recorded store_products row
# (LongTensor), or None when recorded rows already cover the live store in
# order. Set once in main() via build_store_row_body_indices().
_store_row_body_indices = None


def _dataset_store_product_count(dataset_file: str) -> int | None:
    """Store product rows per step in the dataset, or None if none are recorded."""
    try:
        with h5py.File(dataset_file, "r") as handle:
            for demo in handle["data"].keys():
                group = handle["data"][demo]
                if "store_products" in group:
                    return int(group["store_products/transforms"].shape[1])
    except Exception as exc:  # a diagnostic path: never mask the real failure
        print(f"Warning: could not read store_products row count from {dataset_file}: {exc!r}")
    return None


def build_store_row_body_indices(env, dataset_file: str, dataset_env_args: dict):
    """Map recorded store_products rows onto the live store snapshot by prim path.

    Demos recorded with a lean store (--keep_boards in the recorder) carry rows
    for only the populated boards, while replay runs the fully stocked store.
    The recorder's ``<dataset>.store_prims.json`` sidecar gives the recorded
    prim order; each path is matched to its index in the live snapshot. Returns
    ``None`` when the recorded order equals the live order (full-store
    recordings; legacy demos predate the sidecar and were always full-store).
    """
    snapshot = getattr(env.scene, "_store_rigid_snapshot", None)
    if snapshot is None:
        return None
    live_paths = list(snapshot.prim_paths)
    sidecar = Path(dataset_file).with_suffix(".store_prims.json")
    if not sidecar.exists():
        recorded_count = dataset_env_args.get("sberchel_store_prim_count")
        if recorded_count is None:
            # Sessions recorded while the provenance write was broken carry
            # neither the sidecar nor this count, so read the row width off the
            # episodes themselves -- otherwise the mismatch surfaces much later
            # as a raw PhysX "Incompatible size of transform tensor" error.
            recorded_count = _dataset_store_product_count(dataset_file)
        if recorded_count is not None and int(recorded_count) != len(live_paths):
            raise FileNotFoundError(
                f"Dataset records {recorded_count} store products but the replay scene has "
                f"{len(live_paths)}; mapping lean-store rows onto the full store needs the "
                f"{sidecar.name} sidecar next to the dataset file. Recover it with:\n"
                f"  python scripts/sberchel_rebuild_store_prims_sidecar.py {dataset_file}"
            )
        return None
    recorded_paths = json.loads(sidecar.read_text())
    if recorded_paths == live_paths:
        return None
    live_index = {path: i for i, path in enumerate(live_paths)}
    missing = [path for path in recorded_paths if path not in live_index]
    if missing:
        raise ValueError(
            f"{len(missing)} recorded store product prims are absent from the replay scene "
            f"(first: {missing[0]}). The replay store version does not match the recording."
        )
    print(
        f">>> store products: mapping {len(recorded_paths)} recorded rows onto "
        f"{len(live_paths)} live bodies by prim path (lean-store recording)"
    )
    return torch.tensor([live_index[path] for path in recorded_paths], dtype=torch.long)


def sync_store_products_to_recorded(
    env,
    episode_data: EpisodeData,
    step_idx: int,
    num_envs: int,
    only_moved: bool = True,
    zero_velocities: bool = False,
) -> None:
    """Pin unregistered store product bodies to their recorded step state.

    Store products are plain USD rigid bodies, not scene assets, so the robot
    state sync alone leaves them on open-loop physics and a grasped item drifts
    out of the hand. No-op with a one-time warning for demos recorded before
    ``PostStepStoreProductsRecorder`` existed.

    ``only_moved=True`` (interactive replay) restores only bodies drifted beyond
    the restore tolerances (2 cm / ~10 deg / 0.25 m/s), preserving PhysX contact
    warm-start on undisturbed shelves. Camera export must pass ``False``: a
    carried item moves right at those tolerances per frame, so tolerance-gated
    pinning renders it lagging/snapping around the hand. ``zero_velocities``
    pins poses with zero velocity so a settle step does not advance them.
    """
    global _warned_no_store_products
    store = episode_data.data.get("store_products")
    restore = getattr(env.scene, "restore_store_rigid_body_state", None)
    if store is None or restore is None or num_envs != 1:
        if not _warned_no_store_products:
            _warned_no_store_products = True
            if store is None:
                reason = "episode has no store_products rows (recorded before store-state recording)"
            elif restore is None:
                reason = "scene cannot restore store products"
            else:
                reason = "store product sync is single-env only"
            print(f"Warning: store products not synced — {reason}; grasped objects may drift.")
        return
    if _store_row_body_indices is None:
        live_count = len(getattr(env.scene, "_store_rigid_snapshot").prim_paths)
        if store["transforms"].shape[1] != live_count:
            raise ValueError(
                f"Recorded store_products rows ({store['transforms'].shape[1]}) do not match "
                f"the live store ({live_count} bodies) and no .store_prims.json sidecar was "
                "found next to the dataset to map them by prim path. Lean-store recordings "
                "need their sidecar for replay."
            )
    velocities = store["velocities"][step_idx]
    if zero_velocities:
        velocities = torch.zeros_like(velocities)
    restore(
        env.sim,
        {
            "transforms": store["transforms"][step_idx],
            "velocities": velocities,
        },
        only_moved=only_moved,
        body_indices=_store_row_body_indices,
    )


def _clone_tensor_tree(value):
    """Deep-clone nested dicts of tensors (leave non-tensors unchanged)."""
    if isinstance(value, dict):
        return {key: _clone_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return value


def _squeeze_extra_state_dims(value):
    """Drop stray leading singleton dims from recorded state tensors.

    Episodes whose initial_state was re-captured via record_post_reset(None)
    carry an extra dim (e.g. root_pose (1, 1, 7) instead of (1, 7)) because the
    stock InitialStateRecorder indexes value[None] there; scene.reset_to needs
    (num_envs, ...).
    """
    if isinstance(value, dict):
        return {key: _squeeze_extra_state_dims(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        while value.dim() > 2 and value.shape[0] == 1:
            value = value.squeeze(0)
    return value


def clone_episode_data(episode: EpisodeData) -> EpisodeData:
    """Copy episode tensors for export without sharing storage with the replay cursor."""
    cloned = EpisodeData()
    cloned.data = _clone_tensor_tree(episode.data)
    cloned.seed = episode.seed
    cloned.success = episode.success
    cloned.env_id = episode.env_id
    return cloned


def _expected_frame_count(num_actions: int, frame_stride: int) -> int:
    """Frames stored at action indices 0, stride, 2*stride, ..."""
    return -(-num_actions // frame_stride)


def write_episode_with_camera_frames(
    output_handler: HDF5DatasetFileHandler,
    source_episode: EpisodeData,
    frames_by_key: dict[str, list[torch.Tensor]],
    demo_id: int | None = None,
    frame_stride: int = 1,
    extra_attrs: dict | None = None,
) -> None:
    """Copy ``source_episode`` and append stacked ``obs/<img_key>`` frames per camera.

    ``extra_attrs`` are written as HDF5 attrs on the output demo group — the
    stock episode loader/writer only carries seed/success, so per-demo attrs
    like target_item/task_instruction must be threaded through explicitly or
    the camera-augmented file silently loses them.
    """
    actions = source_episode.data.get("actions")
    if actions is None:
        raise ValueError("Source episode is missing 'actions'; cannot export augmented demo.")
    expected = _expected_frame_count(len(actions), frame_stride)
    for img_key, frames in frames_by_key.items():
        if len(frames) != expected:
            raise ValueError(
                f"Collected {len(frames)} {img_key} frames but episode has {len(actions)} "
                f"actions (expected {expected} frames at stride {frame_stride})."
            )
    episode = clone_episode_data(source_episode)
    if "obs" not in episode.data or episode.data["obs"] is None:
        episode.data["obs"] = {}
    for img_key, frames in frames_by_key.items():
        episode.data["obs"][img_key] = torch.stack(frames, dim=0)
    if extra_attrs:
        stash_episode_extra_attrs(episode, **extra_attrs)
    _write_episode_with_extra_attrs(output_handler, episode, demo_id)
    output_handler.flush()


def _maybe_flush_incomplete_ok(
    output_handler: HDF5DatasetFileHandler | None,
    source_episode: EpisodeData | None,
    frames_by_key: dict[str, list[torch.Tensor]],
    demo_id: int | None,
    *,
    allow_incomplete: bool,
    frame_stride: int = 1,
    extra_attrs: dict | None = None,
) -> bool:
    """Write augmented episode; return True if written. Incomplete skipped when allowed."""
    if output_handler is None or source_episode is None:
        return False
    if not any(frames_by_key.values()):
        return False
    actions = source_episode.data.get("actions")
    expected = 0 if actions is None else _expected_frame_count(len(actions), frame_stride)
    for img_key, frames in frames_by_key.items():
        if len(frames) != expected:
            if allow_incomplete:
                print(
                    f"Warning: skipping incomplete camera augment for demo_{demo_id} "
                    f"({len(frames)}/{expected} {img_key} frames)."
                )
                return False
            raise ValueError(
                f"Collected {len(frames)} {img_key} frames but expected {expected} "
                f"at stride {frame_stride}."
            )
    write_episode_with_camera_frames(
        output_handler,
        source_episode,
        frames_by_key,
        demo_id=demo_id,
        frame_stride=frame_stride,
        extra_attrs=extra_attrs,
    )
    return True


def main() -> None:
    global is_paused

    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")

    camera_augment = bool(args_cli.output_dataset_file)
    if camera_augment:
        if not args_cli.enable_cameras:
            raise ValueError("--output_dataset_file requires --enable_cameras.")
        if os.path.abspath(args_cli.output_dataset_file) == os.path.abspath(args_cli.dataset_file):
            raise ValueError("--output_dataset_file must differ from --dataset_file.")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        return 1

    episode_indices_to_replay = list(args_cli.select_episodes)
    if len(episode_indices_to_replay) == 0:
        episode_indices_to_replay = list(range(episode_count))
    # The list above is consumed during replay; keep the valid request for the
    # end-of-run export reconcile (missing exports must fail the process).
    requested_episode_indices = [i for i in episode_indices_to_replay if i < episode_count]

    if args_cli.task is not None:
        env_name = args_cli.task.split(":")[-1]
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    num_envs = args_cli.num_envs
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs)

    # PhysX enhanced determinism is a plant-level setting: replaying with a
    # different value than the recording changes the simulation itself, so
    # auto-match the dataset provenance; --enhanced_determinism forces it on.
    with h5py.File(args_cli.dataset_file, "r") as _f:
        _dataset_env_args = json.loads(_f["data"].attrs.get("env_args", "{}"))
        # Per-demo attrs beyond the stock trio (num_samples/seed/success stay
        # handled by the episode loader/writer): target_item, task_instruction,
        # operator-stamped flags. The stock loader drops them, so read them here
        # and re-stamp each output demo group on export.
        _stock_attrs = {"num_samples", "seed", "success"}
        source_demo_attrs = {
            name: {
                key: value
                for key, value in _f[f"data/{name}"].attrs.items()
                if key not in _stock_attrs
            }
            for name in _f["data"]
        }
    # CPU and GPU PhysX are different implementations, so replaying on the other
    # one is a plant change. Unlike enhanced_determinism this cannot be
    # auto-matched here: the app has already launched with --device, so say what
    # to pass instead of switching underneath the user.
    _recorded_device = _dataset_env_args.get("sberchel_sim_device")
    if _recorded_device and _recorded_device != args_cli.device:
        print(
            f"Warning: this demo was recorded with sim.device={_recorded_device!r} but replay is "
            f"running on {args_cli.device!r}. CPU and GPU PhysX do not step identically, so the "
            f"replayed trajectory may diverge. Re-run with --device {_recorded_device} to match.",
            flush=True,
        )

    # Which policy drove each half of the recording. Datasets written before
    # per-mode policies existed carry no key and were single-policy zest.
    _replay_loco_by_mode = dict(
        _dataset_env_args.get("sberchel_loco_policy_by_mode")
        or {"locomotion": "zest", "static": "zest"}
    )
    if args_cli.loco_policy != "auto":
        _replay_loco_by_mode["locomotion"] = args_cli.loco_policy
        print(
            f"Warning: --loco_policy {args_cli.loco_policy} overrides the dataset's "
            f"recorded locomotion policy; the replay is an A/B, not a reproduction.",
            flush=True,
        )
    _replay_walk_decimation = int(_dataset_env_args.get("sberchel_walk_policy_decimation", 1))
    _replay_multi_policy = len(set(_replay_loco_by_mode.values())) > 1
    if _replay_multi_policy and args_cli.replay_source == "policy_loco" and num_envs > 1:
        raise ValueError(
            "policy_loco replay of a multi-policy dataset requires --num_envs 1: the "
            "per-env modes diverge, but the leg PD swap writes every env row and one "
            "Network reads env 0 only, so the other envs would run the wrong policy "
            "on the wrong gains."
        )
    if args_cli.replay_source == "actions" and (
        _dataset_env_args.get("sberchel_walk_arm_drive", "never") != "never"
    ):
        raise ValueError(
            "This dataset's arms were driven by the walk policy during locomotion, so "
            "the raw 'actions' rows still hold the frozen-snapshot IK targets rather "
            "than what was applied. Use --replay_source policy_loco (default) or "
            "processed_joint_targets, both of which carry the executed arm targets."
        )
    _replay_zest_upper = bool(_dataset_env_args.get("sberchel_zest_upper", False))
    if args_cli.replay_source == "actions" and _replay_zest_upper:
        raise ValueError(
            "This is a --zest_upper dataset: the arm joints executed zest's output, "
            "while the raw 'actions' rows hold the operator's IK intent. Use "
            "--replay_source policy_loco (default, re-runs zest from the recorded "
            "'zest_upper_ref' rows) or processed_joint_targets."
        )
    if _replay_zest_upper:
        print(
            ">>> zest-upper dataset: policy_loco replay feeds the recorded "
            "'zest_upper_ref' commander references back into zest.",
            flush=True,
        )

    # Auto-match the recorded physics step (--sim_dt_denominator sessions).
    # Datasets from before the key existed all ran the task cfg's dt.
    _recorded_sim_dt = _dataset_env_args.get("sberchel_sim_dt")
    if _recorded_sim_dt is None:
        # IsaacLab's own recorder metadata carries the recording env's dt, so a
        # dataset that lost the sberchel provenance block still knows its step
        # rate. Falling through to the task cfg instead is not a harmless
        # default: camera_fps below is derived from this dt, so a 240 Hz
        # recording would be exported (and timestamped) as 150/stride fps.
        _recorded_sim_dt = (_dataset_env_args.get("sim_args") or {}).get("dt")
        if _recorded_sim_dt is not None:
            print(f">>> sim dt: {float(_recorded_sim_dt):.6f} s — from the dataset's sim_args")
    if _recorded_sim_dt is not None and abs(float(_recorded_sim_dt) - env_cfg.sim.dt) > 1e-9:
        print(
            f">>> sim dt: {float(_recorded_sim_dt):.6f} s "
            f"(task cfg {env_cfg.sim.dt:.6f} s) — auto-matched from dataset provenance"
        )
        env_cfg.sim.dt = float(_recorded_sim_dt)

    # Auto-match the recorded self-collision plant. Datasets from before the
    # key existed were all recorded with the MuJoCo-faithful filter (it predates
    # the opt-out), so the default is True.
    if not bool(_dataset_env_args.get("sberchel_mjcf_collision_filters", True)):
        from dsynth_isaac.robots.sberchel_cfg import GREEN_CHALLENGE_SPAWN_FUNC

        env_cfg.scene.robot.spawn = env_cfg.scene.robot.spawn.replace(
            func=GREEN_CHALLENGE_SPAWN_FUNC
        )
        print(
            ">>> self-collisions: green_challenge parity (MJCF masks NOT restored) "
            "— auto-matched from dataset provenance"
        )

    recorded_enhanced_determinism = bool(_dataset_env_args.get("sberchel_enhanced_determinism", False))
    enhanced_determinism = args_cli.enhanced_determinism or recorded_enhanced_determinism
    if recorded_enhanced_determinism and not args_cli.enhanced_determinism:
        print("PhysX enhanced determinism: ON (auto-matched from dataset provenance)")
    elif args_cli.enhanced_determinism and not recorded_enhanced_determinism:
        print(
            "Warning: --enhanced_determinism forced on, but the dataset was recorded "
            "without it — the replayed physics will not match the recording exactly."
        )
    env_cfg.sim.physx.enable_enhanced_determinism = enhanced_determinism

    success_term = None
    if args_cli.validate_success_rate:
        if hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
            env_cfg.terminations.success = None
        else:
            print(
                "No success termination term was found in the environment."
                " Will not be able to mark replayed demos as successful."
            )

    env_cfg.recorders = {}
    env_cfg.terminations = {}

    # Match green_challenge rendering: it re-enables translucency on top of the
    # "balanced" preset (which disables it) and leaves antialiasing_mode unset
    # so the preset decides AA.
    env_cfg.sim.render.enable_translucency = True
    env_cfg.sim.render.antialiasing_mode = None

    selected_cameras = parse_camera_names(args_cli.cameras) if camera_augment else ()
    if camera_augment and not selected_cameras:
        raise ValueError("--output_dataset_file requires at least one camera in --cameras.")
    frame_stride = int(args_cli.frame_stride)
    if frame_stride < 1:
        raise ValueError(f"--frame_stride must be >= 1, got {frame_stride}.")
    camera_size = parse_camera_size(args_cli.camera_size)
    use_gc_pipeline = args_cli.camera_pipeline == "green_challenge"
    # Robot cameras that are not exported only cost render time; strip them
    # (all of them when not augmenting -- viewport --video does not need them).
    strip_camera_configs(env_cfg, keep=selected_cameras)
    # green_challenge pipeline masks+resizes after capture, so render native.
    apply_sberchel_camera_size(env_cfg, None if use_gc_pipeline else camera_size, selected_cameras)
    augment_img_keys: list[str] = []
    frame_postprocess: dict[str, object] = {}
    if camera_augment:
        camera_masks = load_sberchel_camera_masks(selected_cameras) if use_gc_pipeline else {}
        for camera in selected_cameras:
            scene_attr, img_key = SBERCHEL_CAMERA_OBS_KEYS[camera]
            cam_cfg = getattr(env_cfg.scene, scene_attr, None)
            if cam_cfg is None:
                raise ValueError(
                    f"Environment has no {scene_attr}; cannot augment demos with {camera} RGB."
                )
            # One fresh RGB frame per env step (default cfg is ~15 Hz).
            cam_cfg.update_period = 0.0
            augment_img_keys.append(img_key)
            if use_gc_pipeline:
                frame_postprocess[img_key] = make_green_challenge_frame_postprocess(
                    camera_masks[camera], camera_size
                )

    # Static third-person visualization camera. Added after strip_camera_configs
    # so it survives the robot-camera stripping; render-only (no collider, no
    # obs term), so replayed physics and the robot-camera frames are unchanged.
    third_person = bool(args_cli.third_person)
    tp_eye: tuple[float, ...] = ()
    tp_lookat: tuple[float, ...] = ()
    tp_size: tuple[int, int] = (0, 0)
    if third_person:
        if not camera_augment:
            raise ValueError(
                "--third_person stores obs/third_person_img in the camera export; "
                "it requires --output_dataset_file."
            )
        tp_eye, tp_lookat = parse_third_person_pose(args_cli.third_person_pose)
        parsed_tp_size = parse_camera_size(args_cli.third_person_size)
        if parsed_tp_size is None:
            raise ValueError("--third_person_size must be 'WxH' (e.g. 640x360), not 'native'.")
        tp_size = parsed_tp_size
        env_cfg.scene.third_person_camera = make_third_person_camera_cfg(tp_size)
    # Keys written to the output HDF5: robot cameras plus the optional
    # third-person view (the latter is excluded from camera_keys env_args).
    export_img_keys = list(augment_img_keys) + ([THIRD_PERSON_IMG_KEY] if third_person else [])

    # if args_cli.video:
    #     env_cfg.viewer.origin_type = "asset_root"
    #     env_cfg.viewer.asset_name = "robot"
    #     env_cfg.viewer.eye = (1.0, 1.0, 2.0)
    #     env_cfg.viewer.lookat = (0.4, 0.0, 1.6)
    # else:
    env_cfg.viewer.eye = (4, 1.0, 2.5)
    env_cfg.viewer.lookat = (9.5, 6.0, 0.0)

    throttle_render = (
        camera_augment and args_cli.headless and not args_cli.video and not args_cli.per_step_render
    )
    if throttle_render:
        # The capture path renders explicitly at every --frame_stride step (after
        # state-sync injection), so per-step renders in between only burn RTX
        # time. frame_stride (not a huge sentinel) keeps rendering_dt sane —
        # SimulationContext derives rendering_dt = dt * render_interval.
        env_cfg.sim.render_interval = frame_stride
        print(
            f"Render throttle: rendering every {frame_stride} steps instead of every step "
            "(--per_step_render restores old behavior)"
        )

    render_mode = "rgb_array" if args_cli.video else None
    gym_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    video_env: gym.Env | None = None
    if args_cli.video:
        dataset_dir = os.path.dirname(os.path.abspath(args_cli.dataset_file)) or "."
        video_folder = Path(args_cli.video_dir) if args_cli.video_dir else Path(dataset_dir) / "videos"
        video_folder.mkdir(parents=True, exist_ok=True)
        name_prefix = time.strftime("%Y%m%d_%H%M%S")
        gym_env = gym.wrappers.RecordVideo(
            gym_env,
            video_folder=str(video_folder),
            episode_trigger=lambda episode_id: True,
            video_length=0,  # entire episode; MP4 written on episode boundary
            disable_logger=True,
            name_prefix=name_prefix,
        )
        video_env = gym_env
        print(f"Viewport video recording enabled -> {video_folder}")

    env = gym_env.unwrapped
    assert_dataset_matches_robot(_dataset_env_args, env.scene["robot"])
    # Lean-store recordings carry store_products rows for a subset of the fully
    # stocked replay store; resolve the row -> live-body mapping once.
    global _store_row_body_indices
    _store_row_body_indices = build_store_row_body_indices(env, args_cli.dataset_file, _dataset_env_args)
    # FrictionalActuator zeros stiffness wherever force_control=true (all loco
    # joints). The policy PD gains required for joint-position tracking are
    # restored by Network.__init__ and re-asserted on every activate. Recording
    # always constructs these; replay must too even when not stepping the policy
    # each frame.
    policies = LocoPolicySet(
        env,
        _replay_loco_by_mode,
        decimation={"walk": _replay_walk_decimation},
        device="cpu",
    )
    policies.reset_all()
    policies.reassert("locomotion")
    if policies.multi_policy:
        print(
            f">>> loco policy: locomotion={policies.type_for('locomotion')} "
            f"static={policies.type_for('static')} "
            f"(one fixed green_challenge plant; decimation={_replay_walk_decimation})",
            flush=True,
        )

    output_dataset_handler = None
    if camera_augment:
        output_dataset_handler = HDF5DatasetFileHandler()
        output_dataset_handler.create(args_cli.output_dataset_file, env_name=env_name)
        # Conversion metadata: frame k corresponds to action/state step k*stride.
        step_hz = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
        camera_fps = step_hz / frame_stride
        first_cam = getattr(env_cfg.scene, SBERCHEL_CAMERA_OBS_KEYS[selected_cameras[0]][0])
        if camera_size is not None:
            out_width, out_height = camera_size
        else:
            out_width, out_height = first_cam.width, first_cam.height
        # Carry the recording's full provenance (joint order, seeds scheme, PD
        # gains, git commit, target scheme, ...) into the output: the converter
        # depends on sberchel_joint_names, and dropping it forces a silent
        # hardcoded-joint-list fallback downstream.
        merged_env_args = {
            key: value
            for key, value in _dataset_env_args.items()
            if key not in ("env_name", "type")
        }
        merged_env_args.update(
            {
                "camera_keys": list(augment_img_keys),
                "camera_frame_stride": frame_stride,
                "camera_fps": camera_fps,
                "camera_width": out_width,
                "camera_height": out_height,
                "camera_pipeline": args_cli.camera_pipeline,
                "replay_source": args_cli.replay_source,
                "replay_state_sync": (
                    args_cli.replay_source == "processed_joint_targets"
                    and not args_cli.no_state_sync
                ),
            }
        )
        if third_person:
            # Viz-only extra view; kept out of camera_keys on purpose so nothing
            # downstream mistakes it for a robot camera.
            merged_env_args["third_person_camera"] = {
                "key": THIRD_PERSON_IMG_KEY,
                "eye": list(tp_eye),
                "lookat": list(tp_lookat),
                "width": tp_size[0],
                "height": tp_size[1],
            }
        output_dataset_handler.add_env_args(merged_env_args)
        # The output carries the recording's store_products rows verbatim, so
        # its sidecar (recorded prim order) must travel with it too.
        _src_sidecar = Path(args_cli.dataset_file).with_suffix(".store_prims.json")
        if _src_sidecar.exists():
            _out_sidecar = Path(args_cli.output_dataset_file).with_suffix(".store_prims.json")
            _out_sidecar.write_text(_src_sidecar.read_text())
            print(f"Camera augment: store product prim order -> {_out_sidecar}")
        keys_str = ", ".join(f"obs/{key}" for key in export_img_keys)
        print(
            f"Camera augment: writing {keys_str} "
            f"({out_width}x{out_height} @ {camera_fps:g} fps, stride {frame_stride}, "
            f"pipeline {args_cli.camera_pipeline}) -> {args_cli.output_dataset_file}"
        )

    if third_person:
        # One-time static pose per env (env-local coords + env origin); nothing
        # in reset/reset_to touches this prim afterwards.
        origins = env.scene.env_origins
        env.scene["third_person_camera"].set_world_poses_from_view(
            eyes=origins + torch.tensor(tp_eye, device=origins.device, dtype=origins.dtype),
            targets=origins + torch.tensor(tp_lookat, device=origins.device, dtype=origins.dtype),
        )
        print(
            f"Third-person camera: static, eye={tp_eye} lookat={tp_lookat}, "
            f"{tp_size[0]}x{tp_size[1]} -> obs/{THIRD_PERSON_IMG_KEY}"
        )

    state_sync_enabled = (
        args_cli.replay_source == "processed_joint_targets" and not args_cli.no_state_sync
    )
    _policy_desc = (
        f"locomotion={_replay_loco_by_mode.get('locomotion', 'zest')} "
        f"static={_replay_loco_by_mode.get('static', 'zest')}"
    )
    if args_cli.replay_source == "policy_loco":
        print(f"Replay source: policy_loco ({_policy_desc} + recorded upper-body joint targets)")
        print(f"Leg policy: active (device=cpu; {_policy_desc}); PD restored per policy")
        print("State sync: disabled (closed-loop leg policy)")
    elif args_cli.replay_source == "processed_joint_targets":
        print("Replay source: processed_joint_targets (direct joint PD targets from recording)")
        print("Leg policy: inactive (constructed only to restore locomotion PD gains)")
        if state_sync_enabled:
            print("State sync: enabled (inject previous recorded scene state before each step)")
        else:
            print("State sync: disabled (open-loop replay)")
    else:
        print("Replay source: actions (raw Pink IK + leg actions; may diverge from recording)")
        print("Leg policy: inactive (constructed only to restore locomotion PD gains)")
        print("State sync: disabled")

    fast_export = bool(args_cli.fast_export)
    if fast_export:
        if not camera_augment:
            raise ValueError("--fast_export requires --output_dataset_file (camera export).")
        if not state_sync_enabled:
            raise ValueError(
                "--fast_export requires --replay_source processed_joint_targets without --no_state_sync."
            )
        if args_cli.video or args_cli.validate_states or args_cli.validate_success_rate:
            raise ValueError(
                "--fast_export is incompatible with --video, --validate_states and --validate_success_rate."
            )
        print(
            "Fast export: physics stepping skipped; recorded states injected exactly "
            "(all products pinned, velocities zeroed for the settle step) and rendered "
            f"only at capture steps (every {frame_stride} steps)"
        )

    teleop_interface = None
    if not args_cli.headless:
        teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
        teleop_interface.add_callback("N", play_cb)
        teleop_interface.add_callback("B", pause_cb)
        print('Press "B" to pause and "N" to resume the replayed actions.')

    state_validation_enabled = args_cli.validate_states and num_envs == 1
    if args_cli.validate_states and num_envs > 1:
        print("Warning: State validation is only supported with a single environment. Skipping state validation.")

    validate_report_file = None
    if state_validation_enabled:
        validate_report_file = open(args_cli.validate_report, "w", encoding="utf-8")
        validate_report_file.write(
            f"Sberchel XR replay validation report\n"
            f"dataset: {args_cli.dataset_file}\n"
            f"task: {args_cli.task}\n"
            f"device: {args_cli.device}\n"
            f"replay_source: {args_cli.replay_source}\n"
            f"state_sync: {state_sync_enabled}\n"
            f"{'=' * 72}\n"
        )
        validate_report_file.flush()
        print(f"State validation output -> {args_cli.validate_report}")

    if video_env is not None:
        video_env.reset()
    else:
        env.reset()
    idle_action = prepare_sberchel_xr_episode_replay(env, env_cfg, num_envs)
    if teleop_interface is not None:
        teleop_interface.reset()

    # h5py yields group names alphabetically (demo_10 sorts before demo_2), so a
    # file with >10 demos would scramble the index<->name mapping and the output
    # demo names would no longer match the input's. Sort numerically.
    episode_names = sorted(
        dataset_file_handler.get_episode_names(),
        key=lambda name: (
            int(name.rsplit("_", 1)[-1]) if name.rsplit("_", 1)[-1].isdigit() else float("inf"),
            name,
        ),
    )
    # Overall progress across every step of every episode queued for replay.
    with h5py.File(args_cli.dataset_file, "r") as _f:
        total_steps = sum(
            _f[f"data/{episode_names[i]}/actions"].shape[0]
            for i in episode_indices_to_replay
            if i < episode_count
        )
    step_progress = make_step_progress(total_steps)
    replayed_episode_count = 0
    recorded_episode_count = 0
    augmented_episode_count = 0
    current_episode_indices = [None] * num_envs
    failed_demo_ids = []
    validation_matched_count = 0
    validation_mismatched_count = 0
    first_mismatch_step: int | None = None
    first_fall_step: int | None = None
    initial_root_z: float | None = None
    episode_step_counters = [0] * num_envs
    # Per-demo state-validation verdicts, stamped onto output demo attrs so a
    # bad replay is visible downstream instead of only in a scrolled-away log.
    validation_mismatches_by_demo: dict[int, int] = {}
    policy_loco_steps: list[PolicyLocoStep | None] = [None] * num_envs
    export_episode_snapshots: list[EpisodeData | None] = [None] * num_envs
    camera_frame_buffers: list[dict[str, list[torch.Tensor]]] = [
        {key: [] for key in export_img_keys} for _ in range(num_envs)
    ]

    def _write_validation(line: str) -> None:
        if validate_report_file is not None:
            validate_report_file.write(line + "\n")
            validate_report_file.flush()

    def _flush_camera_augment(env_id: int, *, allow_incomplete: bool = False) -> None:
        nonlocal augmented_episode_count
        if not camera_augment or output_dataset_handler is None:
            return
        frames_by_key = camera_frame_buffers[env_id]
        snapshot = export_episode_snapshots[env_id]
        demo_id = current_episode_indices[env_id]
        extra_attrs: dict = {}
        if demo_id is not None and demo_id < len(episode_names):
            extra_attrs.update(source_demo_attrs.get(episode_names[demo_id], {}))
        if demo_id is not None and demo_id in validation_mismatches_by_demo:
            extra_attrs["replay_validation_mismatches"] = int(
                validation_mismatches_by_demo[demo_id]
            )
        wrote = _maybe_flush_incomplete_ok(
            output_dataset_handler,
            snapshot,
            frames_by_key,
            demo_id,
            allow_incomplete=allow_incomplete,
            frame_stride=frame_stride,
            extra_attrs=extra_attrs or None,
        )
        if wrote:
            augmented_episode_count += 1
            frame_count = len(frames_by_key[export_img_keys[0]]) if export_img_keys else 0
            print(
                f"Wrote augmented episode demo_{demo_id} "
                f"({frame_count} frames x {len(export_img_keys)} cameras) "
                f"-> {args_cli.output_dataset_file}"
            )
        camera_frame_buffers[env_id] = {key: [] for key in export_img_keys}
        export_episode_snapshots[env_id] = None

    # NB: comma, not `and` — `A and B` evaluates to B alone, which would let
    # Ctrl+C skip the post-loop salvage flush and lose the in-progress episode.
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
            first_loop = True
            has_next_action = True
            episode_ended = [False] * num_envs
            while has_next_action:
                # Idle filler for envs without a command this step (multi-env only).
                actions = idle_action.clone()
                has_next_action = False
                envs_with_step: list[int] = []
                for env_id in range(num_envs):
                    env_next_action = None
                    if args_cli.replay_source == "policy_loco":
                        policy_loco_steps[env_id] = get_next_policy_loco_step(env_episode_data_map[env_id])
                    else:
                        env_next_action = get_next_replay_command(
                            env_episode_data_map[env_id], args_cli.replay_source
                        )
                    step_exhausted = (
                        policy_loco_steps[env_id] is None
                        if args_cli.replay_source == "policy_loco"
                        else env_next_action is None
                    )
                    if step_exhausted:
                        if (
                            success_term is not None
                            and current_episode_indices[env_id] is not None
                            and not episode_ended[env_id]
                        ):
                            if bool(success_term.func(env, **success_term.params)[env_id]):
                                recorded_episode_count += 1
                                plural_trailing_s = "s" if recorded_episode_count > 1 else ""
                                print(
                                    f"Successfully replayed {recorded_episode_count} episode{plural_trailing_s} out"
                                    f" of {replayed_episode_count} demos."
                                )
                            elif (
                                current_episode_indices[env_id] is not None
                                and current_episode_indices[env_id] not in failed_demo_ids
                            ):
                                failed_demo_ids.append(current_episode_indices[env_id])
                            episode_ended[env_id] = True

                        # Finished episode: export camera frames before loading the next.
                        _flush_camera_augment(env_id)
                        stop_viewport_episode_record(video_env)

                        next_episode_index = None
                        while episode_indices_to_replay:
                            next_episode_index = episode_indices_to_replay.pop(0)
                            if next_episode_index < episode_count:
                                episode_ended[env_id] = False
                                break
                            next_episode_index = None

                        if next_episode_index is not None:
                            replayed_episode_count += 1
                            current_episode_indices[env_id] = next_episode_index
                            print(f"{replayed_episode_count:4}: Loading #{next_episode_index} episode to env_{env_id}")
                            episode_data = dataset_file_handler.load_episode(
                                episode_names[next_episode_index], env.device
                            )
                            if "initial_state" in episode_data.data:
                                episode_data.data["initial_state"] = _squeeze_extra_state_dims(
                                    episode_data.data["initial_state"]
                                )
                            env_episode_data_map[env_id] = episode_data
                            if camera_augment:
                                export_episode_snapshots[env_id] = clone_episode_data(episode_data)
                                camera_frame_buffers[env_id] = {key: [] for key in export_img_keys}
                            episode_step_counters[env_id] = 0
                            initial_state = episode_data.get_initial_state()
                            # Seed reproduces the recorded first-row jitter draw
                            # (reset_to re-rolls it otherwise; the per-step sync
                            # tolerance of 2 cm / 10 deg would let a fresh roll
                            # persist in every rendered frame).
                            episode_seed = (
                                int(episode_data.seed) if episode_data.seed is not None else None
                            )
                            env.reset_to(
                                initial_state,
                                torch.tensor([env_id], device=env.device),
                                seed=episode_seed,
                                is_relative=True,
                            )
                            idle_action = prepare_sberchel_xr_episode_replay(env, env_cfg, num_envs)
                            if num_envs == 1 and camera_augment:
                                # Belt-and-braces on top of the seed: pin every
                                # store product to recorded row 0 (post-step-0
                                # poses, <1 mm from the true initial placement).
                                store = episode_data.data.get("store_products")
                                restore = getattr(env.scene, "restore_store_rigid_body_state", None)
                                if store is not None and restore is not None:
                                    restore(
                                        env.sim,
                                        {
                                            "transforms": store["transforms"][0],
                                            "velocities": store["velocities"][0],
                                        },
                                        only_moved=False,
                                        body_indices=_store_row_body_indices,
                                    )
                                # Renders read transforms from Fabric, which only
                                # syncs from PhysX during a physics step: without
                                # this settle step the first capture is a stale
                                # render of the previous episode's end scene. The
                                # robot's PD targets hold the injected pose.
                                env.scene.write_data_to_sim()
                                env.sim.step(render=False)
                                env.scene.update(dt=env.physics_dt)
                            # Refresh filler row after reset so this env does not keep a stale idle clone.
                            actions[env_id] = idle_action[env_id]
                            # Every net, and clear the active latch, so the first
                            # step of the episode re-runs the full activate
                            # sequence the recorder ran at its episode start.
                            policies.reset_all()
                            policies.reassert("locomotion")
                            initial_root_z = env.scene["robot"].data.root_pos_w[env_id, 2].item()
                            if state_validation_enabled:
                                validation_mismatches_by_demo[next_episode_index] = 0
                            _write_validation(f"Episode {next_episode_index}: loaded (initial root_z={initial_root_z:.4f})")
                            begin_viewport_episode_record(video_env)
                            # Fetch action 0 immediately (same as Isaac Lab replay_demos.py).
                            if args_cli.replay_source == "policy_loco":
                                policy_loco_steps[env_id] = get_next_policy_loco_step(
                                    env_episode_data_map[env_id]
                                )
                            else:
                                env_next_action = get_next_replay_command(
                                    env_episode_data_map[env_id], args_cli.replay_source
                                )
                        else:
                            continue

                    # Apply fetched command for this env (ongoing or just-loaded episode).
                    if args_cli.replay_source == "policy_loco":
                        if policy_loco_steps[env_id] is not None:
                            has_next_action = True
                            envs_with_step.append(env_id)
                    elif env_next_action is not None:
                        has_next_action = True
                        envs_with_step.append(env_id)
                        _write_action_row(
                            actions, env_id, env_next_action, env, args_cli.replay_source
                        )
                if first_loop:
                    first_loop = False
                else:
                    while is_paused:
                        env.sim.render()
                if not envs_with_step:
                    continue
                if state_sync_enabled:
                    for env_id in range(num_envs):
                        step_idx = episode_step_counters[env_id]
                        if fast_export and step_idx % frame_stride != 0:
                            # Fast export: this step's scene state is never
                            # rendered or exported, so skip the injection.
                            continue
                        if step_idx > 0:
                            episode_data = env_episode_data_map[env_id]
                            prev_state = episode_data.get_state(step_idx - 1)
                            if prev_state is None:
                                raise ValueError(
                                    f"Episode is missing recorded states required for state sync at step {step_idx}."
                                )
                            # Camera export needs exact frames: pin every product
                            # to the recorded pose (no drift tolerance), and in
                            # fast export zero all velocities so the settle step
                            # holds the injected pose instead of integrating it.
                            sync_scene_to_recorded_state(
                                env, prev_state, env_id, zero_velocities=fast_export
                            )
                            sync_store_products_to_recorded(
                                env,
                                episode_data,
                                step_idx - 1,
                                num_envs,
                                only_moved=not camera_augment,
                                zero_velocities=fast_export,
                            )
                            if fast_export:
                                # Renders read transforms from Fabric, which only
                                # syncs from PhysX during a physics step. One step
                                # (PD targets hold the injected pose) makes the
                                # injected state visible to the capture render.
                                env.sim.step(render=False)
                                env.scene.update(dt=env.physics_dt)
                if camera_augment and envs_with_step:
                    # Pre-step capture: pair frame k with action k the way teleop
                    # and LeRobot do (the state the operator saw when issuing
                    # action k). With state sync this renders the exact recorded
                    # state k-1 injected above. Store only the stride grid
                    # (action indices 0, stride, 2*stride, ...).
                    stride_envs = [
                        env_id
                        for env_id in envs_with_step
                        if episode_step_counters[env_id] % frame_stride == 0
                    ]
                    if stride_envs:
                        # Re-render so annotators reflect the state-synced scene.
                        env.sim.render()
                        env.obs_buf = env.observation_manager.compute()
                        rgb_obs = env.obs_buf.get("rgb_camera")
                        missing = [
                            key for key in augment_img_keys if rgb_obs is None or key not in rgb_obs
                        ]
                        if missing:
                            raise RuntimeError(
                                f"Missing observations.rgb_camera terms {missing}; "
                                "ensure --enable_cameras and the demo env camera cfg are active."
                            )
                        for img_key in augment_img_keys:
                            img_batch = rgb_obs[img_key]
                            postprocess = frame_postprocess.get(img_key)
                            for env_id in stride_envs:
                                # Keep frames on CPU — buffering a full episode of RGB on CUDA OOMs.
                                # clone(): on a CPU sim device .to("cpu") is a no-op, and the
                                # sensor buffer is overwritten in place each render — an aliased
                                # frame would silently become the episode's last frame.
                                frame = (
                                    img_batch[env_id].detach().to("cpu", non_blocking=False).contiguous().clone()
                                )
                                if frame.is_floating_point():
                                    frame = frame.clamp(0, 255).to(torch.uint8)
                                if postprocess is not None:
                                    frame = postprocess(frame)
                                camera_frame_buffers[env_id][img_key].append(frame)
                        if third_person:
                            # Read the sensor directly (no obs term keeps it out
                            # of the observation pipeline entirely).
                            tp_batch = env.scene["third_person_camera"].data.output["rgb"]
                            for env_id in stride_envs:
                                # clone() for the same reason as above: this aliases the camera's
                                # persistent output buffer on CPU sim devices.
                                frame = (
                                    tp_batch[env_id, ..., :3]
                                    .detach()
                                    .to("cpu", non_blocking=False)
                                    .contiguous()
                                    .clone()
                                )
                                if frame.is_floating_point():
                                    frame = frame.clamp(0, 255).to(torch.uint8)
                                camera_frame_buffers[env_id][THIRD_PERSON_IMG_KEY].append(frame)
                if fast_export:
                    # Physics result is overwritten by the next state-sync
                    # injection, so skip stepping; keep env counters advancing
                    # for anything reading episode progress.
                    env.episode_length_buf += 1
                    env.common_step_counter += 1
                elif args_cli.replay_source == "policy_loco":
                    legs_dim, neck_dim, upper_dim, torso_dim = _processed_term_dims(env)
                    legs_batch = torch.zeros(num_envs, legs_dim, device=env.device, dtype=torch.float32)
                    neck_batch = torch.zeros(
                        num_envs, max(neck_dim, NECK_ACTION_DIM), device=env.device, dtype=torch.float32
                    )
                    if neck_dim > 0:
                        neck_batch = neck_batch[:, :neck_dim]
                    upper_batch = torch.zeros(num_envs, upper_dim, device=env.device, dtype=torch.float32)
                    torso_batch = torch.zeros(
                        num_envs, max(torso_dim, 1), device=env.device, dtype=torch.float32
                    )
                    if torso_dim > 0:
                        torso_batch = torso_batch[:, :torso_dim]
                    for env_id in range(num_envs):
                        step_data = policy_loco_steps[env_id]
                        if step_data is None:
                            continue
                        loco_cmd = step_data.loco_command
                        processed = step_data.processed_actions
                        if loco_cmd.ndim == 0:
                            loco_values = loco_cmd.reshape(-1)
                        else:
                            loco_values = loco_cmd.reshape(-1)[:3]
                        vx, vy, wz = (float(v) for v in loco_values.tolist())
                        # Same selector, decimation and reset-on-swap rule the
                        # recorder used, driven off the recorded mode flag: row k
                        # of loco_commands and teleop_state describe the same step.
                        mode = "locomotion" if step_data.is_locomotion else "static"
                        # --zest_upper datasets: replay the recorded commander
                        # reference too, or the re-run zest sees a standing-pose
                        # ref where the recorder fed live arms and its leg
                        # output stops matching the recording.
                        upper_ref = None
                        if step_data.zest_upper_ref is not None:
                            upper_ref = [
                                float(v) for v in step_data.zest_upper_ref.reshape(-1)[:13].tolist()
                            ]
                        legs = policies.step(mode, vx, vy, wz, upper_ref=upper_ref)[:LEGS_ACTION_DIM]
                        legs_batch[env_id] = torch.as_tensor(legs, dtype=torch.float32, device=env.device)
                        _legs_t, neck_t, upper_t, torso_t = _split_processed_joint_targets(
                            processed.to(device=env.device, dtype=torch.float32), env
                        )
                        if neck_dim > 0:
                            neck_batch[env_id] = neck_t
                        if torso_dim > 0:
                            # Recorded torso_yaw processed target: the walk policy's
                            # output the recorder actually applied (zero on legacy
                            # rows and zest segments).
                            torso_batch[env_id] = torso_t[:torso_dim]
                        # Policy overrides legs; use recorded neck+upper+torso processed targets.
                        if upper_t.numel() != upper_dim:
                            padded = torch.zeros(upper_dim, device=env.device, dtype=torch.float32)
                            padded[: min(upper_t.numel(), upper_dim)] = upper_t[: min(upper_t.numel(), upper_dim)]
                            upper_batch[env_id] = padded
                        else:
                            upper_batch[env_id] = upper_t
                    step_recorded_joint_targets(
                        env,
                        legs_batch,
                        upper_batch,
                        neck_batch if neck_dim > 0 else None,
                        torso_batch if torso_dim > 0 else None,
                    )
                elif args_cli.replay_source == "processed_joint_targets":
                    legs_dim, neck_dim, upper_dim, torso_dim = _processed_term_dims(env)
                    expected = legs_dim + neck_dim + upper_dim + torso_dim
                    step_processed_joint_targets(env, actions[:, :expected])
                else:
                    env.step(actions)
                if video_env is not None:
                    capture_viewport_frame(video_env, env)
                for env_id in range(num_envs):
                    if env_episode_data_map[env_id].next_action_index > episode_step_counters[env_id]:
                        episode_step_counters[env_id] += 1
                step_progress.update(len(envs_with_step))

                robot = env.scene["robot"]
                step_index = env_episode_data_map[0].next_state_index - 1 if state_validation_enabled else -1
                root_z = robot.data.root_pos_w[0, 2].item()
                if state_validation_enabled and initial_root_z is not None and first_fall_step is None:
                    if root_z < initial_root_z - 0.25:
                        first_fall_step = max(step_index, 0)

                if state_validation_enabled:
                    state_from_dataset = env_episode_data_map[0].get_next_state()
                    if state_from_dataset is not None:
                        action_index = env_episode_data_map[0].next_state_index - 1
                        current_runtime_state = env.scene.get_state(is_relative=True)
                        states_matched, comparison_log, max_abs_error = compare_states(
                            state_from_dataset, current_runtime_state, 0
                        )
                        if states_matched:
                            validation_matched_count += 1
                            line = f"step {action_index:4d}: matched (max_err={max_abs_error:.6f}, root_z={root_z:.4f})"
                        else:
                            validation_mismatched_count += 1
                            _demo = current_episode_indices[0]
                            if _demo is not None:
                                validation_mismatches_by_demo[_demo] = (
                                    validation_mismatches_by_demo.get(_demo, 0) + 1
                                )
                            if first_mismatch_step is None:
                                first_mismatch_step = action_index
                            line = (
                                f"step {action_index:4d}: MISMATCH (max_err={max_abs_error:.6f}, root_z={root_z:.4f})"
                            )
                        _write_validation(line)
                        if not states_matched:
                            _write_validation(comparison_log.rstrip())
            break

    step_progress.close()

    # Flush any episode that finished via interrupt / loop exit without a next load.
    for env_id in range(num_envs):
        _flush_camera_augment(env_id, allow_incomplete=True)

    if validate_report_file is not None:
        _write_validation(f"{'=' * 72}")
        _write_validation(
            f"Summary: matched={validation_matched_count} mismatched={validation_mismatched_count} "
            f"first_mismatch_step={first_mismatch_step} first_fall_step={first_fall_step}"
        )
        validate_report_file.close()
        print(
            f"Validation done: matched={validation_matched_count} mismatched={validation_mismatched_count} "
            f"(report: {args_cli.validate_report})"
        )

    plural_trailing_s = "s" if replayed_episode_count > 1 else ""
    print(f"Finished replaying {replayed_episode_count} episode{plural_trailing_s}.")
    if camera_augment:
        print(f"Augmented episodes written: {augmented_episode_count} -> {args_cli.output_dataset_file}")

    if success_term is not None:
        print(f"Successfully replayed: {recorded_episode_count}/{replayed_episode_count}")
        if failed_demo_ids:
            print(f"\nFailed demo IDs ({len(failed_demo_ids)} total):")
            print(f"  {sorted(failed_demo_ids)}")

    # Fail loudly instead of exiting 0 with a quietly incomplete/diverged
    # output — a bad batch must not flow into conversion unnoticed.
    exit_code = 0
    if camera_augment and augmented_episode_count < len(requested_episode_indices):
        print(
            f"[ERROR] exported {augmented_episode_count}/{len(requested_episode_indices)} "
            "augmented episodes — the output is INCOMPLETE; do not convert it."
        )
        exit_code = 1
    if state_validation_enabled and validation_mismatched_count:
        print(
            f"[ERROR] state validation found {validation_mismatched_count} mismatched step(s) "
            f"(first at step {first_mismatch_step}) — inspect the report before using the output."
        )
        exit_code = 1

    if output_dataset_handler is not None:
        output_dataset_handler.close()
    dataset_file_handler.close()
    stop_viewport_episode_record(video_env)
    if video_env is not None:
        video_env.close()
    else:
        env.close()
    return exit_code


if __name__ == "__main__":
    _exit_code = main()
    simulation_app.close()
    if _exit_code:
        raise SystemExit(_exit_code)
