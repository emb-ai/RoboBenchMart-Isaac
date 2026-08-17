# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Replay Sberchel XR demonstrations recorded by ``sberchel_kb_record_demos.py`` / record scripts.

Supports the 40D action layout ``[legs(12), neck(2), upper_body_ik(26)]`` with headset
neck targets. Legacy 38D demos are padded with zero neck where needed.

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
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_DEFAULT_GC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "green_challenge"))
_GC = os.environ.get("GREEN_CHALLENGE_PATH", _DEFAULT_GC)
_gc_parent = os.path.dirname(os.path.abspath(_GC))
if _gc_parent not in sys.path:
    sys.path.insert(0, _gc_parent)

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
        "Replay command source. 'policy_loco' (default) runs the zest leg policy from recorded "
        "loco_commands and applies recorded upper-body joint targets. "
        "'processed_joint_targets' applies recorded post-IK joint targets directly. "
        "'actions' feeds raw 40D Pink+leg+neck actions through the env (re-solves IK)."
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

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher_args = vars(args_cli)
# Viewport rgb_array / RecordVideo needs camera rendering enabled.
if args_cli.video:
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

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import dsynth_isaac.tasks.manager_based  # noqa: F401

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from dsynth_isaac.tasks.manager_based.sberchel_base_env_cfg import (
    LEGS_ACTION_DIM,
    NECK_ACTION_DIM,
    policy_leg_targets,
)
from dsynth_isaac.robots.sberchel_cfg import apply_sberchel_xr_teleop_pd_gains
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
from green_challenge.policy.network import Network

is_paused = False

# Action layout: [legs(12), neck(2), upper_body_ik(26)] = 40D.
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
    apply_sberchel_xr_teleop_pd_gains(env.scene["robot"])
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


def get_next_policy_loco_step(episode_data: EpisodeData) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return recorded ``(loco_commands, processed_actions)`` for the next step."""
    if episode_data.is_empty():
        return None
    loco_commands = episode_data.data.get("loco_commands")
    processed_actions = episode_data.data.get("processed_actions")
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
    return loco_commands[index], processed_actions[index]


def _processed_term_dims(env) -> tuple[int, int, int]:
    """Return ``(legs_dim, neck_dim, upper_dim)`` from live action terms."""
    legs_dim = env.action_manager.get_term("legs_joint_pos")._processed_actions.shape[-1]
    upper_dim = env.action_manager.get_term("upper_body_ik")._processed_actions.shape[-1]
    if "neck_joint_pos" in env.action_manager.active_terms:
        neck_dim = env.action_manager.get_term("neck_joint_pos")._processed_actions.shape[-1]
    else:
        neck_dim = 0
    return legs_dim, neck_dim, upper_dim


def _split_processed_joint_targets(
    processed: torch.Tensor,
    env,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split processed targets into ``(legs, neck, upper)``.

    Uses live action-term processed widths. Legacy recordings without a neck slice
    get zero neck targets.
    """
    flat = processed.reshape(-1)
    legs_dim, neck_dim, upper_dim = _processed_term_dims(env)

    if neck_dim > 0 and flat.numel() == legs_dim + neck_dim + upper_dim:
        return (
            flat[:legs_dim],
            flat[legs_dim : legs_dim + neck_dim],
            flat[legs_dim + neck_dim :],
        )

    if flat.numel() == legs_dim + upper_dim:
        neck = torch.zeros(max(neck_dim, NECK_ACTION_DIM), device=flat.device, dtype=flat.dtype)
        if neck_dim > 0:
            neck = neck[:neck_dim]
        return flat[:legs_dim], neck, flat[legs_dim:]

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
    return legs, neck, upper


def step_hybrid_joint_targets(
    env,
    legs: torch.Tensor,
    upper: torch.Tensor,
    neck: torch.Tensor | None = None,
) -> None:
    """Step physics with separate leg/neck/upper joint targets, bypassing Pink IK re-solve."""
    robot = env.scene["robot"]
    legs_term = env.action_manager.get_term("legs_joint_pos")
    upper_term = env.action_manager.get_term("upper_body_ik")
    neck_term = None
    if "neck_joint_pos" in env.action_manager.active_terms:
        neck_term = env.action_manager.get_term("neck_joint_pos")

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

    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        legs_term.apply_actions()
        if neck_term is not None:
            neck_term.apply_actions()
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
    for env_id in range(processed_actions.shape[0]):
        legs, neck, upper = _split_processed_joint_targets(processed_actions[env_id], env)
        legs_list.append(legs)
        neck_list.append(neck)
        upper_list.append(upper)
    step_hybrid_joint_targets(
        env,
        torch.stack(legs_list, dim=0),
        torch.stack(upper_list, dim=0),
        torch.stack(neck_list, dim=0),
    )


def normalize_processed_to_env(processed: torch.Tensor, env) -> torch.Tensor:
    """Pad legacy processed targets with a zero neck slice when needed."""
    flat = processed.reshape(-1).to(dtype=torch.float32)
    legs_dim, neck_dim, upper_dim = _processed_term_dims(env)
    expected = legs_dim + neck_dim + upper_dim
    if flat.numel() == expected:
        return flat
    if neck_dim > 0 and flat.numel() == legs_dim + upper_dim:
        return torch.cat(
            [
                flat[:legs_dim],
                torch.zeros(neck_dim, device=flat.device, dtype=flat.dtype),
                flat[legs_dim:],
            ],
            dim=0,
        )
    if flat.numel() < expected:
        out = torch.zeros(expected, device=flat.device, dtype=flat.dtype)
        out[: flat.numel()] = flat
        return out
    return flat[:expected]


def normalize_raw_action_to_env(action: torch.Tensor, action_dim: int) -> torch.Tensor:
    """Pad legacy 38D raw actions with zero neck, or truncate/pad to ``action_dim``."""
    flat = action.reshape(-1).to(dtype=torch.float32)
    if flat.numel() == action_dim:
        return flat
    # Legacy [legs(12), upper(26)] -> insert zero neck.
    if flat.numel() == LEGS_ACTION_DIM + _UPPER_ACTION_DIM and action_dim == LEGS_ACTION_DIM + NECK_ACTION_DIM + _UPPER_ACTION_DIM:
        return torch.cat(
            [
                flat[:LEGS_ACTION_DIM],
                torch.zeros(NECK_ACTION_DIM, device=flat.device, dtype=flat.dtype),
                flat[LEGS_ACTION_DIM:],
            ],
            dim=0,
        )
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


def sync_scene_to_recorded_state(env, recorded_state: dict, env_id: int) -> None:
    """Inject a recorded post-step scene state without running env ``reset_to`` managers."""
    env_ids = torch.tensor([env_id], device=env.device)
    env.scene.reset_to(recorded_state, env_ids=env_ids, is_relative=True)
    apply_sberchel_xr_teleop_pd_gains(env.scene["robot"])
    env.scene.write_data_to_sim()
    env.sim.forward()


_warned_no_store_products = False


def sync_store_products_to_recorded(env, episode_data: EpisodeData, step_idx: int, num_envs: int) -> None:
    """Pin unregistered store product bodies to their recorded step state.

    Store products are plain USD rigid bodies, not scene assets, so the robot
    state sync alone leaves them on open-loop physics and a grasped item drifts
    out of the hand. No-op with a one-time warning for demos recorded before
    ``PostStepStoreProductsRecorder`` existed.
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
    restore(
        env.sim,
        {
            "transforms": store["transforms"][step_idx],
            "velocities": store["velocities"][step_idx],
        },
    )


def _clone_tensor_tree(value):
    """Deep-clone nested dicts of tensors (leave non-tensors unchanged)."""
    if isinstance(value, dict):
        return {key: _clone_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
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
) -> None:
    """Copy ``source_episode`` and append stacked ``obs/<img_key>`` frames per camera."""
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
    output_handler.write_episode(episode, demo_id=demo_id)
    output_handler.flush()


def _maybe_flush_incomplete_ok(
    output_handler: HDF5DatasetFileHandler | None,
    source_episode: EpisodeData | None,
    frames_by_key: dict[str, list[torch.Tensor]],
    demo_id: int | None,
    *,
    allow_incomplete: bool,
    frame_stride: int = 1,
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
        output_handler, source_episode, frames_by_key, demo_id=demo_id, frame_stride=frame_stride
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
        return

    episode_indices_to_replay = list(args_cli.select_episodes)
    if len(episode_indices_to_replay) == 0:
        episode_indices_to_replay = list(range(episode_count))

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

    # if args_cli.video:
    #     env_cfg.viewer.origin_type = "asset_root"
    #     env_cfg.viewer.asset_name = "robot"
    #     env_cfg.viewer.eye = (1.0, 1.0, 2.0)
    #     env_cfg.viewer.lookat = (0.4, 0.0, 1.6)
    # else:
    env_cfg.viewer.eye = (4, 1.0, 2.5)
    env_cfg.viewer.lookat = (9.5, 6.0, 0.0)

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
    # FrictionalActuator zeros stiffness wherever force_control=true (all loco
    # joints). Network.__init__ restores the policy PD gains required for
    # joint-position tracking. Recording always constructs Network; replay must
    # too even when not stepping the policy each frame.
    policy_net = Network(env, type="zest", device="cpu", closed_kinematics=False)
    policy_net.reset()

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
        output_dataset_handler.add_env_args(
            {
                "camera_keys": list(augment_img_keys),
                "camera_frame_stride": frame_stride,
                "camera_fps": camera_fps,
                "camera_width": out_width,
                "camera_height": out_height,
                "camera_pipeline": args_cli.camera_pipeline,
            }
        )
        keys_str = ", ".join(f"obs/{key}" for key in augment_img_keys)
        print(
            f"Camera augment: writing {keys_str} "
            f"({out_width}x{out_height} @ {camera_fps:g} fps, stride {frame_stride}, "
            f"pipeline {args_cli.camera_pipeline}) -> {args_cli.output_dataset_file}"
        )

    state_sync_enabled = (
        args_cli.replay_source == "processed_joint_targets" and not args_cli.no_state_sync
    )
    if args_cli.replay_source == "policy_loco":
        print("Replay source: policy_loco (zest leg policy + recorded upper-body joint targets)")
        print("Zest policy: active (device=cpu); locomotion PD restored via Network")
        print("State sync: disabled (closed-loop leg policy)")
    elif args_cli.replay_source == "processed_joint_targets":
        print("Replay source: processed_joint_targets (direct joint PD targets from recording)")
        print("Zest policy: inactive (Network used only to restore locomotion PD gains)")
        if state_sync_enabled:
            print("State sync: enabled (inject previous recorded scene state before each step)")
        else:
            print("State sync: disabled (open-loop replay)")
    else:
        print("Replay source: actions (raw Pink IK + leg actions; may diverge from recording)")
        print("Zest policy: inactive (Network used only to restore locomotion PD gains)")
        print("State sync: disabled")

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

    episode_names = list(dataset_file_handler.get_episode_names())
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
    policy_loco_steps: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_envs
    export_episode_snapshots: list[EpisodeData | None] = [None] * num_envs
    camera_frame_buffers: list[dict[str, list[torch.Tensor]]] = [
        {key: [] for key in augment_img_keys} for _ in range(num_envs)
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
        wrote = _maybe_flush_incomplete_ok(
            output_dataset_handler,
            snapshot,
            frames_by_key,
            demo_id,
            allow_incomplete=allow_incomplete,
            frame_stride=frame_stride,
        )
        if wrote:
            augmented_episode_count += 1
            frame_count = len(frames_by_key[augment_img_keys[0]]) if augment_img_keys else 0
            print(
                f"Wrote augmented episode demo_{demo_id} "
                f"({frame_count} frames x {len(augment_img_keys)} cameras) "
                f"-> {args_cli.output_dataset_file}"
            )
        camera_frame_buffers[env_id] = {key: [] for key in augment_img_keys}
        export_episode_snapshots[env_id] = None

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
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
                            env_episode_data_map[env_id] = episode_data
                            if camera_augment:
                                export_episode_snapshots[env_id] = clone_episode_data(episode_data)
                                camera_frame_buffers[env_id] = {key: [] for key in augment_img_keys}
                            episode_step_counters[env_id] = 0
                            initial_state = episode_data.get_initial_state()
                            env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)
                            idle_action = prepare_sberchel_xr_episode_replay(env, env_cfg, num_envs)
                            # Refresh filler row after reset so this env does not keep a stale idle clone.
                            actions[env_id] = idle_action[env_id]
                            if policy_net is not None:
                                policy_net.reset()
                            initial_root_z = env.scene["robot"].data.root_pos_w[env_id, 2].item()
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
                        if step_idx > 0:
                            episode_data = env_episode_data_map[env_id]
                            prev_state = episode_data.get_state(step_idx - 1)
                            if prev_state is None:
                                raise ValueError(
                                    f"Episode is missing recorded states required for state sync at step {step_idx}."
                                )
                            sync_scene_to_recorded_state(env, prev_state, env_id)
                            sync_store_products_to_recorded(env, episode_data, step_idx - 1, num_envs)
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
                                frame = img_batch[env_id].detach().to("cpu", non_blocking=False).contiguous()
                                if frame.is_floating_point():
                                    frame = frame.clamp(0, 255).to(torch.uint8)
                                if postprocess is not None:
                                    frame = postprocess(frame)
                                camera_frame_buffers[env_id][img_key].append(frame)
                if args_cli.replay_source == "policy_loco":
                    legs_dim, neck_dim, upper_dim = _processed_term_dims(env)
                    legs_batch = torch.zeros(num_envs, legs_dim, device=env.device, dtype=torch.float32)
                    neck_batch = torch.zeros(
                        num_envs, max(neck_dim, NECK_ACTION_DIM), device=env.device, dtype=torch.float32
                    )
                    if neck_dim > 0:
                        neck_batch = neck_batch[:, :neck_dim]
                    upper_batch = torch.zeros(num_envs, upper_dim, device=env.device, dtype=torch.float32)
                    for env_id in range(num_envs):
                        step_data = policy_loco_steps[env_id]
                        if step_data is None:
                            continue
                        loco_cmd, processed = step_data
                        if loco_cmd.ndim == 0:
                            loco_values = loco_cmd.reshape(-1)
                        else:
                            loco_values = loco_cmd.reshape(-1)[:3]
                        vx, vy, wz = (float(v) for v in loco_values.tolist())
                        legs = policy_leg_targets(policy_net, vx, vy, wz)
                        legs_batch[env_id] = torch.as_tensor(legs, dtype=torch.float32, device=env.device)
                        _legs_t, neck_t, upper_t = _split_processed_joint_targets(
                            processed.to(device=env.device, dtype=torch.float32), env
                        )
                        if neck_dim > 0:
                            neck_batch[env_id] = neck_t
                        # Policy overrides legs; use recorded neck+upper processed targets.
                        if upper_t.numel() != upper_dim:
                            padded = torch.zeros(upper_dim, device=env.device, dtype=torch.float32)
                            padded[: min(upper_t.numel(), upper_dim)] = upper_t[: min(upper_t.numel(), upper_dim)]
                            upper_batch[env_id] = padded
                        else:
                            upper_batch[env_id] = upper_t
                    step_hybrid_joint_targets(
                        env,
                        legs_batch,
                        upper_batch,
                        neck_batch if neck_dim > 0 else None,
                    )
                elif args_cli.replay_source == "processed_joint_targets":
                    legs_dim, neck_dim, upper_dim = _processed_term_dims(env)
                    expected = legs_dim + neck_dim + upper_dim
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

    if output_dataset_handler is not None:
        output_dataset_handler.close()
    dataset_file_handler.close()
    stop_viewport_episode_record(video_env)
    if video_env is not None:
        video_env.close()
    else:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
