#!/usr/bin/env python3
"""Convert Sberchel XR demo HDF5 files (with rendered camera obs) into a LeRobot v2.1
dataset matching the green_challenge sample layout
(e.g. ``super_robot_absolute_50demos_2026-08-08/kitchen_xr_lerobot_super_robot_*``).

Input files are produced by ``sberchel_replay_demos.py --output_dataset_file ...
--enable_cameras`` and contain, per ``data/demo_N``:

- ``obs/robot_joint_pos``      (S, 53)  pre-step measured joint positions, 150 Hz
- ``obs/head_img`` / ``obs/left_wrist_img`` / ``obs/right_wrist_img``
                               (T, H, W, 3) uint8 RGB, one frame per ``camera_frame_stride``
- ``joint_targets/joint_pos_target`` (S, 53) commanded joint positions
- ``loco_commands``            (S, 3)   teleop (vx, vy, wz) base velocity commands
- ``states/articulation/robot/root_pose`` (S, 7) base pos + wxyz quat

The output dataset frame rate equals the camera rate (default 150/5 = 30 fps):
frame k uses sim step ``k * camera_frame_stride``.

Camera frames are rotated on the way in -- head 90 deg clockwise, both wrists
180 deg (``CAMERA_ROT90_CW``), as green_challenge specified for this conversion.
Disable with ``--no_camera_rotation``.

Feature construction (reverse-engineered from the sample dataset):

- ``observation.state`` (51) = 44 named joints (sim crank joints aliased to the real
  robot's knee/ankle pitch-roll names) + base [roll, pitch, z, yaw] in the
  torso_roll/torso_pitch/torso_z/torso_yaw slots + [vx, vy, 0, 0, 0, wz] from
  loco_commands in the linear_*/angular_* slots.
- ``action`` (52) = next-frame observation.state (absolute positions; last frame
  repeats itself) + episode_progress in [0, 1].
- ``observation.robot_command_35x.position`` (35) =
  [12 default leg pose (frame-0 legs), 0, 10 arm joint targets from
  joint_pos_target, 0, 0, 0.9 (height command), 9 zeros].
  The constant slots replicate the sample dataset exactly.

Task instructions are per-episode: each demo's ``task_instruction`` HDF5 attr
(or its ``seed`` + ``sberchel_target_items`` derivation) becomes that episode's
task, with one ``tasks.jsonl`` row per distinct instruction. ``--task`` is only
a uniform fallback for files with no per-episode source.

Each episode additionally carries a 5-way SUBTASK decomposition (walk-to-shelf
/ pick / carry-to-table / place-in-container / retract-hand), with prompts from
the demo's ``subtask_instructions`` attr (or rebuilt from the target item):
per-frame ``subtask_index`` column (same index space as ``task_index``, i.e.
``tasks.jsonl``) plus per-segment rows in ``meta/subtasks.jsonl``. Subtasks 1-4
are delimited by the operator's three mode-switch presses (``teleop_state[:,0]``
flips); the place->retract boundary has no button press and is derived from
object motion alone: the step after which the carried product (the store
product with metres of displacement in ``store_products``) stays at its final
rest pose — placed in the container and no longer moved by the hand. The
squeeze channel is deliberately not used (hand-tracking demos may record zero
squeeze throughout). Demos where the split fails keep the main task as their
only subtask, with a warning.

Usage:
    python scripts/sberchel_hdf5_to_lerobot.py datasets/dex_*cam.hdf5 \
        --output /path/to/out_dataset \
        [--raw_file datasets/dex_*.hdf5] \
        [--relabel_target_items "Monster Pipeline Punch,Kagome Juice,Tomato Paste,Palmolive Shampoo"]
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Joint naming
# ---------------------------------------------------------------------------

# No hardcoded joint-order fallback: sim joint order is read from each file's
# "sberchel_joint_names" provenance. A baked-in list survives a robot-model
# change silently -- the previous one described the retired open/hybrid model,
# where the serial knee/ankle joints wore the crank names and there were no
# passive linkage dofs at all -- and a wrong order maps real joints onto the
# wrong sim columns without any error. On the closed-kinematics robot_v3_0a
# model the real-robot names below (knee_pitch, ankle_pitch/roll, wrist_pitch)
# resolve directly to the linkage output joints, so no aliasing is needed.

# 44 real joints of observation.state, in the sample dataset's order.
STATE_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint",
    "neck_yaw_joint", "neck_pitch_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "left_pinky_proximal_joint", "left_ring_proximal_joint",
    "left_middle_proximal_joint", "left_index_proximal_joint",
    "left_thumb_proximal_pitch_joint", "left_thumb_proximal_yaw_joint",
    "right_pinky_proximal_joint", "right_ring_proximal_joint",
    "right_middle_proximal_joint", "right_index_proximal_joint",
    "right_thumb_proximal_pitch_joint", "right_thumb_proximal_yaw_joint",
    "torso_roll_joint", "torso_pitch_joint", "torso_z_joint",
]

STATE_NAMES = STATE_JOINT_NAMES + [
    "torso_yaw", "linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z",
]
ACTION_NAMES = STATE_NAMES + ["episode_progress"]

# The 10 arm joints written into robot_command_35x slots 13..22 (same order as state).
ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint",
]

CMD35_NAMES = [f"robot_command_35x_{i:02d}" for i in range(35)]
CMD35_HEIGHT = 0.9  # constant height-command slot, matches the sample dataset

# Default standing leg command, cmd35 slots 0..11 (left hip p/r/y, knee, ankle
# p/r, then right). The sample dataset holds this exact constant in every frame
# — it is the balance controller's stance command, NOT the measured frame-0 leg
# pose (with start-pose randomization a mid-settle first frame would otherwise
# bake a wrong "default pose" into the whole episode).
DEFAULT_STANCE_LEGS = np.array(
    [-0.086, 0.017, -0.044, 0.299, -0.225, -0.018,
     -0.086, -0.017, 0.044, 0.299, -0.225, 0.018],
    dtype=np.float32,
)

# HDF5 obs image key -> LeRobot video key.
CAMERA_MAP = {
    "head_img": "observation.images.cam_head",
    "left_wrist_img": "observation.images.cam_left_wrist",
    "right_wrist_img": "observation.images.cam_right_wrist",
}

# Quarter-turns CLOCKWISE applied to each camera on the way into the dataset:
# head 1 (90 deg), both wrists 2 (180 deg), as specified by green_challenge for
# this conversion.
#
# The two halves of the camera question are settled in different places, and
# they must not be confused. The SIM CAMERAS carry green_challenge's own
# ``extrinsic.rotation`` (0.5, 0.5, -0.5, -0.5) verbatim -- see
# ``sberchel_camera_cfg`` -- so a rendered frame is oriented exactly as their
# evaluation stack renders it, and the fisheye masks (authored in native 960x600
# sensor pixels) land where they were drawn. The DATASET IMAGES then get the
# roll, here. Do not move this into the CameraCfg offsets: that breaks both.
CAMERA_ROT90_CW = {
    "head_img": 1,
    "left_wrist_img": 2,
    "right_wrist_img": 2,
}


def rotate_frames_cw(frames: np.ndarray, quarter_turns: int) -> np.ndarray:
    """Rotate (T, H, W, 3) frames by ``quarter_turns`` * 90 deg clockwise."""
    k = quarter_turns % 4
    if k == 0:
        return frames
    # np.rot90 turns counter-clockwise in the (H, W) plane, so negate k.
    return np.ascontiguousarray(np.rot90(frames, k=-k, axes=(1, 2)))


CHUNKS_SIZE = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quat_wxyz_to_rpy(q: np.ndarray) -> np.ndarray:
    """Batched wxyz quaternion -> (roll, pitch, yaw), ZYX convention."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - x * z), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=1)


def derive_target_index(episode_seed: int, num_items: int) -> int:
    """Pick-target index from the demo's seed attr (mirrors the record script)."""
    digest = hashlib.blake2b(f"{episode_seed}:target".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % num_items


def target_task_instruction(item: str) -> str:
    """Natural-language pick task for a target item (mirrors the record script)."""
    article = "an" if item.strip()[:1].lower() in "aeiou" else "a"
    return f"Pick {article} {item.strip()} and place it on the table."


def target_subtask_instructions(item: str | None) -> list[str]:
    """Five subtask prompts in execution order (mirrors the record script).

    ``item=None`` (no per-demo target source) falls back to a generic noun so
    the segmentation is still usable, just without item conditioning.
    """
    noun = (item or "object").strip()
    article = "an" if noun[:1].lower() in "aeiou" else "a"
    return [
        f"Go to the shelf with the {noun}.",
        f"Pick {article} {noun} from the shelf.",
        f"Carry the {noun} to the table.",
        f"Put the {noun} into the container.",
        "Move the hand out of the container and return it to the default position.",
    ]


# Skill label per subtask slot, used in subtasks.jsonl segment entries.
SUBTASK_SKILLS = ["walk", "pick", "walk", "place", "retract"]


def _attr_str(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def resolve_task_instruction(
    demo_attrs: dict,
    raw_attrs: dict,
    env_args: dict,
    raw_env_args: dict,
    relabel_items: list[str] | None,
    fallback_task: str | None,
) -> str:
    """Per-episode language instruction, most-trustworthy source first.

    A VLA is conditioned on this text at eval, so a single generic string for
    demos that target different items destroys item conditioning — hence the
    hard error when nothing per-episode is available and no --task fallback
    was given explicitly.
    """
    seed = demo_attrs.get("seed", raw_attrs.get("seed"))
    # --relabel_target_items: recompute the item from the seed with corrected
    # names (old recordings carry placeholder names in their attrs; selection
    # is seed-index based, so the positional mapping is exact).
    if relabel_items:
        if seed is None:
            raise SystemExit("--relabel_target_items needs a per-demo 'seed' attr.")
        return target_task_instruction(relabel_items[derive_target_index(int(seed), len(relabel_items))])
    instruction = _attr_str(demo_attrs.get("task_instruction")) or _attr_str(
        raw_attrs.get("task_instruction")
    )
    if instruction:
        return instruction
    items = env_args.get("sberchel_target_items") or raw_env_args.get("sberchel_target_items")
    if items and seed is not None:
        return target_task_instruction(items[derive_target_index(int(seed), len(items))])
    if fallback_task:
        return fallback_task
    raise SystemExit(
        "No per-episode task instruction found (no task_instruction attr, no "
        "seed+sberchel_target_items to derive one). Pass --raw_file with the "
        "original teleop HDF5, or --task as an explicit uniform fallback."
    )


def resolve_target_item(
    demo_attrs: dict,
    raw_attrs: dict,
    env_args: dict,
    raw_env_args: dict,
    relabel_items: list[str] | None,
) -> str | None:
    """Per-episode target item name, same source priority as the instruction."""
    seed = demo_attrs.get("seed", raw_attrs.get("seed"))
    if relabel_items:
        if seed is None:
            raise SystemExit("--relabel_target_items needs a per-demo 'seed' attr.")
        return relabel_items[derive_target_index(int(seed), len(relabel_items))]
    item = _attr_str(demo_attrs.get("target_item")) or _attr_str(raw_attrs.get("target_item"))
    if item:
        return item
    items = env_args.get("sberchel_target_items") or raw_env_args.get("sberchel_target_items")
    if items and seed is not None:
        return items[derive_target_index(int(seed), len(items))]
    return None


def resolve_subtask_instructions(
    demo_attrs: dict,
    raw_attrs: dict,
    env_args: dict,
    raw_env_args: dict,
    relabel_items: list[str] | None,
) -> list[str]:
    """Per-episode subtask prompts: stamped attr first, else rebuilt from the item.

    ``--relabel_target_items`` skips the stamped attr — it was written with the
    outdated item names the relabel exists to correct.
    """
    if not relabel_items:
        stamped = _attr_str(demo_attrs.get("subtask_instructions")) or _attr_str(
            raw_attrs.get("subtask_instructions")
        )
        if stamped:
            try:
                texts = json.loads(stamped)
            except json.JSONDecodeError:
                texts = None
            if isinstance(texts, list) and len(texts) == 5:
                return [str(t) for t in texts]
            print("[warn] malformed 'subtask_instructions' attr; rebuilding from target item")
    return target_subtask_instructions(
        resolve_target_item(demo_attrs, raw_attrs, env_args, raw_env_args, relabel_items)
    )


def joint_index_map(sim_names: list, wanted: list) -> np.ndarray:
    name_to_id = {n: i for i, n in enumerate(sim_names)}
    idx = []
    for name in wanted:
        if name not in name_to_id:
            raise KeyError(
                f"Joint '{name}' not found in the recording's sim joint names. "
                "Datasets recorded on the retired open/hybrid model name the serial "
                "leg joints after the cranks and have no passive dofs; they cannot be "
                "converted with this version."
            )
        idx.append(name_to_id[name])
    return np.asarray(idx, dtype=np.int64)


def encode_video(frames: np.ndarray, out_path: Path, fps: float) -> None:
    """Encode (T, H, W, 3) uint8 RGB frames to h264/yuv420p mp4 via ffmpeg."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, _ = frames.shape
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
        "-r", f"{fps:g}", "-i", "pipe:0",
        "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
        "-crf", "18", "-g", "30", "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.stdin.write(np.ascontiguousarray(frames).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")


def feature_stats(arr: np.ndarray) -> dict:
    """Per-dimension stats over (T, D) array, in the sample's list format."""
    a = arr.astype(np.float64)
    return {
        "min": a.min(axis=0).tolist(),
        "max": a.max(axis=0).tolist(),
        "mean": a.mean(axis=0).tolist(),
        "std": a.std(axis=0).tolist(),
        "count": [int(a.shape[0])],
    }


def image_placeholder_stats(num_frames: int) -> dict:
    """The sample dataset stores fixed placeholder stats for video features."""
    return {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[0.5]], [[0.5]], [[0.5]]],
        "std": [[[0.25]], [[0.25]], [[0.25]]],
        "count": [int(num_frames)],
    }


def video_feature(height: int, width: int, fps: float) -> dict:
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": fps if fps % 1 else int(fps),
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }


def scalar_feature(dtype: str) -> dict:
    return {"dtype": dtype, "shape": [1], "names": None}


# ---------------------------------------------------------------------------
# Subtask segmentation
# ---------------------------------------------------------------------------

# The carried product must have moved at least this far to be identified;
# jittered/nudged shelf neighbours move millimetres, the carried item metres.
SUBTASK_CARRY_MIN_DISP = 0.5
# "At rest": within this of the final pose for the remainder of the episode.
SUBTASK_SETTLE_EPS = 0.01


def detect_subtask_spans(demo: h5py.Group) -> tuple[list[tuple[int, int, int]] | None, str | None]:
    """Split a demo into subtask spans ``(start_step, end_step, prompt_slot)``.

    Slots index ``target_subtask_instructions``: 0 walk-to-shelf, 1 pick,
    2 carry-to-table, 3 place-in-container, 4 retract-hand. Subtasks 0-3 are
    delimited by the three mode switches (``teleop_state[:,0]`` flips — the
    operator's C / Create presses). The place->retract boundary has no button
    press; it is derived purely from object motion in ``store_products``: the
    step after which the carried product stays at its final rest pose for the
    remainder of the episode. Released into the container, the item drops the
    last few centimetres and stops for good; while the hand still holds it,
    hand motion keeps perturbing it — so the last settle instant marks "the
    object is placed and the hand no longer moves it", and everything after is
    the retract. Deliberately independent of the squeeze channel
    (``teleop_state[:,1]``): hand-tracking sessions can close the grip with
    finger curl alone and record zero squeeze for the whole demo.

    Returns ``(spans, problem)``: spans cover [0, S) exactly; ``problem`` is a
    human-readable reason when the canonical 5-way split degraded (spans may
    then have 4 entries, or be None for no split at all).
    """
    if "teleop_state" not in demo:
        return None, "no teleop_state stream (recorded before mode recording)"
    ts = demo["teleop_state"][:]
    S = int(ts.shape[0])
    loco = ts[:, 0] > 0.5
    flips = (np.where(np.diff(loco.astype(np.int8)) != 0)[0] + 1).tolist()
    if not bool(loco[0]) or len(flips) != 3:
        return None, (
            f"non-canonical mode sequence (starts_loco={bool(loco[0])}, "
            f"{len(flips)} switches at {flips}; expected 3)"
        )
    s0, s1, s2 = (int(v) for v in flips)

    place_spans = [(0, s0, 0), (s0, s1, 1), (s1, s2, 2), (s2, S, 3)]
    if "store_products" not in demo:
        return place_spans, (
            "no store_products stream — cannot locate the object settling; "
            "place segment runs to the episode end"
        )
    pos = demo["store_products/transforms"][:, :, :3]
    disp = np.linalg.norm(pos - pos[0], axis=2)
    target = int(disp.max(axis=0).argmax())
    if float(disp[:, target].max()) < SUBTASK_CARRY_MIN_DISP:
        return place_spans, (
            f"no store product moved >{SUBTASK_CARRY_MIN_DISP} m — cannot "
            "identify the carried item; place segment runs to the episode end"
        )
    dist_final = np.linalg.norm(pos[:, target] - pos[-1, target], axis=1)
    unsettled = np.where(dist_final > SUBTASK_SETTLE_EPS)[0]
    settle = int(unsettled[-1]) + 1 if len(unsettled) else None
    if settle is None or not (s2 < settle < S):
        return place_spans, (
            f"carried item's settle step ({settle}) is not inside the final "
            "static segment; place segment runs to the episode end"
        )
    return (
        [(0, s0, 0), (s0, s1, 1), (s1, s2, 2), (s2, settle, 3), (settle, S, 4)],
        None,
    )


def subtask_frame_spans(
    spans: list[tuple[int, int, int]], stride: int, num_frames: int
) -> list[tuple[int, int, int]]:
    """Sim-step spans -> camera-frame spans (frame k = sim step k*stride).

    A boundary at sim step b starts at the first frame whose sim step >= b.
    Clamped to the exported frame count; empty frame spans are dropped.
    """
    out = []
    for start, end, slot in spans:
        f0 = min(-(-start // stride), num_frames)
        f1 = min(-(-end // stride), num_frames)
        if f1 > f0:
            out.append((f0, f1, slot))
    if out:
        out[-1] = (out[-1][0], num_frames, out[-1][2])
    return out


# ---------------------------------------------------------------------------
# Episode conversion
# ---------------------------------------------------------------------------

def convert_episode(demo: h5py.Group, env_args: dict, state_idx: np.ndarray,
                    arm_idx: np.ndarray, rotate_cameras: bool = True) -> dict:
    stride = int(env_args.get("camera_frame_stride", 5))

    joint_pos = demo["obs/robot_joint_pos"][:]                      # (S, 53)
    joint_target = demo["joint_targets/joint_pos_target"][:]        # (S, 53)
    loco = demo["loco_commands"][:]                                 # (S, 3)
    root_pose = demo["states/articulation/robot/root_pose"][:]      # (S, 7)

    num_sim_steps = joint_pos.shape[0]
    steps = np.arange(0, num_sim_steps, stride)
    num_img = min(demo[f"obs/{k}"].shape[0] for k in CAMERA_MAP)
    T = min(len(steps), num_img)
    steps = steps[:T]

    # Row T of this extended grid supplies the last frame's action: the true
    # next sim state whenever the recording extends past the final capture
    # (it nearly always does — the sample keeps real motion there, and a
    # repeated state would write one zero-motion supervision frame).
    last_next = min(int(steps[-1]) + stride, num_sim_steps - 1)
    steps_ext = np.append(steps, last_next)                         # (T+1,)

    # observation.state at the extended grid (T+1, 51); state = first T rows.
    rpy = quat_wxyz_to_rpy(root_pose[steps_ext, 3:7])
    # The store task keeps the base near world yaw ±pi (facing the shelf), so
    # the arctan2 branch cut would inject ~2*pi jumps into the absolute
    # next-state action targets; unwrap per episode (row 0 stays in (-pi, pi]).
    yaw = np.unwrap(rpy[:, 2])
    state_ext = np.zeros((T + 1, 51), dtype=np.float32)
    state_ext[:, :41] = joint_pos[steps_ext][:, state_idx]
    state_ext[:, 41] = rpy[:, 0]                # torso_roll_joint  <- base roll
    state_ext[:, 42] = rpy[:, 1]                # torso_pitch_joint <- base pitch
    state_ext[:, 43] = root_pose[steps_ext, 2]  # torso_z_joint     <- base height
    state_ext[:, 44] = yaw                      # torso_yaw         <- base yaw
    state_ext[:, 45] = loco[steps_ext, 0]       # linear_x  (vx command)
    state_ext[:, 46] = loco[steps_ext, 1]       # linear_y  (vy command)
    state_ext[:, 50] = loco[steps_ext, 2]       # angular_z (wz command)
    state = state_ext[:T]

    # action (T, 52): absolute next-state targets + episode progress
    action = np.zeros((T, 52), dtype=np.float32)
    action[:, :51] = state_ext[1:]
    action[:, 51] = (np.arange(T) / max(T - 1, 1)).astype(np.float32)

    # observation.robot_command_35x.position (T, 35)
    cmd = np.zeros((T, 35), dtype=np.float32)
    cmd[:, :12] = DEFAULT_STANCE_LEGS
    cmd[:, 13:23] = joint_target[steps][:, arm_idx]
    cmd[:, 25] = CMD35_HEIGHT
    if np.abs(state[0, :12] - DEFAULT_STANCE_LEGS).max() > 0.2:
        print(
            "[warn] frame-0 legs deviate >0.2 rad from the default stance "
            "command — episode may start mid-motion."
        )

    videos = {
        video_key: rotate_frames_cw(
            demo[f"obs/{img_key}"][:T],
            CAMERA_ROT90_CW[img_key] if rotate_cameras else 0,
        )
        for img_key, video_key in CAMERA_MAP.items()
    }
    return {"state": state, "action": action, "cmd": cmd, "videos": videos}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="+", help="Input HDF5 files (replay outputs with camera obs).")
    parser.add_argument("--output", required=True, help="Output LeRobot dataset directory.")
    parser.add_argument("--task", default=None,
                        help="Uniform fallback instruction, used only for demos that have no "
                             "per-episode task source. Episodes normally get their instruction "
                             "from the demo's 'task_instruction' attr (or seed derivation); a "
                             "single generic text for demos with different targets destroys "
                             "the VLA's language conditioning.")
    parser.add_argument("--raw_file", default=None,
                        help="Original teleop HDF5 companion: supplies per-demo attrs "
                             "(task_instruction, seed) and env_args for inputs produced by "
                             "older replay versions that dropped them.")
    parser.add_argument("--relabel_target_items", default=None,
                        help="Comma-separated corrected item names: re-derive each demo's "
                             "target from its 'seed' attr and rebuild the instruction with "
                             "these names (selection is seed-index based, so the positional "
                             "old->new mapping is exact; for recordings made with outdated "
                             "product names).")
    parser.add_argument("--skill", default=None,
                        help="Skill label for the whole-episode action_config entry (and the "
                             "single fallback subtask when no split is possible); segmented "
                             "subtasks carry fixed skills walk/pick/walk/place/retract. "
                             "Default: first word of the episode's task instruction.")
    parser.add_argument("--no_camera_rotation", action="store_true",
                        help="Write the camera frames as rendered instead of applying "
                             "CAMERA_ROT90_CW (head 90 deg clockwise, wrists 180 deg), "
                             "the orientation green_challenge asked the dataset images "
                             "to carry. For lining a dataset up against the replay "
                             "preview videos only — a dataset converted with this flag "
                             "does not match the images their stack expects.")
    parser.add_argument("--robot_type", default="gamma")
    parser.add_argument("--only_successful", action="store_true",
                        help="Skip demos whose 'success' attribute is False.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete the output directory if it already exists.")
    args = parser.parse_args()

    out = Path(args.output)
    if out.exists():
        if not args.overwrite:
            sys.exit(f"Output {out} already exists (use --overwrite).")
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)

    relabel_items = (
        [s.strip() for s in args.relabel_target_items.split(",") if s.strip()]
        if args.relabel_target_items
        else None
    )
    raw_demo_attrs: dict[str, dict] = {}
    raw_env_args: dict = {}
    if args.raw_file:
        with h5py.File(args.raw_file, "r") as raw_f:
            raw_env_args = json.loads(raw_f["data"].attrs.get("env_args", "{}"))
            raw_demo_attrs = {name: dict(raw_f[f"data/{name}"].attrs) for name in raw_f["data"]}

    rotate_cameras = not args.no_camera_rotation
    if rotate_cameras:
        print("[cameras] clockwise quarter-turns: "
              + ", ".join(f"{CAMERA_MAP[k]}={v}" for k, v in CAMERA_ROT90_CW.items()))
    else:
        print("[cameras] rotation disabled (--no_camera_rotation)")

    state_idx = arm_idx = None
    fps = None
    # Per-video (height, width) after rotation: a quarter turn transposes a
    # non-square frame, so the head and the wrists can differ.
    video_hw: dict[str, tuple[int, int]] = {}

    episodes_meta, episodes_stats, subtasks = [], [], []
    # Main prompts and subtask prompts share one index space (tasks.jsonl):
    # per-frame 'task_index' points at a main prompt, 'subtask_index' at a
    # subtask prompt (or the main one when no split was possible).
    task_to_index: dict[str, int] = {}
    main_tasks: set[str] = set()
    ep_index = 0
    global_index = 0
    skipped = 0

    for path in args.inputs:
        with h5py.File(path, "r") as f:
            env_args = json.loads(f["data"].attrs.get("env_args", "{}"))
            sim_names = env_args.get("sberchel_joint_names") or raw_env_args.get(
                "sberchel_joint_names"
            )
            if sim_names is None:
                sys.exit(
                    f"{path}: env_args lacks 'sberchel_joint_names' (older replay "
                    "output dropped the recording provenance). Pass --raw_file with "
                    "the original teleop HDF5. There is no hardcoded fallback: a baked-in "
                    "order silently survives a robot-model change and would map every "
                    "joint of state/action onto the wrong column."
                )
            # slots 41..43 (torso roll/pitch/z) come from base pose, not sim joints
            state_idx = joint_index_map(sim_names, STATE_JOINT_NAMES[:41])
            arm_idx = joint_index_map(sim_names, ARM_JOINT_NAMES)

            file_fps = float(env_args.get("camera_fps", 30.0))
            if fps is None:
                fps = file_fps
            elif fps != file_fps:
                sys.exit(f"{path}: camera_fps {file_fps} != {fps} of earlier files.")

            demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            for demo_name in demos:
                demo = f[f"data/{demo_name}"]
                if args.only_successful and not bool(demo.attrs.get("success", False)):
                    skipped += 1
                    continue

                task = resolve_task_instruction(
                    dict(demo.attrs),
                    raw_demo_attrs.get(demo_name, {}),
                    env_args,
                    raw_env_args,
                    relabel_items,
                    args.task,
                )
                task_index = task_to_index.setdefault(task, len(task_to_index))
                main_tasks.add(task)
                skill = args.skill or task.split()[0]

                ep = convert_episode(demo, env_args, state_idx, arm_idx, rotate_cameras)
                T = ep["state"].shape[0]
                if T < 2:
                    print(f"[skip] {path}:{demo_name}: only {T} frames")
                    skipped += 1
                    continue

                # --- subtask segmentation ---
                sub_texts = resolve_subtask_instructions(
                    dict(demo.attrs),
                    raw_demo_attrs.get(demo_name, {}),
                    env_args,
                    raw_env_args,
                    relabel_items,
                )
                step_spans, sub_problem = detect_subtask_spans(demo)
                stride = int(env_args.get("camera_frame_stride", 5))
                if step_spans is None:
                    print(
                        f"[warn] {path}:{demo_name}: no subtask split — {sub_problem}; "
                        "whole episode keeps the main task as its only subtask"
                    )
                    frame_spans = [(0, T, None)]
                else:
                    if sub_problem:
                        print(f"[subtask] {path}:{demo_name}: {sub_problem}")
                    frame_spans = subtask_frame_spans(step_spans, stride, T)
                subtask_index_col = np.empty(T, dtype=np.int64)
                subtask_rows = []
                for f0, f1, slot in frame_spans:
                    text = task if slot is None else sub_texts[slot]
                    sub_idx = task_to_index.setdefault(text, len(task_to_index))
                    subtask_index_col[f0:f1] = sub_idx
                    subtask_rows.append({
                        "descriptions": [text], "end_frame": int(f1),
                        "is_mirrored": False, "is_negative_episode": False,
                        "is_reversed": False,
                        "skill": skill if slot is None else SUBTASK_SKILLS[slot],
                        "start_frame": int(f0), "uses_task_text": slot is None,
                    })

                chunk = ep_index // CHUNKS_SIZE

                # --- videos ---
                for video_key, frames in ep["videos"].items():
                    hw = (int(frames.shape[1]), int(frames.shape[2]))
                    if video_hw.setdefault(video_key, hw) != hw:
                        sys.exit(
                            f"{path}:{demo_name}: {video_key} is {hw[0]}x{hw[1]}, but "
                            f"earlier episodes are {video_hw[video_key][0]}x"
                            f"{video_hw[video_key][1]}; one dataset cannot hold both."
                        )
                    encode_video(
                        frames,
                        out / "videos" / f"chunk-{chunk:03d}" / video_key
                        / f"episode_{ep_index:06d}.mp4",
                        fps,
                    )

                # --- parquet ---
                timestamps = np.arange(T, dtype=np.float64) / fps
                df = pd.DataFrame({
                    "observation.state": list(ep["state"]),
                    "action": list(ep["action"]),
                    "observation.robot_command_35x.position": list(ep["cmd"]),
                    "timestamp": timestamps,
                    "frame_index": np.arange(T, dtype=np.int64),
                    "episode_index": np.full(T, ep_index, dtype=np.int64),
                    "episode_id": np.full(T, ep_index, dtype=np.int64),
                    "index": np.arange(global_index, global_index + T, dtype=np.int64),
                    "task_index": np.full(T, task_index, dtype=np.int64),
                    "subtask_index": subtask_index_col,
                    "task_name_id": np.zeros(T, dtype=np.int64),
                    "init_scene_text_id": np.zeros(T, dtype=np.int64),
                    "skill_id": np.ones(T, dtype=np.int64),
                    "action_text_id": np.ones(T, dtype=np.int64),
                    "loss_mask_index": np.full(T, 3, dtype=np.int64),
                    "rosbag_timestamp": timestamps.astype(np.float32),
                })
                pq_dir = out / "data" / f"chunk-{chunk:03d}"
                pq_dir.mkdir(parents=True, exist_ok=True)
                df.to_parquet(pq_dir / f"episode_{ep_index:06d}.parquet", index=False)

                # --- meta ---
                action_cfg = [{
                    "action_text": task, "end_frame": T, "is_mirrored": False,
                    "is_negative_episode": False, "is_reversed": False, "skill": skill,
                    "start_frame": 0, "uses_task_text": True,
                }]
                episodes_meta.append({
                    "action_config": action_cfg,
                    "action_valid_mask": [True] * 52,
                    "episode_id": ep_index,
                    "episode_index": ep_index,
                    "initial_scene_text": [task],
                    "is_problem_episode": False,
                    "length": T,
                    "state_valid_mask": [True] * 51,
                    "task_index": task_index,
                    "tasks": [task],
                    "variation": "simxr",
                    "variation_name": task,
                })
                subtasks.append({
                    "episode_index": ep_index,
                    "subtasks": subtask_rows,
                })
                episodes_stats.append({
                    "episode_index": ep_index,
                    "stats": {
                        "action": feature_stats(ep["action"]),
                        "observation.images.cam_head": image_placeholder_stats(T),
                        "observation.images.cam_left_wrist": image_placeholder_stats(T),
                        "observation.images.cam_right_wrist": image_placeholder_stats(T),
                        "observation.robot_command_35x.position": feature_stats(ep["cmd"]),
                        "observation.state": feature_stats(ep["state"]),
                    },
                })

                seg_desc = " ".join(
                    f"{f0}-{f1}:{'main' if slot is None else SUBTASK_SKILLS[slot]}"
                    for f0, f1, slot in frame_spans
                )
                print(f"[ep {ep_index:03d}] {Path(path).name}:{demo_name}  frames={T}"
                      f"  success={bool(demo.attrs.get('success', False))}"
                      f"  task[{task_index}]={task!r}  subtasks[{seg_desc}]")
                global_index += T
                ep_index += 1

    if ep_index == 0:
        sys.exit("No episodes converted.")

    fps_out = int(fps) if float(fps).is_integer() else fps
    features = {
        **{
            video_key: video_feature(hw[0], hw[1], fps_out)
            for video_key, hw in video_hw.items()
        },
        "observation.state": {"dtype": "float32", "shape": [51], "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": [52], "names": ACTION_NAMES},
        "observation.robot_command_35x.position": {
            "dtype": "float32", "shape": [35], "names": CMD35_NAMES},
        "timestamp": scalar_feature("float32"),
        "frame_index": scalar_feature("int64"),
        "episode_index": scalar_feature("int64"),
        "episode_id": scalar_feature("int64"),
        "index": scalar_feature("int64"),
        "task_index": scalar_feature("int64"),
        "subtask_index": scalar_feature("int64"),
        "task_name_id": scalar_feature("int64"),
        "init_scene_text_id": scalar_feature("int64"),
        "skill_id": scalar_feature("int64"),
        "action_text_id": scalar_feature("int64"),
        "loss_mask_index": scalar_feature("int64"),
        "rosbag_timestamp": scalar_feature("float32"),
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": args.robot_type,
        "total_episodes": ep_index,
        "total_frames": global_index,
        "total_tasks": len(task_to_index),
        "total_variations": len(main_tasks),
        "total_videos": ep_index * len(CAMERA_MAP),
        "total_chunks": (ep_index + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
        "chunks_size": CHUNKS_SIZE,
        "fps": fps_out,
        "splits": {"train": f"0:{ep_index}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "joint_names": {"observation.state": STATE_NAMES, "action": ACTION_NAMES},
    }

    with open(out / "meta" / "info.json", "w") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
    with open(out / "meta" / "tasks.jsonl", "w") as fh:
        for task, task_index in sorted(task_to_index.items(), key=lambda kv: kv[1]):
            fh.write(json.dumps({"task": task, "task_index": task_index}, ensure_ascii=False) + "\n")
    for fname, rows in (("episodes.jsonl", episodes_meta),
                        ("subtasks.jsonl", subtasks),
                        ("episodes_stats.jsonl", episodes_stats)):
        with open(out / "meta" / fname, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDone: {ep_index} episodes, {global_index} frames -> {out}"
          + (f" ({skipped} demos skipped)" if skipped else ""))


if __name__ == "__main__":
    main()
