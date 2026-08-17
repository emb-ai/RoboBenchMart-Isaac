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

Usage:
    python scripts/sberchel_hdf5_to_lerobot.py datasets/dex_*.hdf5 \
        --output /path/to/out_dataset --task "взять банку с полки"
"""

import argparse
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

# Isaac articulation joint order (breadth-first). Used as fallback when the input
# file's env_args lacks "sberchel_joint_names" (replay outputs drop it).
FALLBACK_SIM_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "torso_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "neck_yaw_joint", "left_hip_yaw_joint",
    "right_hip_yaw_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "neck_pitch_joint", "left_knee_crank_joint", "right_knee_crank_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_ankle_u_crank_joint",
    "right_ankle_u_crank_joint", "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_ankle_b_crank_joint", "right_ankle_b_crank_joint", "left_elbow_yaw_joint",
    "right_elbow_yaw_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint", "left_thumb_proximal_yaw_joint",
    "left_index_proximal_joint", "left_middle_proximal_joint", "left_ring_proximal_joint",
    "left_pinky_proximal_joint", "right_thumb_proximal_yaw_joint",
    "right_index_proximal_joint", "right_middle_proximal_joint",
    "right_ring_proximal_joint", "right_pinky_proximal_joint",
    "left_thumb_proximal_pitch_joint", "left_index_intermediate_joint",
    "left_middle_intermediate_joint", "left_ring_intermediate_joint",
    "left_pinky_intermediate_joint", "right_thumb_proximal_pitch_joint",
    "right_index_intermediate_joint", "right_middle_intermediate_joint",
    "right_ring_intermediate_joint", "right_pinky_intermediate_joint",
    "left_thumb_intermediate_joint", "right_thumb_intermediate_joint",
    "left_thumb_distal_joint", "right_thumb_distal_joint",
]

# The sim URDF uses crank joints where the real robot names pitch/roll joints.
SIM_JOINT_ALIASES = {
    "left_knee_pitch_joint": "left_knee_crank_joint",
    "right_knee_pitch_joint": "right_knee_crank_joint",
    "left_ankle_pitch_joint": "left_ankle_u_crank_joint",
    "right_ankle_pitch_joint": "right_ankle_u_crank_joint",
    "left_ankle_roll_joint": "left_ankle_b_crank_joint",
    "right_ankle_roll_joint": "right_ankle_b_crank_joint",
}

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

# HDF5 obs image key -> LeRobot video key.
CAMERA_MAP = {
    "head_img": "observation.images.cam_head",
    "left_wrist_img": "observation.images.cam_left_wrist",
    "right_wrist_img": "observation.images.cam_right_wrist",
}

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


def joint_index_map(sim_names: list, wanted: list) -> np.ndarray:
    name_to_id = {n: i for i, n in enumerate(sim_names)}
    idx = []
    for name in wanted:
        sim_name = name if name in name_to_id else SIM_JOINT_ALIASES.get(name)
        if sim_name is None or sim_name not in name_to_id:
            raise KeyError(f"Joint '{name}' not found in sim joint names (or aliases).")
        idx.append(name_to_id[sim_name])
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
# Episode conversion
# ---------------------------------------------------------------------------

def convert_episode(demo: h5py.Group, env_args: dict, state_idx: np.ndarray,
                    arm_idx: np.ndarray) -> dict:
    stride = int(env_args.get("camera_frame_stride", 5))

    joint_pos = demo["obs/robot_joint_pos"][:]                      # (S, 53)
    joint_target = demo["joint_targets/joint_pos_target"][:]        # (S, 53)
    loco = demo["loco_commands"][:]                                 # (S, 3)
    root_pose = demo["states/articulation/robot/root_pose"][:]      # (S, 7)

    steps = np.arange(0, joint_pos.shape[0], stride)
    num_img = min(demo[f"obs/{k}"].shape[0] for k in CAMERA_MAP)
    T = min(len(steps), num_img)
    steps = steps[:T]

    # observation.state (T, 51)
    rpy = quat_wxyz_to_rpy(root_pose[steps, 3:7])
    state = np.zeros((T, 51), dtype=np.float32)
    state[:, :41] = joint_pos[steps][:, state_idx]
    state[:, 41] = rpy[:, 0]                    # torso_roll_joint  <- base roll
    state[:, 42] = rpy[:, 1]                    # torso_pitch_joint <- base pitch
    state[:, 43] = root_pose[steps, 2]          # torso_z_joint     <- base height
    state[:, 44] = rpy[:, 2]                    # torso_yaw         <- base yaw
    state[:, 45] = loco[steps, 0]               # linear_x  (vx command)
    state[:, 46] = loco[steps, 1]               # linear_y  (vy command)
    state[:, 50] = loco[steps, 2]               # angular_z (wz command)

    # action (T, 52): absolute next-frame state + episode progress
    action = np.zeros((T, 52), dtype=np.float32)
    action[:-1, :51] = state[1:]
    action[-1, :51] = state[-1]
    action[:, 51] = (np.arange(T) / max(T - 1, 1)).astype(np.float32)

    # observation.robot_command_35x.position (T, 35)
    cmd = np.zeros((T, 35), dtype=np.float32)
    cmd[:, :12] = state[0, :12]                 # default standing leg pose
    cmd[:, 13:23] = joint_target[steps][:, arm_idx]
    cmd[:, 25] = CMD35_HEIGHT

    videos = {
        video_key: demo[f"obs/{img_key}"][:T]
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
    parser.add_argument("--task", required=True, help="Task instruction text (goes to meta/tasks.jsonl).")
    parser.add_argument("--skill", default=None,
                        help="Skill label for subtasks.jsonl. Default: first word of --task.")
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

    skill = args.skill or args.task.split()[0]
    state_idx = arm_idx = None
    fps = image_hw = None

    episodes_meta, episodes_stats, subtasks = [], [], []
    ep_index = 0
    global_index = 0
    skipped = 0

    for path in args.inputs:
        with h5py.File(path, "r") as f:
            env_args = json.loads(f["data"].attrs.get("env_args", "{}"))
            sim_names = env_args.get("sberchel_joint_names", FALLBACK_SIM_JOINT_NAMES)
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

                ep = convert_episode(demo, env_args, state_idx, arm_idx)
                T = ep["state"].shape[0]
                if T < 2:
                    print(f"[skip] {path}:{demo_name}: only {T} frames")
                    skipped += 1
                    continue

                chunk = ep_index // CHUNKS_SIZE
                if image_hw is None:
                    sample_img = next(iter(ep["videos"].values()))
                    image_hw = sample_img.shape[1:3]

                # --- videos ---
                for video_key, frames in ep["videos"].items():
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
                    "task_index": np.zeros(T, dtype=np.int64),
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
                    "action_text": args.task, "end_frame": T, "is_mirrored": False,
                    "is_negative_episode": False, "is_reversed": False, "skill": skill,
                    "start_frame": 0, "uses_task_text": True,
                }]
                episodes_meta.append({
                    "action_config": action_cfg,
                    "action_valid_mask": [True] * 52,
                    "episode_id": ep_index,
                    "episode_index": ep_index,
                    "initial_scene_text": [args.task],
                    "is_problem_episode": False,
                    "length": T,
                    "state_valid_mask": [True] * 51,
                    "task_index": 0,
                    "tasks": [args.task],
                    "variation": "simxr",
                    "variation_name": args.task,
                })
                subtasks.append({
                    "episode_index": ep_index,
                    "subtasks": [{
                        "descriptions": [args.task], "end_frame": T,
                        "is_mirrored": False, "is_negative_episode": False,
                        "is_reversed": False, "skill": skill,
                        "start_frame": 0, "uses_task_text": True,
                    }],
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

                print(f"[ep {ep_index:03d}] {Path(path).name}:{demo_name}  frames={T}"
                      f"  success={bool(demo.attrs.get('success', False))}")
                global_index += T
                ep_index += 1

    if ep_index == 0:
        sys.exit("No episodes converted.")

    h, w = int(image_hw[0]), int(image_hw[1])
    fps_out = int(fps) if float(fps).is_integer() else fps
    features = {
        "observation.images.cam_head": video_feature(h, w, fps_out),
        "observation.images.cam_left_wrist": video_feature(h, w, fps_out),
        "observation.images.cam_right_wrist": video_feature(h, w, fps_out),
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
        "total_tasks": 1,
        "total_variations": 1,
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
        fh.write(json.dumps({"task": args.task, "task_index": 0}, ensure_ascii=False) + "\n")
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
