#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""XR teleop with OpenXR landmark vs robot finger-link dumps.

Same loop as ``scripts/teleop_se3_agent.py``, plus periodic paste-friendly
tables so you can show OpenXR bone positions, robot link positions, proximal
q vs q_des, and retargeter packed finger targets side-by-side.

Example::

    ./IsaacLab/isaaclab.sh -p scripts/teleop_sberchel_xr_compare.py \\
      --task Dsynth-Fetch-Any-Item-Sberchel-Converted-Upperbody-XR-v0 \\
      --teleop_device handtracking \\
      --enable_pinocchio \\
      --print_every_n 30 \\
      --log_file /tmp/sberchel_xr_compare.log
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sberchel XR teleop with OpenXR vs robot finger dumps.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--teleop_device", type=str, default="handtracking")
parser.add_argument("--task", type=str, default="Dsynth-Fetch-Any-Item-Sberchel-Converted-Upperbody-XR-v0")
parser.add_argument("--sensitivity", type=float, default=1.0)
parser.add_argument("--enable_pinocchio", action="store_true", default=False)
parser.add_argument(
    "--print_every_n",
    type=int,
    default=30,
    help="Print a compare block every N simulation steps while teleop is active.",
)
parser.add_argument(
    "--log_file",
    type=str,
    default="",
    help="Optional path to append compare blocks (also printed to stdout).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

_use_handtracking = "handtracking" in args_cli.teleop_device.lower()
if args_cli.enable_pinocchio or _use_handtracking:
    import pinocchio  # noqa: F401

if _use_handtracking:
    try:
        import dex_retargeting  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Hand-tracking teleop requires dex-retargeting. "
            "Install with: pip install 'dex-retargeting==0.4.6'"
        ) from exc
    app_launcher_args["xr"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import sys

import gymnasium as gym
import numpy as np
import torch

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.openxr.common import HAND_JOINT_NAMES
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

sys.path.append(".")
import dsynth_isaac.tasks.manager_based  # noqa: F401

from dsynth_isaac.tasks.manager_based.sberchel_converted_env_cfg import (
    CONVERTED_SBERCHEL_HAND_JOINT_NAMES,
)

logger = logging.getLogger(__name__)

# OpenXR landmarks we care about for finger compare (subset of HAND_JOINT_NAMES).
_OPENXR_LANDMARKS = [
    "wrist",
    "thumb_metacarpal",
    "thumb_proximal",
    "thumb_distal",
    "thumb_tip",
    "index_metacarpal",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "index_tip",
    "middle_metacarpal",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "middle_tip",
    "ring_metacarpal",
    "ring_proximal",
    "ring_intermediate",
    "ring_distal",
    "ring_tip",
    "little_metacarpal",
    "little_proximal",
    "little_intermediate",
    "little_distal",
    "little_tip",
]

# Robot body name patterns (resolved via find_bodies substring / exact).
_ROBOT_FINGER_LINKS = [
    "left_hand_base_link",
    "right_hand_base_link",
    "left_thumb_proximal_yaw_link",
    "left_thumb_proximal_pitch_link",
    "left_thumb_intermediate_link",
    "left_thumb_distal_link",
    "left_index_proximal_link",
    "left_index_intermediate_link",
    "left_index_distal_link",
    "left_middle_proximal_link",
    "left_middle_intermediate_link",
    "left_middle_distal_link",
    "left_ring_proximal_link",
    "left_ring_intermediate_link",
    "left_ring_distal_link",
    "left_pinky_proximal_link",
    "left_pinky_intermediate_link",
    "left_pinky_distal_link",
    "right_thumb_proximal_yaw_link",
    "right_thumb_proximal_pitch_link",
    "right_thumb_intermediate_link",
    "right_thumb_distal_link",
    "right_index_proximal_link",
    "right_index_intermediate_link",
    "right_index_distal_link",
    "right_middle_proximal_link",
    "right_middle_intermediate_link",
    "right_middle_distal_link",
    "right_ring_proximal_link",
    "right_ring_intermediate_link",
    "right_ring_distal_link",
    "right_pinky_proximal_link",
    "right_pinky_intermediate_link",
    "right_pinky_distal_link",
]


def _fmt_xyz(xyz) -> str:
    arr = np.asarray(xyz, dtype=float).reshape(-1)[:3]
    return f"[{arr[0]:7.3f} {arr[1]:7.3f} {arr[2]:7.3f}]"


def _emit(text: str, log_path: Path | None) -> None:
    print(text, flush=True)
    if log_path is not None:
        with log_path.open("a") as f:
            f.write(text + "\n")


def _dump_openxr_hand(hand_poses: dict, side: str) -> list[str]:
    lines = [f"  OpenXR {side}:"]
    for name in _OPENXR_LANDMARKS:
        pose = hand_poses.get(name) if hand_poses else None
        if pose is None or len(pose) < 3:
            lines.append(f"    {name:24s}  MISSING")
        else:
            lines.append(f"    {name:24s}  {_fmt_xyz(pose)}")
    return lines


def _resolve_body_ids(robot, names: list[str]) -> list[tuple[str, int]]:
    """Return (requested_name, body_id) for names that exist on the articulation."""
    resolved = []
    body_names = list(robot.body_names)
    name_to_id = {n: i for i, n in enumerate(body_names)}
    for name in names:
        if name in name_to_id:
            resolved.append((name, name_to_id[name]))
            continue
        # Fuzzy: some spawns drop/rename trailing _link.
        matches = [n for n in body_names if name in n or n in name]
        if len(matches) == 1:
            resolved.append((name, name_to_id[matches[0]]))
    return resolved


def _dump_robot_links(robot, env_id: int = 0) -> list[str]:
    lines = ["  Robot finger/hand links (world):"]
    body_pos = robot.data.body_pos_w[env_id]
    resolved = _resolve_body_ids(robot, _ROBOT_FINGER_LINKS)
    if not resolved:
        # Fallback: list any body with finger-ish tokens.
        for i, name in enumerate(robot.body_names):
            if any(tok in name for tok in ("thumb", "index", "middle", "ring", "pinky", "hand_base")):
                lines.append(f"    {name:40s}  {_fmt_xyz(body_pos[i].detach().cpu().numpy())}")
        return lines
    for name, bid in resolved:
        lines.append(f"    {name:40s}  {_fmt_xyz(body_pos[bid].detach().cpu().numpy())}")
    return lines


def _dump_finger_joints(robot, action: torch.Tensor | None, env_id: int = 0) -> list[str]:
    lines = ["  Proximal finger joints (q vs q_des / action hand slice):"]
    joint_pos = robot.data.joint_pos[env_id]
    joint_names = list(robot.joint_names)
    name_to_id = {n: i for i, n in enumerate(joint_names)}

    hand_action = None
    if action is not None and action.numel() >= 12:
        hand_action = action.detach().cpu().numpy().reshape(-1)[-12:]

    for i, name in enumerate(CONVERTED_SBERCHEL_HAND_JOINT_NAMES):
        jid = name_to_id.get(name)
        q = float(joint_pos[jid].item()) if jid is not None else float("nan")
        q_des = float(hand_action[i]) if hand_action is not None else float("nan")
        lines.append(f"    {name:36s}  q={q:7.3f}  q_des={q_des:7.3f}")
    return lines


def _assert_pink_hand_joint_order(env) -> None:
    """Fail loudly if Pink hand order drifts from packed converted names.

    Converted XR uses ``OrderedHandPinkInverseKinematicsAction`` so hand targets
    follow ``CONVERTED_SBERCHEL_HAND_JOINT_NAMES`` (``preserve_order=True``).
    """
    action_term = env.action_manager.get_term("upper_body_ik")
    resolved_names = list(getattr(action_term, "_hand_joint_names", []) or [])
    resolved_ids = list(getattr(action_term, "_hand_joint_ids", []) or [])
    expected = list(CONVERTED_SBERCHEL_HAND_JOINT_NAMES)

    print("[xr_compare] Pink hand joint order (resolved):")
    for i, (name, jid) in enumerate(zip(resolved_names, resolved_ids)):
        print(f"  [{i:2d}] id={jid:3d}  {name}")
    print("[xr_compare] Expected packed names:")
    for i, name in enumerate(expected):
        print(f"  [{i:2d}] {name}")

    if resolved_names != expected:
        raise RuntimeError(
            "Pink hand joint order does not match CONVERTED_SBERCHEL_HAND_JOINT_NAMES.\n"
            f"  resolved={resolved_names}\n"
            f"  expected={expected}\n"
            "Check OrderedHandPinkInverseKinematicsAction / preserve_order=True wiring."
        )
    print("[xr_compare] OK: Pink hand joint order matches packed names.")


def _dump_retargeter_packed(teleop_interface) -> list[str]:
    lines = ["  Retargeter packed fingers:"]
    retargeters = getattr(teleop_interface, "_retargeters", None) or []
    found = False
    for ret in retargeters:
        packed = getattr(ret, "_last_packed_fingers", None)
        if packed is None:
            continue
        found = True
        lines.append(f"    {np.array2string(np.asarray(packed), precision=3)}")
        names = getattr(ret, "_hand_joint_names", None)
        if names:
            lines.append(f"    names={list(names)}")
        signed = getattr(ret, "_last_opposition_signed", None)
        left_curls = getattr(ret, "_prev_left_curls", None) or {}
        right_curls = getattr(ret, "_prev_right_curls", None) or {}
        if signed is not None or left_curls or right_curls:
            lines.append(
                "  Thumb yaw / opposition:"
                f"  L opp={left_curls.get('opposition', float('nan')):.3f}"
                f" signed={signed.get('left', float('nan')) if signed else float('nan'):.3f}"
                f"  R opp={right_curls.get('opposition', float('nan')):.3f}"
                f" signed={signed.get('right', float('nan')) if signed else float('nan'):.3f}"
            )
    if not found:
        lines.append("    (no _last_packed_fingers on retargeters)")
    return lines


def dump_compare_block(
    teleop_interface,
    env,
    action: torch.Tensor | None,
    step: int,
    log_path: Path | None,
) -> None:
    robot = env.scene["robot"]
    lines = [
        "",
        "=" * 72,
        f"[xr_compare] step={step}  time={datetime.now().isoformat(timespec='seconds')}",
        "=" * 72,
    ]

    raw = None
    get_raw = getattr(teleop_interface, "_get_raw_data", None)
    if callable(get_raw):
        try:
            raw = get_raw()
        except Exception as exc:
            lines.append(f"  OpenXR raw fetch failed: {exc}")

    if raw is not None:
        left = raw.get(DeviceBase.TrackingTarget.HAND_LEFT, {})
        right = raw.get(DeviceBase.TrackingTarget.HAND_RIGHT, {})
        lines.extend(_dump_openxr_hand(left, "LEFT"))
        lines.extend(_dump_openxr_hand(right, "RIGHT"))
    else:
        lines.append("  OpenXR: device has no _get_raw_data()")

    lines.extend(_dump_robot_links(robot))
    lines.extend(_dump_finger_joints(robot, action))
    lines.extend(_dump_retargeter_packed(teleop_interface))
    lines.append("=" * 72)
    _emit("\n".join(lines), log_path)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "Teleoperation is only supported for ManagerBasedRLEnv environments. "
            f"Received: {type(env_cfg).__name__}"
        )
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    if args_cli.xr:
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    env_cfg.scene.scene_config_dir_path = "generated_envs/isaac_sample"
    env_cfg.target_item_name_regex = ".*Nestle Honey*"

    log_path = Path(args_cli.log_file) if args_cli.log_file else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"# sberchel xr compare start {datetime.now().isoformat()}\n")

    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    try:
        _assert_pink_hand_joint_order(env)
    except Exception as e:
        logger.error(f"Pink hand joint order check failed: {e}")
        env.close()
        simulation_app.close()
        return

    should_reset_recording_instance = False
    teleoperation_active = True

    def reset_recording_instance() -> None:
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = True
        if hasattr(teleop_interface, "_retargeters"):
            for retargeter in teleop_interface._retargeters:
                recalibrate = getattr(retargeter, "recalibrate_wrists", None)
                if callable(recalibrate):
                    recalibrate()
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    teleoperation_active = False if args_cli.xr else True

    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
        else:
            sensitivity = args_cli.sensitivity
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "gamepad":
                teleop_interface = Se3Gamepad(
                    Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
                )
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                env.close()
                simulation_app.close()
                return
            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")
    print(f"Compare dump every {args_cli.print_every_n} steps; log_file={log_path or '(stdout only)'}")
    print(f"OpenXR landmark set size={len(_OPENXR_LANDMARKS)} (of {len(HAND_JOINT_NAMES)} HAND_JOINT_NAMES)")

    env.reset()
    teleop_interface.reset()

    # One-shot body/joint inventory for naming mismatches.
    robot = env.scene["robot"]
    inv_lines = [
        "",
        "[xr_compare] robot body inventory (finger/hand related):",
    ]
    for name in robot.body_names:
        if any(tok in name for tok in ("thumb", "index", "middle", "ring", "pinky", "hand_base", "wrist")):
            inv_lines.append(f"  body: {name}")
    inv_lines.append("[xr_compare] robot joint inventory (finger related):")
    for name in robot.joint_names:
        if any(tok in name for tok in ("thumb", "index", "middle", "ring", "pinky")):
            inv_lines.append(f"  joint: {name}")
    _emit("\n".join(inv_lines), log_path)

    print("Teleoperation started. Press START in XR; dumps print while active.")
    step = 0

    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                action = teleop_interface.advance()

                if teleoperation_active:
                    actions = action.repeat(env.num_envs, 1)
                    env.step(actions)
                    if step % max(1, args_cli.print_every_n) == 0:
                        dump_compare_block(teleop_interface, env, action, step, log_path)
                    step += 1
                else:
                    env.sim.render()

                if should_reset_recording_instance:
                    env.reset()
                    teleop_interface.reset()
                    for retargeter in getattr(teleop_interface, "_retargeters", []):
                        recalibrate = getattr(retargeter, "recalibrate_wrists", None)
                        if callable(recalibrate):
                            recalibrate()
                    should_reset_recording_instance = False
                    step = 0
                    print("Environment reset complete")
        except Exception as e:
            logger.error(f"Error during simulation step: {e}")
            break

    env.close()
    print("Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()
