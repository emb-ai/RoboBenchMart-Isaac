"""Record Sberchel XR demos with WASD keyboard locomotion.

Upper-body teleop is XR hand tracking; legs use the zest locomotion policy
driven by keyboard velocity commands (like ``demo_sberchel_kb.py``). Press **C**
to toggle locomotion (WASD + soft frozen upper body) vs static teleop (station-keep
at the latch pose + live XR hands + headset neck look). Press **T** to teleport the
XR anchor onto the robot (same pose relative to the robot as at episode start).

OpenXR often steals window focus (carb WASD drops). This script polls X11 keymap
globally so hold-to-move works while the headset is focused. During locomotion,
arm PD is lowered to policy-scale gains — stiff XR teleop PD (~3000) on frozen
arms fights balance and makes walking shuffle or stall.

Episodes export on RESET/R. Press **X** to discard the current episode and reset
without writing to HDF5.

With ``--video``, also records the viewport (``rgb_array``) via Gymnasium
``RecordVideo``; each RESET/R saves an MP4 (discard/X skips the video write).

``--control_device gamepad`` swaps the keyboard for a PS5 (DualSense) pad read
from ``/dev/input/js*`` (focus-independent, like the X11 poll): left stick = WASD,
Create = C, L1 = T, R1 = V, D-pad up = R, D-pad down = X, D-pad left = V,
Options = START/STOP.
See ``dsynth_isaac/teleop/gamepad.py``.

Usage:
    python scripts/sberchel_kb_record_demos.py \\
        --task Isaac-Demo-Sberchel-XR-v0 \\
        --teleop_device handtracking \\
        --enable_pinocchio \\
        --dataset_file ./datasets/sberchel_kb_demos.hdf5 \\
        --control_device gamepad \\
        --video
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from isaaclab.app import AppLauncher

_DEFAULT_GC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "green_challenge"))
_GC = os.environ.get("GREEN_CHALLENGE_PATH", _DEFAULT_GC)
_gc_parent = os.path.dirname(os.path.abspath(_GC))
if _gc_parent not in sys.path:
    sys.path.insert(0, _gc_parent)

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

parser = argparse.ArgumentParser(
    description="Record Sberchel XR demos with WASD keyboard locomotion."
)
parser.add_argument("--task", type=str, default="Isaac-Demo-Sberchel-XR-v0", help="Gym task id.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/sberchel_kb_demos.hdf5",
    help="File path to export recorded demos.",
)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="handtracking",
    help="Teleop device key from env teleop_devices.",
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz (non-XR only).")
parser.add_argument(
    "--num_demos",
    type=int,
    default=0,
    help="Number of demonstrations to record. Set to 0 for unlimited.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio (required for Pink IK / hand tracking).",
)
parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help="Forward/back speed for W/S (m/s).",
)
parser.add_argument(
    "--turn",
    type=float,
    default=1.5,
    help="Turn rate for A/D (rad/s).",
)
parser.add_argument(
    "--strafe",
    type=float,
    default=0.6,
    help="Strafe speed for Q/E (m/s).",
)
parser.add_argument(
    "--station_lin_gain",
    type=float,
    default=1.5,
    help="P-gain for static-mode XY station-keep (m/s per meter).",
)
parser.add_argument(
    "--station_ang_gain",
    type=float,
    default=2.5,
    help="P-gain for static-mode yaw station-keep (rad/s per rad).",
)
parser.add_argument(
    "--station_max_lin",
    type=float,
    default=0.8,
    help="Max linear velocity for static-mode station-keep (m/s).",
)
parser.add_argument(
    "--station_max_ang",
    type=float,
    default=1.5,
    help="Max angular velocity for static-mode station-keep (rad/s).",
)
parser.add_argument(
    "--station_pos_deadzone",
    type=float,
    default=0.06,
    help=(
        "XY engage threshold for static-mode station-keep (m). Below it the legs get "
        "an exact zero command and the robot stands still, like green_challenge's own "
        "stack during manipulation. Once the base drifts past it, a correction runs "
        "until the error is back under half this value (hysteresis) -- a single "
        "threshold parks the base exactly on the boundary and chatters forever."
    ),
)
parser.add_argument(
    "--station_yaw_deadzone",
    type=float,
    default=0.10,
    help="Yaw engage threshold for static-mode station-keep (rad). See --station_pos_deadzone.",
)
parser.add_argument(
    "--station_min_lin",
    type=float,
    default=0.15,
    help=(
        "Minimum linear speed while a station-keep correction is engaged (m/s). "
        "Boundary-sized commands (~0.09 m/s) are below what the zest policy turns "
        "into actual motion, so without a floor the correction stalls and the error "
        "never closes."
    ),
)
parser.add_argument(
    "--station_min_ang",
    type=float,
    default=0.3,
    help="Minimum yaw rate while a station-keep correction is engaged (rad/s).",
)
parser.add_argument(
    "--ik_gain",
    type=float,
    default=None,
    help=(
        "Override Pink FrameTask gain (fraction of task error corrected per sim step). "
        "Default: whatever the env cfg ships (0.85). Raise toward 1.0 for snappier "
        "tracking of the hand markers; lower if the arms feel twitchy."
    ),
)
parser.add_argument(
    "--ik_lm_damping",
    type=float,
    default=None,
    help=(
        "Override Pink FrameTask lm_damping. Scales with squared task error, so high "
        "values brake hardest exactly when the target is far. Default: env cfg (1.0)."
    ),
)
parser.add_argument(
    "--ik_damping_cost",
    type=float,
    default=None,
    help="Override the Pink DampingTask cost (resists joint motion). Default: env cfg (0.1).",
)
parser.add_argument(
    "--gravity_compensation",
    action="store_true",
    default=False,
    help=(
        "Feed forward gravity torques on the Pink-controlled joints. With the soft "
        "green_challenge arm PD (42/28 Nm/rad) gravity droop is a constant offset the "
        "PD cannot close, so the hands sit below the markers; this removes it without "
        "touching the gains. NOTE: it is still a plant change (extra effort feedforward) "
        "-- recorded into dataset provenance as sberchel_gravity_compensation. During "
        "WASD locomotion the Pink hold bypasses the feedforward update, so the last "
        "static-mode torques persist on the frozen arms."
    ),
)
parser.add_argument(
    "--print_every_n",
    type=int,
    default=30,
    help="Print debug line every N steps (0 to disable periodic prints).",
)
parser.add_argument(
    "--auto_start",
    action="store_true",
    default=False,
    help="Start recording immediately (skip XR START). Useful to debug WASD locomotion.",
)
parser.add_argument(
    "--loco_render_interval",
    type=int,
    default=5,
    help=(
        "Physics steps per render while in locomotion mode (1 = every step). "
        "Higher skips Pink-free frames' XR/GUI presents and raises wall-clock Hz. "
        "Static teleop always renders every step."
    ),
)
parser.add_argument(
    "--idle_pause",
    action="store_true",
    default=False,
    help=(
        "Freeze the sim (no physics, no recording) whenever locomotion mode has a "
        "zero stick/WASD command. Stepping resumes the instant a command arrives or "
        "static mode is toggled. Idle standing between mode switches never reaches "
        "the dataset. The view keeps rendering while frozen."
    ),
)
parser.add_argument(
    "--idle_pause_settle",
    type=float,
    default=1.5,
    help=(
        "Seconds to keep stepping after the loco command returns to zero before "
        "--idle_pause freezes the sim. Lets the policy damp walking momentum into a "
        "stand so the frozen state (and any later static-mode hold pose) is calm."
    ),
)
parser.add_argument(
    "--control_device",
    type=str,
    default="keyboard",
    choices=["keyboard", "gamepad", "both"],
    help=(
        "Locomotion/episode input: WASD keyboard, PS5 (DualSense) gamepad, or both. "
        "Gamepad: left stick = WASD, Create = toggle locomotion, L1 = teleport, "
        "D-pad right = save+reset, D-pad left = discard+reset, Options = START/STOP."
    ),
)
parser.add_argument(
    "--gamepad_dev",
    type=str,
    default="",
    help="Joystick device for --control_device gamepad (default: auto-detect DualSense).",
)
parser.add_argument(
    "--gamepad_deadzone",
    type=float,
    default=0.15,
    help="Analog stick deadzone (fraction of full travel).",
)
parser.add_argument(
    "--gamepad_no_sticks",
    action="store_true",
    default=False,
    help="Ignore the analog sticks; drive locomotion from the D-pad (digital WASD) only. "
    "D-pad left/right then turn instead of discarding/saving episodes.",
)
parser.add_argument(
    "--gamepad_debug",
    action="store_true",
    default=False,
    help="Print every gamepad button/axis event (to verify the button mapping).",
)
parser.add_argument(
    "--legacy_stiff_teleop",
    action="store_true",
    default=False,
    help=(
        "Restore the old two-regime behaviour: stiff XR PD (arms 3000, fingers 400) "
        "in static teleop, green_challenge gains while walking. Easier to teleop but "
        "the demos are recorded on a plant green_challenge does not have (its arms run "
        "at 42/28), so they are NOT valid training data for their VLA. Use only to A/B "
        "the feel against the default single-regime run."
    ),
)
parser.add_argument(
    "--retargeter",
    type=str,
    default="vrd",
    choices=["vrd", "dex"],
    help=(
        "Finger mapping. 'vrd' = per-digit curl heuristic (default). 'dex' = DexPilot "
        "optimizer matching inter-fingertip vectors, as IsaacLab ships for GR1T2; better "
        "in principle for pinch/precision grasps. Affects only how your hand becomes "
        "finger commands -- the robot, its gains and the recorded action layout are "
        "identical either way. Validate first: scripts/sberchel_check_dex_retargeting.py"
    ),
)
parser.add_argument(
    "--squeeze_scale",
    type=float,
    default=1.0,
    help=(
        "Multiplier on the operator squeeze input (gamepad L2 / keyboard G) before it "
        "is clamped to 0..1. Squeeze commands finger closure past where your own hand "
        "stopped, which is what generates grip force on a position-controlled hand. "
        "Raise above 1.0 to reach full closure at partial trigger travel."
    ),
)
parser.add_argument(
    "--squeeze_binary",
    action="store_true",
    default=False,
    help=(
        "Make grasp squeeze 0/1 instead of analog: L2 travel is latched through a "
        "Schmitt trigger (engage above 0.6, release below 0.4 -- hysteresis so a "
        "half-pulled trigger cannot flutter the grip open mid-carry) into either 0 or "
        "--squeeze_on_value. Matches green_challenge's own convention (their keyboard/"
        "socket paths always push finger targets to full closure, never modulate) and "
        "makes every recorded grasp identical."
    ),
)
parser.add_argument(
    "--squeeze_on_value",
    type=float,
    default=1.0,
    help=(
        "Squeeze level commanded while the binary latch is engaged (0..1; applies to "
        "keyboard G too). Lower toward ~0.6 for gentler contact on small rigid objects. "
        "Only used with --squeeze_binary; --squeeze_scale still multiplies on top."
    ),
)
parser.add_argument(
    "--disable_left_hand",
    action="store_true",
    default=False,
    help=(
        "The left arm/hand never follows the operator: it holds the measured pose "
        "(wrist pelvis-relative + fingers) latched shortly after each episode reset, "
        "re-latched on every static-mode entry, in both locomotion and static modes; "
        "squeeze does not affect the left fingers."
    ),
)
parser.add_argument(
    "--grip_report_every_n",
    type=int,
    default=0,
    help=(
        "Print a per-hand grip readout (commanded vs achieved finger angle, error, "
        "estimated torque vs the 8.5 Nm effort limit) every N steps. 0 disables. "
        "Use this to confirm grasps actually generate force instead of assuming it."
    ),
)
parser.add_argument(
    "--keep_stiff_neck",
    action="store_true",
    default=False,
    help=(
        "Keep the stiff XR neck PD (200/20) instead of green_challenge's 28/1.4. "
        "Headset look feels snappier but diverges from their config -- the neck sits "
        "in their torso_joint_pos group and aims the head camera, so it changes the "
        "recorded images."
    ),
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record viewport MP4s via Gymnasium RecordVideo (saved on RESET/R).",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default="",
    help="Directory for viewport MP4s. Default: <dataset_dir>/videos.",
)
parser.add_argument(
    "--cameras",
    type=str,
    default="left_head",
    help=(
        "Comma-separated robot cameras to render and record into the HDF5 when "
        "--enable_cameras is set: left_head,right_head,left_wrist,right_wrist "
        "(or 'all'/'none'). Default keeps only the left head camera; extra "
        "cameras slow teleop rendering. Prefer recording without cameras and "
        "adding RGB later via sberchel_replay_demos.py --output_dataset_file."
    ),
)
parser.add_argument(
    "--camera_size",
    type=str,
    default="448x448",
    help=(
        "Robot camera resolution: 'WxH', a single square size, or 'native' (960x600). "
        "Default 448x448 matches the LeRobot sample layout (center-crop-resize "
        "rescaled intrinsics)."
    ),
)
parser.add_argument(
    "--camera_hz",
    type=float,
    default=30.0,
    help=(
        "Robot camera refresh rate during teleop (sensor update_period = 1/hz; "
        "0 = re-render every env step). The recorder still writes an image row "
        "per 150 Hz env step, so rows repeat the latest frame between refreshes; "
        "30 Hz matches the LeRobot target rate (green_challenge's default is 15)."
    ),
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
            "Sberchel hand-tracking teleop requires dex-retargeting. "
            "Install with: pip install 'dex-retargeting==0.4.6'"
        ) from exc

if _use_handtracking:
    app_launcher_args["xr"] = True
# Viewport rgb_array / RecordVideo needs camera rendering enabled.
if args_cli.video:
    app_launcher_args["enable_cameras"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.devices.device_base import DeviceBase
from dsynth_isaac.robots.sberchel_camera_cfg import (
    SBERCHEL_CAMERA_OBS_KEYS,
    apply_sberchel_camera_size,
    parse_camera_names,
    parse_camera_size,
    strip_camera_configs,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import DatasetExportMode
from isaaclab_tasks.utils import parse_env_cfg

import dsynth_isaac.tasks.manager_based  # noqa: F401

from dsynth_isaac.recorders import (
    SberchelXRRecorderManagerCfg,
    stash_locomotion_command,
    stash_teleop_state,
)

from dsynth_isaac.teleop.sberchel_head_follow import (
    FrozenUpperSnapshot,
    UpperBodyMode,
    build_idle_upper_from_robot,
    build_upper_from_frozen_snapshot,
    capture_frozen_upper,
    create_sberchel_xr_teleop_interface,
    headset_to_neck_targets,
    pack_sberchel_xr_action,
    robot_root_state,
    StationKeeper,
    sync_retargeter_idle_from_robot,
)
from dsynth_isaac.teleop.gamepad import RecordGamepad
from dsynth_isaac.teleop.xr_anchor import XrAnchorTeleporter
from dsynth_isaac.robots.sberchel_cfg import (
    apply_sberchel_loco_upper_pd_gains,
    apply_sberchel_xr_teleop_pd_gains,
    sberchel_upper_pd_gain_table,
)
from dsynth_isaac.tasks.manager_based.sberchel_upperbody_xr_env_cfg import (
    SBERCHEL_HAND_JOINT_NAMES,
    build_sberchel_teleop_devices,
)
from dsynth_isaac.tasks.manager_based.sberchel_base_env_cfg import (
    LEGS_ACTION_DIM,
    NECK_ACTION_DIM,
    NECK_PITCH_LIMIT,
    NECK_YAW_LIMIT,
    policy_leg_targets,
)
from green_challenge.policy.network import Network

logger = logging.getLogger(__name__)

LocoMode = Literal["locomotion", "static"]

# Left-hand slots in the 26D upper command (--disable_left_hand): wrist pose7 at
# [0:7], plus the left entries of the 12D finger block at [14:26].
_LEFT_WRIST_SLICE = slice(0, 7)
_LEFT_FINGER_DIMS = tuple(
    14 + i for i, name in enumerate(SBERCHEL_HAND_JOINT_NAMES) if name.startswith("left_")
)

# The reset pose writes left thumb yaw at -0.1, inside the palm collision hull
# (PhysX self-collisions ignore the MJCF contype/conaffinity filters), so PhysX
# kicks the thumb out and it settles onto a contact equilibrium near 0.24 rad.
# Commanding the unreachable -0.1 back at it saturates the 8.5 Nm finger drive
# against that contact and the joint chatters at ~37 Hz. Latch the
# --disable_left_hand hold from the *measured* pose only after this many env
# steps (~0.2 s at 150 Hz) so it records the reachable equilibrium, not the
# in-penetration reset value.
_LEFT_HOLD_SETTLE_STEPS = 30

_HELP_KEYBOARD = """
==================== KB RECORD CONTROLS ====================
  W forward    S backward    A turn left    D turn right
  Q strafe L   E strafe R    (release = stop legs in loco mode)
  C  toggle locomotion (WASD + Pink hold-off) / static teleop
     (static: XR hands + headset neck; Pink IK on, slower)
  G  hold to squeeze (close fingers past your hand -> real grip force)
  T  teleport XR view onto the robot (same offset as at episode start)
  V / Left   restore last checkpoint (auto-taken at each C into static
             mode: rolls sim + recording back — retry a grasp seamlessly)
  R / Up     save episode + reset (HDF5; viewport MP4 if --video)
  X / Down   discard episode + reset (no HDF5 / no MP4 write)
  XR START / STOP  begin or pause recording (or pass --auto_start)
  Keys: X11 global poll (works while headset has focus) + Isaac window
  Tip: if loco hz still low, raise --loco_render_interval (e.g. 10)
===========================================================
"""

_HELP_GAMEPAD = """
================= PS5 PAD RECORD CONTROLS =================
  Left stick        forward/back + turn  (= WASD)
  Right stick X     analog strafe
  D-pad             digital WASD fallback (with --gamepad_no_sticks)
  Create   toggle locomotion / static teleop   (= C)
  L2       analog grasp squeeze (0..1)         (= G)
  L1       teleport XR view onto the robot     (= T)
  R1       restore loco->static checkpoint     (= V / Left arrow)
  D-pad ^  save episode + reset                (= R / Up arrow)
  D-pad v  discard episode + reset             (= X / Down arrow)
  D-pad <  restore loco->static checkpoint     (= V / Left arrow)
  Options  start / stop recording        (= XR START / STOP)
  Pad is read from /dev/input/js* — works while the headset has focus
  Check the mapping any time: python -m dsynth_isaac.teleop.gamepad
===========================================================
"""

_MOVE_KEYS = ("W", "A", "S", "D", "Q", "E")
_EDGE_KEYS = ("C", "R", "X", "T", "V", "UP", "DOWN", "LEFT")


class PinkJointHold:
    """Bypass Pink IK solves by holding arm/hand joint targets.

    During WASD locomotion the upper body is frozen anyway; re-solving Pink every
    physics step (~10 Hz wall-clock with XR) is pure overhead. Holding joint-space
    targets keeps arms still relative to the torso at policy rate.
    """

    def __init__(self, env) -> None:
        self._env = env
        self._term = env.action_manager.get_term("upper_body_ik")
        self._orig_apply = self._term.apply_actions
        self._targets: torch.Tensor | None = None
        self._active = False
        self._targets_valid = False

    def invalidate(self) -> None:
        """Mark commanded targets as stale — call right after ``env.reset()``.

        ``joint_pos_target`` is allocated as zeros and is *not* restored to the
        default pose on reset (``articulation.py:1641``), so after a reset it
        still holds the previous episode's commands while the robot has been
        teleported back to its init pose.
        """
        self._targets_valid = False

    def mark_stepped(self) -> None:
        """Commanded targets are meaningful once a step has written them."""
        self._targets_valid = True

    def _capture(self, robot) -> torch.Tensor:
        """Snapshot the *commanded* joint targets, not the measured pose.

        This distinction is the whole grasp. A hand squeezing an object sits at a
        measured angle the object allows, while the command sits deeper; the
        difference is exactly what the PD converts into grip force. Capturing
        ``joint_pos`` would set target := measured, zeroing that error and
        dropping whatever is being held the instant the hold engages.

        Before the first step of an episode there is no meaningful command yet,
        so fall back to the measured pose — otherwise the arms would be driven to
        the zero-initialised target and snap out of their init pose.
        """
        ids = self._term._controlled_joint_ids
        source = robot.data.joint_pos_target if self._targets_valid else robot.data.joint_pos
        return source[:, ids].detach().clone()

    def enable(self, robot) -> None:
        self._targets = self._capture(robot)
        if not self._active:
            self._term.apply_actions = self._apply_hold  # type: ignore[method-assign]
            self._active = True

    def refresh(self, robot) -> None:
        """Re-capture hold targets (call after reset / mode entry)."""
        self._targets = self._capture(robot)

    def disable(self) -> None:
        if self._active:
            self._term.apply_actions = self._orig_apply  # type: ignore[method-assign]
            self._active = False

    def _apply_hold(self) -> None:
        if self._targets is None:
            self._orig_apply()
            return
        self._term._asset.set_joint_position_target(self._targets, self._term._controlled_joint_ids)
        # Keep ``processed_actions`` in step with what we actually applied.
        # Bypassing apply_actions also bypasses the assignment inside it, so
        # IsaacLab's PostStepProcessedActionsRecorder would otherwise log a stale
        # vector from the last real IK solve for the whole locomotion segment --
        # i.e. the dataset would disagree with the simulator.
        processed = self._term._processed_actions
        if processed.shape != self._targets.shape:
            raise RuntimeError(
                "PinkJointHold: processed_actions shape "
                f"{tuple(processed.shape)} != held targets {tuple(self._targets.shape)}; "
                "the upper_body_ik action layout changed and the hold would record garbage."
            )
        processed[:] = self._targets


def set_render_interval(env, interval: int) -> None:
    """Set physics-steps-per-render (read every ManagerBasedEnv.step)."""
    env.cfg.sim.render_interval = max(1, int(interval))


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env) -> None:
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class _X11KeyPoller:
    """Poll physical key state via XQueryKeymap (no window-focus requirement)."""

    # X11 keysyms (lowercase letter keys).
    _KEYSYMS = {
        "W": 0x0077,
        "A": 0x0061,
        "S": 0x0073,
        "D": 0x0064,
        "Q": 0x0071,
        "E": 0x0065,
        "C": 0x0063,
        "R": 0x0072,
        "X": 0x0078,
        "T": 0x0074,
        "G": 0x0067,
        "V": 0x0076,
        # Arrow keys (X11 cursor keysyms).
        "UP": 0xFF52,
        "DOWN": 0xFF54,
        "LEFT": 0xFF51,
    }

    def __init__(self) -> None:
        import ctypes
        import ctypes.util

        self._ctypes = ctypes
        lib_name = ctypes.util.find_library("X11")
        if not lib_name:
            raise RuntimeError("libX11 not found")
        self._x11 = ctypes.cdll.LoadLibrary(lib_name)
        self._x11.XOpenDisplay.restype = ctypes.c_void_p
        self._x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self._x11.XKeysymToKeycode.restype = ctypes.c_uint
        self._x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self._x11.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self._display = self._x11.XOpenDisplay(None)
        if not self._display:
            raise RuntimeError("XOpenDisplay failed")
        self._keycodes = {
            name: int(self._x11.XKeysymToKeycode(self._display, keysym))
            for name, keysym in self._KEYSYMS.items()
        }

    def pressed(self) -> set[str]:
        ctypes = self._ctypes
        keymap = (ctypes.c_char * 32)()
        self._x11.XQueryKeymap(self._display, keymap)
        down: set[str] = set()
        for name, keycode in self._keycodes.items():
            if keycode <= 0:
                continue
            byte = keymap[keycode // 8]
            bit = ord(byte) if isinstance(byte, (bytes, bytearray)) else byte
            if bit & (1 << (keycode % 8)):
                down.add(name)
        return down

    def close(self) -> None:
        if getattr(self, "_display", None):
            try:
                self._x11.XCloseDisplay(self._display)
            except Exception:
                pass
            self._display = None


class RecordKeyboard:
    """WASDQE hold-to-move plus edge-triggered C / R / X / T / V and arrows.

    Arrow aliases: Up = save+reset (R), Down = discard+reset (X),
    Left = restore checkpoint (V).

    Uses X11 keymap polling (global, works under OpenXR focus steal) unioned with
    Isaac carb press/release events as a fallback.
    """

    def __init__(
        self,
        speed: float,
        turn: float,
        strafe: float,
        *,
        on_toggle_mode: Callable[[], None] | None = None,
        on_save_reset: Callable[[], None] | None = None,
        on_discard_reset: Callable[[], None] | None = None,
        on_teleport_view: Callable[[], None] | None = None,
        on_restore_checkpoint: Callable[[], None] | None = None,
    ) -> None:
        import carb
        import omni

        self._carb = carb
        self._speed = speed
        self._turn = turn
        self._strafe = strafe
        self._on_toggle_mode = on_toggle_mode
        self._on_save_reset = on_save_reset
        self._on_discard_reset = on_discard_reset
        self._on_teleport_view = on_teleport_view
        self._on_restore_checkpoint = on_restore_checkpoint
        self._carb_pressed: set[str] = set()
        self._edge_prev: set[str] = set()
        self._ok = False
        self._x11: _X11KeyPoller | None = None
        try:
            self._x11 = _X11KeyPoller()
            print("[KB] X11 global key poll active (WASD works without Isaac focus)", flush=True)
        except Exception as exc:
            print(
                f"[KB] X11 key poll unavailable ({exc!r}); "
                f"click the Isaac window so carb keyboard works.",
                flush=True,
            )
        try:
            self._appwindow = omni.appwindow.get_default_app_window()
            self._input = carb.input.acquire_input_interface()
            self._keyboard = self._appwindow.get_keyboard()
            self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)
            self._ok = True
        except Exception as exc:  # headless / no window -> no carb keyboard
            print(
                f"[KB] carb keyboard unavailable ({exc!r}).",
                flush=True,
            )

    def _on_event(self, event, *args, **kwargs):
        carb = self._carb
        try:
            et = event.type
            if et not in (
                carb.input.KeyboardEventType.KEY_PRESS,
                carb.input.KeyboardEventType.KEY_RELEASE,
            ):
                return True
            raw = event.input
            name = raw if isinstance(raw, str) else getattr(raw, "name", "")
            name = name.rsplit(".", 1)[-1].upper()
            if et == carb.input.KeyboardEventType.KEY_PRESS:
                self._carb_pressed.add(name)
            else:
                self._carb_pressed.discard(name)
        except Exception:
            pass
        return True

    def _pressed_now(self) -> set[str]:
        pressed = set(self._carb_pressed)
        if self._x11 is not None:
            try:
                pressed |= self._x11.pressed()
            except Exception:
                pass
        return pressed

    def poll(self) -> None:
        """Process edge-triggered C / R / X / T / V / arrows; call every loop iteration."""
        pressed = self._pressed_now()
        edge_now = pressed & set(_EDGE_KEYS)
        newly = edge_now - self._edge_prev
        if "C" in newly and self._on_toggle_mode is not None:
            self._on_toggle_mode()
        if newly & {"R", "UP"} and self._on_save_reset is not None:
            self._on_save_reset()
        if newly & {"X", "DOWN"} and self._on_discard_reset is not None:
            self._on_discard_reset()
        if "T" in newly and self._on_teleport_view is not None:
            self._on_teleport_view()
        if newly & {"V", "LEFT"} and self._on_restore_checkpoint is not None:
            self._on_restore_checkpoint()
        self._edge_prev = edge_now

    def squeeze(self) -> float:
        """Hold G for full grasp squeeze (keyboard has no analog axis)."""
        return 1.0 if "G" in self._pressed_now() else 0.0

    def command(self) -> tuple[float, float, float]:
        """Return (lin_vel_x, lin_vel_y, ang_vel_z) from held WASDQE keys."""
        p = self._pressed_now()
        vx = (self._speed if "W" in p else 0.0) - (self._speed if "S" in p else 0.0)
        wz = (self._turn if "A" in p else 0.0) - (self._turn if "D" in p else 0.0)
        vy = (self._strafe if "Q" in p else 0.0) - (self._strafe if "E" in p else 0.0)
        return vx, vy, wz

    def pressed_keys(self) -> str:
        """Compact debug string of currently held move keys."""
        p = self._pressed_now()
        return "".join(k for k in _MOVE_KEYS if k in p) or "-"

    def close(self) -> None:
        if self._ok:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            except Exception:
                pass
        if self._x11 is not None:
            self._x11.close()
            self._x11 = None


def setup_output_directories() -> tuple[str, str]:
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir, output_file_name


def exported_demo_count(env) -> int:
    """Total episodes written to HDF5."""
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    rm = base.recorder_manager
    return rm.exported_successful_episode_count + rm.exported_failed_episode_count


def discard_viewport_video_frames(env: gym.Env) -> None:
    """Drop in-progress RecordVideo frames so RESET discard does not write an MP4."""
    if isinstance(env, gym.wrappers.RecordVideo) and env.recording:
        env.recorded_frames.clear()


def viewport_video_frame_count(env: gym.Env) -> int:
    """In-progress RecordVideo frame count (0 when video is off)."""
    if isinstance(env, gym.wrappers.RecordVideo) and env.recording:
        return len(env.recorded_frames)
    return 0


def truncate_viewport_video_frames(env: gym.Env, keep: int) -> None:
    """Roll the in-progress RecordVideo buffer back to ``keep`` frames."""
    if isinstance(env, gym.wrappers.RecordVideo) and env.recording:
        del env.recorded_frames[keep:]


def episode_leaf_lengths(data: dict, prefix: str = "") -> dict[str, int]:
    """Per-leaf frame counts of an in-memory ``EpisodeData.data`` dict.

    Leaves are lists of per-frame tensors (``EpisodeData.add`` appends), keyed
    here by their "a/b/c" path. Counting every leaf separately keeps rollback
    correct even for keys recorded at a different cadence (e.g. the once-per-
    episode ``initial_state``).
    """
    lengths: dict[str, int] = {}
    for key, value in data.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            lengths.update(episode_leaf_lengths(value, path))
        elif isinstance(value, list):
            lengths[path] = len(value)
    return lengths


def truncate_episode_data(data: dict, lengths: dict[str, int], prefix: str = "") -> None:
    """Drop frames appended after ``lengths`` was captured (checkpoint rollback).

    A leaf that did not exist at capture time is truncated to zero frames.
    """
    for key, value in data.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            truncate_episode_data(value, lengths, path)
        elif isinstance(value, list):
            del value[lengths.get(path, 0):]


def create_environment_config(output_dir: str, output_file_name: str) -> ManagerBasedRLEnvCfg:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(f"Expected ManagerBasedRLEnvCfg, got {type(env_cfg).__name__}")

    env_cfg.env_name = args_cli.task.split(":")[-1]
    env_cfg.seed = args_cli.seed
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    if args_cli.enable_cameras:
        # Keep only the requested robot cameras: extras slow teleop rendering
        # and bloat the HDF5 via the RGB recorder term.
        selected_cameras = parse_camera_names(args_cli.cameras)
        env_cfg = strip_camera_configs(env_cfg, keep=selected_cameras)
        apply_sberchel_camera_size(env_cfg, parse_camera_size(args_cli.camera_size), selected_cameras)
        update_period = (1.0 / args_cli.camera_hz) if args_cli.camera_hz > 0 else 0.0
        for camera in selected_cameras:
            scene_attr, _ = SBERCHEL_CAMERA_OBS_KEYS[camera]
            cam_cfg = getattr(env_cfg.scene, scene_attr, None)
            if cam_cfg is not None:
                cam_cfg.update_period = update_period
        if selected_cameras:
            print(
                f">>> robot cameras: {', '.join(selected_cameras)} "
                f"({args_cli.camera_size} @ {args_cli.camera_hz:g} Hz refresh)",
                flush=True,
            )
    else:
        env_cfg = strip_camera_configs(env_cfg)
    if args_cli.xr:
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    if args_cli.retargeter != "vrd":
        # Swap the finger mapping only; the Pink export done in the env cfg's
        # __post_init__ still stands, so no need to redo it.
        build_sberchel_teleop_devices(env_cfg, args_cli.retargeter)
        print(f">>> retargeter: {args_cli.retargeter}", flush=True)

    # Optional Pink IK task overrides for in-headset tuning. These shape only how
    # commanded joint targets chase the operator's hand -- the demonstrator, not
    # the plant -- so they never invalidate demos for green_challenge.
    from pink.tasks import DampingTask as _PinkDampingTask
    from pink.tasks import FrameTask as _PinkFrameTask

    ik_tasks = env_cfg.actions.upper_body_ik.controller.variable_input_tasks
    for task in ik_tasks:
        if isinstance(task, _PinkFrameTask):
            if args_cli.ik_gain is not None:
                task.gain = float(args_cli.ik_gain)
            if args_cli.ik_lm_damping is not None:
                task.lm_damping = float(args_cli.ik_lm_damping)
        elif isinstance(task, _PinkDampingTask) and args_cli.ik_damping_cost is not None:
            task.cost = float(args_cli.ik_damping_cost)
    _frame_tasks = [t for t in ik_tasks if isinstance(t, _PinkFrameTask)]
    if _frame_tasks:
        print(
            f">>> Pink IK: gain={_frame_tasks[0].gain:g} "
            f"lm_damping={_frame_tasks[0].lm_damping:g}",
            flush=True,
        )

    if args_cli.gravity_compensation:
        env_cfg.actions.upper_body_ik.enable_gravity_compensation = True
        print(">>> gravity compensation: ON (recorded in dataset provenance)", flush=True)

    env_cfg.recorders = SberchelXRRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    env_cfg.recorders.export_in_record_pre_reset = True
    # Avoid exporting a stub episode (initial_state only) when closing after RESET.
    env_cfg.recorders.export_in_close = False
    return env_cfg


def capture_hold_pose_from_robot(robot) -> tuple[float, float, float]:
    """Latch current robot world XY and yaw for static-mode station-keeping."""
    (x, y, _), yaw = robot_root_state(robot)
    return x, y, yaw


def reset_teleop_state(
    *,
    env,
    net,
    robot,
    teleop_interface,
    loco_mode: LocoMode = "locomotion",
) -> tuple[FrozenUpperSnapshot | None, torch.Tensor | None, torch.Tensor]:
    """Reset policy and teleop state after env.reset()."""
    net.reset()
    # Single regime by default: the plant must not depend on the mode, or the
    # dataset becomes non-Markovian (same observation, different dynamics).
    if args_cli.legacy_stiff_teleop and loco_mode == "static":
        apply_sberchel_xr_teleop_pd_gains(robot)
    else:
        apply_sberchel_loco_upper_pd_gains(robot, include_neck=not args_cli.keep_stiff_neck)
    teleop_interface.reset()
    sync_retargeter_idle_from_robot(teleop_interface, robot)
    idle_upper = build_idle_upper_from_robot(robot, env.device)
    return None, None, idle_upper


def main() -> None:
    output_dir, output_file_name = setup_output_directories()
    env_cfg = create_environment_config(output_dir, output_file_name)
    env_cfg.viewer.eye = (4, 1.0, 2.5)
    env_cfg.viewer.lookat = (9.5, 6.0, 0.0)

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if args_cli.video:
        video_folder = Path(args_cli.video_dir) if args_cli.video_dir else Path(output_dir) / "videos"
        video_folder.mkdir(parents=True, exist_ok=True)
        name_prefix = time.strftime("%Y%m%d_%H%M%S")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            episode_trigger=lambda episode_id: True,
            video_length=0,  # entire episode; MP4 written on reset
            disable_logger=True,
            name_prefix=name_prefix,
        )
        print(f"Viewport video recording enabled → {video_folder}", flush=True)

    base_env = env.unwrapped
    base_env.sim.reset()
    env.reset(seed=args_cli.seed)

    net = Network(env, type="zest", device="cpu", closed_kinematics=False)
    net.reset()
    robot = base_env.scene["robot"]

    def apply_native_gains() -> None:
        """green_challenge's effective PD -- the plant their VLA is evaluated on."""
        apply_sberchel_loco_upper_pd_gains(robot, include_neck=not args_cli.keep_stiff_neck)

    apply_native_gains()
    station_keeper = StationKeeper(
        lin_gain=args_cli.station_lin_gain,
        ang_gain=args_cli.station_ang_gain,
        max_lin=args_cli.station_max_lin,
        max_ang=args_cli.station_max_ang,
        pos_engage=args_cli.station_pos_deadzone,
        yaw_engage=args_cli.station_yaw_deadzone,
        min_lin=args_cli.station_min_lin,
        min_ang=args_cli.station_min_ang,
    )
    pink_hold = PinkJointHold(base_env)
    pink_hold.enable(robot)
    set_render_interval(base_env, args_cli.loco_render_interval)
    upper_body_dim = base_env.action_manager.total_action_dim - LEGS_ACTION_DIM - NECK_ACTION_DIM
    print(
        f">>> action dim = {base_env.action_manager.total_action_dim} "
        f"(legs {LEGS_ACTION_DIM} + neck {NECK_ACTION_DIM} + upper_body {upper_body_dim})",
        flush=True,
    )
    print(
        f">>> loco fast-path: Pink IK hold-on + render_interval={args_cli.loco_render_interval}",
        flush=True,
    )
    if args_cli.legacy_stiff_teleop:
        print(
            ">>> GAINS: LEGACY two-regime (stiff XR PD in static teleop). "
            "Demos are NOT valid green_challenge training data -- A/B use only.",
            flush=True,
        )
    else:
        print(">>> GAINS: single regime, green_challenge-native (arms 42/28)", flush=True)
    _gain_table = sberchel_upper_pd_gain_table(robot)
    print(
        ">>> upper-body PD: "
        + ", ".join(f"{n}={k:g}/{d:g}" for n, (k, d) in sorted(_gain_table.items())),
        flush=True,
    )

    # --- dataset provenance ---------------------------------------------------
    # Record the joint ordering and the plant the demos were produced on. Every
    # joint-space consumer (green_challenge's VLA groups in particular) needs the
    # name order to slice these vectors correctly, and `find_joints` does NOT
    # preserve the order given in the cfg -- so it must be read at runtime rather
    # than assumed. Without this a converter is guessing, and a wrong guess is
    # silent: our 12 finger values and theirs are both length 12 but ordered
    # differently, so thumb commands would quietly land on the pinky.
    def _write_dataset_provenance() -> None:
        handler = getattr(base_env.recorder_manager, "_dataset_file_handler", None)
        if handler is None or not hasattr(handler, "add_env_args"):
            print("[provenance] no dataset file handler; joint order NOT recorded", flush=True)
            return
        upper_term = base_env.action_manager.get_term("upper_body_ik")
        try:
            handler.add_env_args(
                {
                    "sberchel_joint_names": list(robot.data.joint_names),
                    "sberchel_upper_controlled_joint_names": list(
                        upper_term._controlled_joint_names
                    ),
                    "sberchel_hand_joint_names": list(upper_term._hand_joint_names),
                    "sberchel_upper_pd_gains": {n: list(v) for n, v in _gain_table.items()},
                    "sberchel_legacy_stiff_teleop": bool(args_cli.legacy_stiff_teleop),
                    "sberchel_squeeze_scale": float(args_cli.squeeze_scale),
                    "sberchel_squeeze_binary": bool(args_cli.squeeze_binary),
                    "sberchel_squeeze_on_value": float(args_cli.squeeze_on_value),
                    # Which finger mapping produced these demos. Not a plant
                    # difference (the robot is identical), but it is a
                    # demonstrator-style difference, so keep episodes separable.
                    "sberchel_retargeter": str(args_cli.retargeter),
                    "sberchel_gravity_compensation": bool(
                        getattr(upper_term.cfg, "enable_gravity_compensation", False)
                    ),
                    # Demonstrator-side Pink IK shaping (not a plant difference,
                    # but affects how closely demos track the operator's hand).
                    "sberchel_pink_ik_tasks": [
                        {
                            "type": type(t).__name__,
                            "gain": float(t.gain),
                            "lm_damping": float(t.lm_damping),
                        }
                        if hasattr(t, "frame")  # FrameTask and subclasses
                        else {"type": type(t).__name__, "cost": float(getattr(t, "cost", 0.0) or 0.0)}
                        for t in upper_term.cfg.controller.variable_input_tasks
                    ],
                    "sberchel_action_layout": "legs(12) + neck(2) + upper_body_ik(26)",
                    "sberchel_teleop_state_layout": "is_locomotion, squeeze, legacy_gains",
                }
            )
            print(">>> dataset provenance written (joint order + gains)", flush=True)
        except Exception as exc:  # never lose a recording session over metadata
            print(f"[provenance] could not write env args: {exc!r}", flush=True)

    _write_dataset_provenance()

    should_reset = False
    should_discard_reset = False
    teleoperation_active = (not args_cli.xr) or args_cli.auto_start
    running_recording_instance = (not args_cli.xr) or args_cli.auto_start
    step_count = 0
    first_active_logged = False
    loco_mode: LocoMode = "locomotion"
    hold_xy_yaw: tuple[float, float, float] | None = None
    frozen_snapshot: FrozenUpperSnapshot | None = None
    left_hold_snapshot: FrozenUpperSnapshot | None = None
    last_teleop_upper: torch.Tensor | None = None
    last_neck_targets: tuple[float, float] = (0.0, 0.0)
    last_hz_t = time.time()
    hz_window_steps = 0
    effective_hz = 0.0
    # --idle_pause: wall-clock of the last nonzero loco command. None means "no
    # command since entering locomotion mode" -> frozen immediately, no settle.
    last_loco_cmd_t: float | None = None
    idle_paused_prev = False
    # Auto-checkpoint taken at every loco->static switch: full scene physics
    # state + policy carry-over + episode/video frame counts for rollback.
    sim_checkpoint: dict | None = None

    def enter_locomotion_mode(*, capture_frozen: bool) -> None:
        nonlocal loco_mode, hold_xy_yaw, frozen_snapshot, last_neck_targets
        nonlocal last_loco_cmd_t
        loco_mode = "locomotion"
        hold_xy_yaw = None
        last_neck_targets = (0.0, 0.0)
        last_loco_cmd_t = None
        if capture_frozen:
            frozen_snapshot = capture_frozen_upper(robot, last_teleop_upper)
        # Only the legacy path switches plants; by default the gains never move,
        # so C changes what is being commanded, not what the robot physically is.
        if args_cli.legacy_stiff_teleop:
            apply_native_gains()
        pink_hold.enable(robot)
        set_render_interval(base_env, args_cli.loco_render_interval)

    def enter_static_mode() -> None:
        nonlocal loco_mode, hold_xy_yaw, frozen_snapshot, last_neck_targets
        nonlocal left_hold_snapshot
        loco_mode = "static"
        hold_xy_yaw = capture_hold_pose_from_robot(robot)
        station_keeper.reset()
        sync_retargeter_idle_from_robot(teleop_interface, robot)
        if args_cli.disable_left_hand:
            # The left hand is never used, so the measured pose is the hold target.
            left_hold_snapshot = capture_frozen_upper(robot)
        frozen_snapshot = None
        last_neck_targets = (0.0, 0.0)
        pink_hold.disable()
        if args_cli.legacy_stiff_teleop:
            apply_sberchel_xr_teleop_pd_gains(robot)
        set_render_interval(base_env, 1)

    def apply_left_hold(upper_body: torch.Tensor) -> torch.Tensor:
        """Overwrite left wrist/finger slots with the --disable_left_hand hold."""
        if not args_cli.disable_left_hand or left_hold_snapshot is None:
            return upper_body
        left_hold = build_upper_from_frozen_snapshot(
            robot, left_hold_snapshot, upper_body.device
        ).to(upper_body.dtype)
        upper_body[..., _LEFT_WRIST_SLICE] = left_hold[_LEFT_WRIST_SLICE]
        for dim in _LEFT_FINGER_DIMS:
            upper_body[..., dim] = left_hold[dim]
        return upper_body

    def reset_env() -> None:
        nonlocal should_reset
        should_reset = True
        print("Reset requested — episode will be saved and environment reset on next step")

    def discard_reset_env() -> None:
        nonlocal should_discard_reset
        should_discard_reset = True
        print("Discard reset requested — episode will NOT be saved; environment reset on next step")

    def capture_checkpoint() -> None:
        """Snapshot everything needed to roll the episode back to this instant.

        The scene state covers the robot (root + joints + velocities) and every
        rigid object, so a dropped item is teleported back too. The zest policy's
        only cross-step state is ``controller.actions`` (last action, fed back
        into its observation); the commander is rewritten every tick. Episode
        and video buffers are rolled back by frame count, which is exact because
        restore reproduces the checkpoint state bit-for-bit — the stitched
        recording stays continuous.
        """
        nonlocal sim_checkpoint
        episode = base_env.recorder_manager.get_episode(0)
        sim_checkpoint = {
            "scene_state": base_env.scene.get_state(is_relative=False),
            # Store products are plain USD rigid bodies, not registered scene
            # assets, so scene.get_state() misses them — snapshot via PhysX view.
            "store_state": (
                base_env.scene.capture_store_rigid_body_state(base_env.sim)
                if hasattr(base_env.scene, "capture_store_rigid_body_state")
                else None
            ),
            "policy_actions": net.controller.actions.clone(),
            "episode_lengths": episode_leaf_lengths(episode.data),
            "video_frames": viewport_video_frame_count(env),
        }
        print("[checkpoint] captured (V / R1 rolls sim + recording back here)", flush=True)

    def restore_checkpoint() -> None:
        nonlocal last_teleop_upper
        if not teleoperation_active:
            print("[checkpoint] ignored — recording is stopped", flush=True)
            return
        if sim_checkpoint is None:
            print("[checkpoint] none captured yet this episode (press C to enter static mode first)", flush=True)
            return
        base_env.scene.reset_to(sim_checkpoint["scene_state"], None, is_relative=False)
        if sim_checkpoint["store_state"] is not None:
            restored = base_env.scene.restore_store_rigid_body_state(
                base_env.sim, sim_checkpoint["store_state"]
            )
            print(f"[checkpoint] store products restored: {restored}", flush=True)
        net.controller.actions.copy_(sim_checkpoint["policy_actions"])
        truncate_episode_data(
            base_env.recorder_manager.get_episode(0).data,
            sim_checkpoint["episode_lengths"],
        )
        truncate_viewport_video_frames(env, sim_checkpoint["video_frames"])
        # Commanded targets were overwritten by scene.reset_to; treat them as
        # stale until the next stepped frame, same as after env.reset().
        pink_hold.invalidate()
        last_teleop_upper = None
        enter_static_mode()
        if anchor_teleporter is not None:
            anchor_teleporter.teleport_to_robot(robot)
        print(
            "[checkpoint] restored — static teleop re-latched at the checkpoint pose "
            "(checkpoint kept; V / R1 works again)",
            flush=True,
        )

    def toggle_loco_mode() -> None:
        if not teleoperation_active:
            return
        if loco_mode == "locomotion":
            enter_static_mode()
            capture_checkpoint()
            print(
                f"Mode: STATIC teleop (station-keep at "
                f"xy=({hold_xy_yaw[0]:.3f}, {hold_xy_yaw[1]:.3f}) yaw={hold_xy_yaw[2]:.3f}; "
                f"headset drives neck; Pink IK on)",
                flush=True,
            )
        else:
            enter_locomotion_mode(capture_frozen=True)
            print("Mode: LOCOMOTION (WASD + Pink IK hold-off + soft arm PD)", flush=True)

    def start_recording_instance() -> None:
        nonlocal running_recording_instance, teleoperation_active, first_active_logged
        nonlocal frozen_snapshot, last_teleop_upper
        running_recording_instance = True
        teleoperation_active = True
        first_active_logged = False
        frozen_snapshot = None
        last_teleop_upper = None
        enter_locomotion_mode(capture_frozen=False)
        sync_retargeter_idle_from_robot(teleop_interface, robot)
        print("Recording started (WASD locomotion fast-path; press C for static teleop)")

    def stop_recording_instance() -> None:
        nonlocal running_recording_instance, teleoperation_active
        nonlocal hold_xy_yaw, frozen_snapshot, last_teleop_upper, last_neck_targets
        running_recording_instance = False
        teleoperation_active = False
        hold_xy_yaw = None
        frozen_snapshot = None
        last_teleop_upper = None
        last_neck_targets = (0.0, 0.0)
        enter_locomotion_mode(capture_frozen=False)
        print("Recording paused")

    def toggle_recording_instance() -> None:
        """Gamepad stand-in for the XR START / STOP buttons."""
        if running_recording_instance:
            stop_recording_instance()
        else:
            start_recording_instance()

    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_env,
        "START": start_recording_instance,
        "STOP": stop_recording_instance,
        "RESET": reset_env,
    }

    # Head tracking required for static-mode neck look.
    teleop_interface = create_sberchel_xr_teleop_interface(
        env_cfg, args_cli.teleop_device, teleoperation_callbacks, require_head_tracking=True
    )
    teleop_interface.reset()
    sync_retargeter_idle_from_robot(teleop_interface, robot)
    idle_upper = build_idle_upper_from_robot(robot, base_env.device)

    # Anchor prim only exists once the OpenXR device has been created above.
    anchor_teleporter = (
        XrAnchorTeleporter(env_cfg.xr, robot) if args_cli.xr else None
    )

    def teleport_view_to_robot() -> None:
        """Snap the XR view back onto the robot at the episode-start offset."""
        if anchor_teleporter is None:
            print("[XR] Teleport ignored: no XR anchor (run with hand tracking).", flush=True)
            return
        pose = anchor_teleporter.teleport_to_robot(robot)
        if pose is None:
            print("[XR] Teleport failed: could not write the XR anchor prim.", flush=True)
            return
        (ax, ay, az), ayaw = pose
        print(
            f"[XR] View teleported to robot: anchor xy=({ax:.3f}, {ay:.3f}) "
            f"z={az:.3f} yaw={ayaw:.3f}",
            flush=True,
        )

    # Keyboard and gamepad expose the same poll/command/pressed_keys surface, so the
    # loop below is agnostic to which one (or both) is driving.
    controllers: list[RecordKeyboard | RecordGamepad] = []
    use_keyboard = args_cli.control_device in ("keyboard", "both")
    use_gamepad = args_cli.control_device in ("gamepad", "both")
    if use_keyboard:
        controllers.append(
            RecordKeyboard(
                args_cli.speed,
                args_cli.turn,
                args_cli.strafe,
                on_toggle_mode=toggle_loco_mode,
                on_save_reset=reset_env,
                on_discard_reset=discard_reset_env,
                on_teleport_view=teleport_view_to_robot,
                on_restore_checkpoint=restore_checkpoint,
            )
        )
    if use_gamepad:
        controllers.append(
            RecordGamepad(
                args_cli.speed,
                args_cli.turn,
                args_cli.strafe,
                device=args_cli.gamepad_dev,
                deadzone=args_cli.gamepad_deadzone,
                use_sticks=not args_cli.gamepad_no_sticks,
                debug=args_cli.gamepad_debug,
                on_toggle_mode=toggle_loco_mode,
                on_save_reset=reset_env,
                on_discard_reset=discard_reset_env,
                on_teleport_view=teleport_view_to_robot,
                on_toggle_recording=toggle_recording_instance,
                on_restore_checkpoint=restore_checkpoint,
            )
        )

    def poll_controllers() -> None:
        for controller in controllers:
            controller.poll()

    def controller_command() -> tuple[float, float, float]:
        """Largest-magnitude command per axis across the active input devices."""
        vx = vy = wz = 0.0
        for controller in controllers:
            cx, cy, cz = controller.command()
            vx = cx if abs(cx) > abs(vx) else vx
            vy = cy if abs(cy) > abs(vy) else vy
            wz = cz if abs(cz) > abs(wz) else wz
        return vx, vy, wz

    def controller_pressed() -> str:
        active = [c.pressed_keys() for c in controllers]
        return " ".join(a for a in active if a != "-") or "-"

    def controller_squeeze() -> float:
        """Strongest squeeze command across the active input devices."""
        value = 0.0
        for controller in controllers:
            getter = getattr(controller, "squeeze", None)
            if getter is not None:
                value = max(value, float(getter()))
        return value

    # --squeeze_binary: Schmitt-trigger the raw trigger travel into 0 / on_value.
    # Hysteresis (0.6 up, 0.4 down), not a single threshold: a trigger held near
    # the switching point would otherwise flutter the grip open mid-carry.
    _SQUEEZE_ENGAGE = 0.6
    _SQUEEZE_RELEASE = 0.4
    _squeeze_latched = False

    def shape_squeeze(raw: float) -> float:
        nonlocal _squeeze_latched
        if not args_cli.squeeze_binary:
            return raw
        if _squeeze_latched:
            if raw < _SQUEEZE_RELEASE:
                _squeeze_latched = False
        elif raw > _SQUEEZE_ENGAGE:
            _squeeze_latched = True
        return float(args_cli.squeeze_on_value) if _squeeze_latched else 0.0

    _squeeze_sinks = [
        r for r in getattr(teleop_interface, "_retargeters", []) if hasattr(r, "set_squeeze")
    ]
    for _sink in _squeeze_sinks:
        _sink._squeeze_scale = float(args_cli.squeeze_scale)
    if not _squeeze_sinks:
        print(
            "[squeeze] no retargeter accepts a squeeze command — grip force will come "
            "only from your own hand curl, which is usually too shallow to hold anything.",
            flush=True,
        )

    def push_squeeze(value: float) -> None:
        for retargeter in _squeeze_sinks:
            retargeter.set_squeeze(value)

    # --- grip instrumentation -------------------------------------------------
    # Grip force on a position-controlled hand is clip(k_p*err - k_d*qdot, +-limit)
    # where err = commanded - achieved. Reporting it turns "does the grasp hold?"
    # into a measurement during the session instead of a guess afterwards.
    _FINGER_EFFORT_LIMIT = 8.5  # N.m, per green_challenge's actuator config
    _upper_term = base_env.action_manager.get_term("upper_body_ik")
    _finger_ids = list(_upper_term._hand_joint_ids)
    _finger_names = list(_upper_term._hand_joint_names)
    _finger_gains: list[tuple[float, float]] = []
    _xr_actuator = robot.actuators.get("xr_upper")
    for _name in _finger_names:
        if _xr_actuator is not None and _name in _xr_actuator.joint_names:
            _li = list(_xr_actuator.joint_names).index(_name)
            _finger_gains.append(
                (
                    float(_xr_actuator.stiffness[0, _li].item()),
                    float(_xr_actuator.damping[0, _li].item()),
                )
            )
        else:
            _finger_gains.append((0.0, 0.0))

    def grip_report() -> str:
        """Per-hand peak *closing* torque, and how many digits are saturated.

        Only positive error (commanded closed of achieved) counts. All finger
        joints run 0/open -> positive/closed, so a negative error means the hand
        is being driven open, which is not grip. That distinction matters here:
        the open-hand command sits at the joint's lower limit while the thumb
        physically rests above it, so an idle hand shows a permanent -8.5 Nm
        opening torque that would otherwise mask every real grasp.
        """
        target = robot.data.joint_pos_target[0, _finger_ids]
        actual = robot.data.joint_pos[0, _finger_ids]
        vel = robot.data.joint_vel[0, _finger_ids]
        lines = []
        for side in ("left", "right"):
            peak_tau = 0.0
            peak_name = ""
            peak_err = 0.0
            saturated = 0
            count = 0
            for idx, name in enumerate(_finger_names):
                if not name.startswith(side):
                    continue
                count += 1
                k_p, k_d = _finger_gains[idx]
                err = float(target[idx].item() - actual[idx].item())
                if err <= 0.0:  # opening, not gripping
                    continue
                tau = min(_FINGER_EFFORT_LIMIT, k_p * err - k_d * float(vel[idx].item()))
                if tau >= _FINGER_EFFORT_LIMIT - 1e-3:
                    saturated += 1
                if tau > peak_tau:
                    peak_tau, peak_err, peak_name = tau, err, name.replace(f"{side}_", "")
            lines.append(
                f"{side[0].upper()}: grip {peak_tau:+.2f}Nm on {peak_name or '-'} "
                f"(err {peak_err:+.3f} rad) sat {saturated}/{count}"
            )
        return " | ".join(lines)

    if args_cli.xr:
        rate_limiter = None
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    print(f"Using teleop device: {teleop_interface}")
    if use_keyboard:
        print(_HELP_KEYBOARD, flush=True)
    if use_gamepad:
        print(_HELP_GAMEPAD, flush=True)
    if args_cli.idle_pause:
        print(
            f"Idle-pause ON: sim freezes whenever the loco command is zero "
            f"(settle {args_cli.idle_pause_settle:.1f}s; stick/WASD or C resumes).",
            flush=True,
        )
    if args_cli.xr and not args_cli.auto_start:
        print("Press START in XR to begin recording (or relaunch with --auto_start).")
    elif args_cli.auto_start:
        print("Auto-start enabled: recording/WASD active immediately.", flush=True)

    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                poll_controllers()
                idle_paused = False
                if (
                    args_cli.idle_pause
                    and running_recording_instance
                    and teleoperation_active
                    and loco_mode == "locomotion"
                ):
                    if any(abs(c) > 1e-9 for c in controller_command()):
                        last_loco_cmd_t = time.time()
                    idle_paused = (
                        last_loco_cmd_t is None
                        or time.time() - last_loco_cmd_t >= args_cli.idle_pause_settle
                    )
                if idle_paused != idle_paused_prev:
                    idle_paused_prev = idle_paused
                    if idle_paused:
                        print(
                            "[idle-pause] loco command zero — sim frozen "
                            "(stick/WASD resumes, C/Create for static teleop)",
                            flush=True,
                        )
                    elif running_recording_instance:
                        # Drop the frozen span from the Hz window so the first
                        # post-resume rate print is not diluted by wall-clock pause.
                        last_hz_t = time.time()
                        hz_window_steps = 0
                        print("[idle-pause] sim resumed", flush=True)
                if running_recording_instance and not idle_paused:
                    step_count += 1
                    hz_window_steps += 1
                    now = time.time()
                    if now - last_hz_t >= 1.0:
                        effective_hz = hz_window_steps / (now - last_hz_t)
                        hz_window_steps = 0
                        last_hz_t = now
                    robot_pos, robot_yaw = robot_root_state(robot)
                    # Latch the left-hand hold from the settled measured pose so
                    # it applies from (nearly) step 0 of every episode, not only
                    # after the first static-mode entry.
                    if (
                        args_cli.disable_left_hand
                        and left_hold_snapshot is None
                        and step_count >= _LEFT_HOLD_SETTLE_STEPS
                    ):
                        left_hold_snapshot = capture_frozen_upper(robot)
                        print(
                            f"[left-hold] latched at step {step_count} from measured pose",
                            flush=True,
                        )
                    vx = vy = wz = 0.0
                    upper_body_mode: UpperBodyMode | None = None
                    neck_yaw_cmd, neck_pitch_cmd = 0.0, 0.0
                    squeeze_cmd = shape_squeeze(controller_squeeze())
                    push_squeeze(squeeze_cmd)

                    if teleoperation_active:
                        if loco_mode == "locomotion":
                            vx, vy, wz = controller_command()
                            if frozen_snapshot is None:
                                frozen_snapshot = capture_frozen_upper(robot, last_teleop_upper)
                                pink_hold.refresh(robot)
                            # Wrist targets still recorded for replay; Pink solve is skipped.
                            upper_body = apply_left_hold(
                                build_upper_from_frozen_snapshot(
                                    robot, frozen_snapshot, base_env.device
                                )
                            )
                            upper_body_mode = "frozen"
                            last_neck_targets = (0.0, 0.0)
                        else:
                            raw = teleop_interface._get_raw_data()
                            if hold_xy_yaw is None:
                                hold_xy_yaw = capture_hold_pose_from_robot(robot)
                            vx, vy, wz = station_keeper.command(robot, hold_xy_yaw)
                            upper_body = torch.cat(
                                [r.retarget(raw) for r in teleop_interface._retargeters],
                                dim=-1,
                            )
                            upper_body = apply_left_hold(upper_body)
                            last_teleop_upper = (
                                upper_body.squeeze(0).clone()
                                if upper_body.ndim > 1
                                else upper_body.clone()
                            )
                            upper_body_mode = "teleop"
                            head_pose = raw.get(DeviceBase.TrackingTarget.HEAD)
                            if head_pose is not None:
                                last_neck_targets = headset_to_neck_targets(
                                    head_pose,
                                    robot_yaw,
                                    yaw_limit=NECK_YAW_LIMIT,
                                    pitch_limit=NECK_PITCH_LIMIT,
                                )
                            # Missing HEAD: hold last neck targets from static mode.

                        neck_yaw_cmd, neck_pitch_cmd = last_neck_targets

                        if upper_body.ndim == 1:
                            upper_body = upper_body.unsqueeze(0)
                        upper_body = upper_body.to(base_env.device).repeat(base_env.num_envs, 1)
                    else:
                        upper_body = idle_upper.unsqueeze(0).repeat(base_env.num_envs, 1)
                        upper_body_mode = "idle"
                        last_neck_targets = (0.0, 0.0)
                        neck_yaw_cmd, neck_pitch_cmd = last_neck_targets

                    legs = policy_leg_targets(net, vx, vy, wz)
                    legs_tensor = torch.as_tensor(legs, dtype=torch.float32, device=base_env.device)
                    legs_tensor = legs_tensor.unsqueeze(0).repeat(base_env.num_envs, 1)
                    neck_tensor = torch.tensor(
                        [[neck_yaw_cmd, neck_pitch_cmd]],
                        dtype=torch.float32,
                        device=base_env.device,
                    ).repeat(base_env.num_envs, 1)
                    actions = pack_sberchel_xr_action(legs_tensor, upper_body, neck_tensor)
                    stash_locomotion_command(base_env, vx, vy, wz)
                    stash_teleop_state(
                        base_env,
                        loco_mode=loco_mode,
                        squeeze=squeeze_cmd,
                        legacy_gains=args_cli.legacy_stiff_teleop,
                    )
                    env.step(actions)
                    # From here on joint_pos_target holds a real command, so the
                    # Pink hold can capture grasp intent instead of measured pose.
                    pink_hold.mark_stepped()

                    should_print = False
                    if args_cli.print_every_n > 0:
                        if teleoperation_active and not first_active_logged:
                            should_print = True
                            first_active_logged = True
                        elif step_count % args_cli.print_every_n == 0:
                            should_print = True
                    if should_print:
                        hold_dbg = (
                            f" hold=({hold_xy_yaw[0]:.2f},{hold_xy_yaw[1]:.2f},{hold_xy_yaw[2]:.2f})"
                            if hold_xy_yaw is not None
                            else ""
                        )
                        print(
                            f"[step {step_count}] mode={loco_mode} upper={upper_body_mode} "
                            f"pos=({robot_pos[0]:.2f},{robot_pos[1]:.2f}) yaw={robot_yaw:.2f} "
                            f"cmd=({vx:.2f},{vy:.2f},{wz:.2f}) keys={controller_pressed()} "
                            f"hz={effective_hz:.1f} "
                            f"neck=({neck_yaw_cmd:.2f},{neck_pitch_cmd:.2f}){hold_dbg}",
                            flush=True,
                        )
                        if effective_hz > 0.0 and effective_hz < 60.0 and loco_mode == "locomotion":
                            print(
                                f"[warn] loco wall-clock {effective_hz:.1f} Hz "
                                f"(want ~150). Try --loco_render_interval 10",
                                flush=True,
                            )
                        elif effective_hz > 0.0 and effective_hz < 60.0 and loco_mode == "static":
                            print(
                                f"[warn] static wall-clock {effective_hz:.1f} Hz "
                                f"(Pink IK + XR); expected slower than loco.",
                                flush=True,
                            )
                    if (
                        args_cli.grip_report_every_n > 0
                        and step_count % args_cli.grip_report_every_n == 0
                    ):
                        print(
                            f"[grip {step_count}] squeeze={squeeze_cmd:.2f} {grip_report()}",
                            flush=True,
                        )
                else:
                    base_env.sim.render()

                if should_reset or should_discard_reset:
                    discard = should_discard_reset
                    if discard:
                        base_env.recorder_manager.cfg.export_in_record_pre_reset = False
                        discard_viewport_video_frames(env)
                    env.reset()
                    # Targets survive reset while the robot is teleported back to
                    # its init pose, so they no longer describe the robot.
                    pink_hold.invalidate()
                    if discard:
                        base_env.recorder_manager.cfg.export_in_record_pre_reset = True
                    hold_xy_yaw = None
                    sim_checkpoint = None  # belongs to the finished episode
                    # Re-latch after the reset pose settles; the stale hold's
                    # pelvis-relative wrist belongs to the previous episode.
                    left_hold_snapshot = None
                    enter_locomotion_mode(capture_frozen=False)
                    last_neck_targets = (0.0, 0.0)
                    (
                        frozen_snapshot,
                        last_teleop_upper,
                        idle_upper,
                    ) = reset_teleop_state(
                        env=base_env,
                        net=net,
                        robot=robot,
                        teleop_interface=teleop_interface,
                        loco_mode="locomotion",
                    )
                    pink_hold.refresh(robot)
                    if anchor_teleporter is not None:
                        # Robot is back at spawn; restore the start-of-episode XR view.
                        anchor_teleporter.teleport_to_robot(robot)
                    should_reset = False
                    should_discard_reset = False
                    first_active_logged = False
                    step_count = 0
                    demo_count = exported_demo_count(base_env)
                    if discard:
                        print(
                            f"Discard reset complete (episode not saved). "
                            f"Recorded {demo_count} episode(s).",
                            flush=True,
                        )
                    else:
                        print(
                            f"Environment reset complete. Recorded {demo_count} episode(s).",
                            flush=True,
                        )

                if args_cli.num_demos > 0 and exported_demo_count(base_env) >= args_cli.num_demos:
                    print(f"Recorded {exported_demo_count(base_env)} demonstrations. Exiting.")
                    break

                if rate_limiter is not None and running_recording_instance:
                    rate_limiter.sleep(base_env)
        except Exception as exc:
            logger.error("Error during simulation step: %s", exc)
            break

    for controller in controllers:
        controller.close()
    env.close()
    print(f"Recording session complete. Demonstrations saved to: {args_cli.dataset_file}")
    if args_cli.video:
        video_folder = Path(args_cli.video_dir) if args_cli.video_dir else Path(output_dir) / "videos"
        print(f"Viewport videos saved under: {video_folder}")

if __name__ == "__main__":
    main()
    simulation_app.close()
