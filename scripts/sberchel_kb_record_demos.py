"""Record Sberchel XR demos with WASD keyboard locomotion.

Upper-body teleop is XR hand tracking; legs use the zest locomotion policy
driven by keyboard velocity commands (like ``demo_sberchel_kb.py``). Press **C**
to toggle locomotion (WASD + soft frozen upper body) vs static teleop (station-keep
at the latch pose + live XR hands + headset neck look).

OpenXR often steals window focus (carb WASD drops). This script polls X11 keymap
globally so hold-to-move works while the headset is focused. During locomotion,
arm PD is lowered to policy-scale gains — stiff XR teleop PD (~3000) on frozen
arms fights balance and makes walking shuffle or stall.

Episodes export on RESET/R. Press **X** to discard the current episode and reset
without writing to HDF5.

With ``--video``, also records the viewport (``rgb_array``) via Gymnasium
``RecordVideo``; each RESET/R saves an MP4 (discard/X skips the video write).

Usage:
    python scripts/sberchel_kb_record_demos.py \\
        --task Isaac-Demo-Sberchel-XR-v0 \\
        --teleop_device handtracking \\
        --enable_pinocchio \\
        --dataset_file ./datasets/sberchel_kb_demos.hdf5 \\
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
from dsynth_isaac.robots.sberchel_camera_cfg import strip_camera_configs
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import DatasetExportMode
from isaaclab_tasks.utils import parse_env_cfg

import dsynth_isaac.tasks.manager_based  # noqa: F401

from dsynth_isaac.recorders import SberchelXRRecorderManagerCfg, stash_locomotion_command

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
    station_keep_command,
    sync_retargeter_idle_from_robot,
)
from dsynth_isaac.robots.sberchel_cfg import (
    apply_sberchel_loco_upper_pd_gains,
    apply_sberchel_xr_teleop_pd_gains,
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

_HELP = """
==================== KB RECORD CONTROLS ====================
  W forward    S backward    A turn left    D turn right
  Q strafe L   E strafe R    (release = stop legs in loco mode)
  C  toggle locomotion (WASD + Pink hold-off) / static teleop
     (static: XR hands + headset neck; Pink IK on, slower)
  R  save episode + reset (HDF5; viewport MP4 if --video)
  X  discard episode + reset (no HDF5 / no MP4 write)
  XR START / STOP  begin or pause recording (or pass --auto_start)
  Keys: X11 global poll (works while headset has focus) + Isaac window
  Tip: if loco hz still low, raise --loco_render_interval (e.g. 10)
===========================================================
"""

_MOVE_KEYS = ("W", "A", "S", "D", "Q", "E")
_EDGE_KEYS = ("C", "R", "X")


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

    def enable(self, robot) -> None:
        ids = self._term._controlled_joint_ids
        self._targets = robot.data.joint_pos[:, ids].detach().clone()
        if not self._active:
            self._term.apply_actions = self._apply_hold  # type: ignore[method-assign]
            self._active = True

    def refresh(self, robot) -> None:
        """Re-capture hold pose (call after reset / mode entry)."""
        ids = self._term._controlled_joint_ids
        self._targets = robot.data.joint_pos[:, ids].detach().clone()

    def disable(self) -> None:
        if self._active:
            self._term.apply_actions = self._orig_apply  # type: ignore[method-assign]
            self._active = False

    def _apply_hold(self) -> None:
        if self._targets is None:
            self._orig_apply()
            return
        self._term._asset.set_joint_position_target(self._targets, self._term._controlled_joint_ids)


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
    """WASDQE hold-to-move plus edge-triggered C / R / X.

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
        """Process edge-triggered C / R / X; call every loop iteration."""
        pressed = self._pressed_now()
        edge_now = pressed & set(_EDGE_KEYS)
        newly = edge_now - self._edge_prev
        if "C" in newly and self._on_toggle_mode is not None:
            self._on_toggle_mode()
        if "R" in newly and self._on_save_reset is not None:
            self._on_save_reset()
        if "X" in newly and self._on_discard_reset is not None:
            self._on_discard_reset()
        self._edge_prev = edge_now

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


def create_environment_config(output_dir: str, output_file_name: str) -> ManagerBasedRLEnvCfg:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(f"Expected ManagerBasedRLEnvCfg, got {type(env_cfg).__name__}")

    env_cfg.env_name = args_cli.task.split(":")[-1]
    env_cfg.seed = args_cli.seed
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    if args_cli.xr:
        if not args_cli.enable_cameras:
            env_cfg = strip_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

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
    if loco_mode == "static":
        apply_sberchel_xr_teleop_pd_gains(robot)
    else:
        apply_sberchel_loco_upper_pd_gains(robot)
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
    # Start in locomotion mode: soft upper PD (stiff XR gains fight WASD walking).
    apply_sberchel_loco_upper_pd_gains(base_env.scene["robot"])
    robot = base_env.scene["robot"]
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

    should_reset = False
    should_discard_reset = False
    teleoperation_active = (not args_cli.xr) or args_cli.auto_start
    running_recording_instance = (not args_cli.xr) or args_cli.auto_start
    step_count = 0
    first_active_logged = False
    loco_mode: LocoMode = "locomotion"
    hold_xy_yaw: tuple[float, float, float] | None = None
    frozen_snapshot: FrozenUpperSnapshot | None = None
    last_teleop_upper: torch.Tensor | None = None
    last_neck_targets: tuple[float, float] = (0.0, 0.0)
    last_hz_t = time.time()
    hz_window_steps = 0
    effective_hz = 0.0

    def enter_locomotion_mode(*, capture_frozen: bool) -> None:
        nonlocal loco_mode, hold_xy_yaw, frozen_snapshot, last_neck_targets
        loco_mode = "locomotion"
        hold_xy_yaw = None
        last_neck_targets = (0.0, 0.0)
        if capture_frozen:
            frozen_snapshot = capture_frozen_upper(robot, last_teleop_upper)
        apply_sberchel_loco_upper_pd_gains(robot)
        pink_hold.enable(robot)
        set_render_interval(base_env, args_cli.loco_render_interval)

    def enter_static_mode() -> None:
        nonlocal loco_mode, hold_xy_yaw, frozen_snapshot, last_neck_targets
        loco_mode = "static"
        hold_xy_yaw = capture_hold_pose_from_robot(robot)
        sync_retargeter_idle_from_robot(teleop_interface, robot)
        frozen_snapshot = None
        last_neck_targets = (0.0, 0.0)
        pink_hold.disable()
        apply_sberchel_xr_teleop_pd_gains(robot)
        set_render_interval(base_env, 1)

    def reset_env() -> None:
        nonlocal should_reset
        should_reset = True
        print("Reset requested — episode will be saved and environment reset on next step")

    def discard_reset_env() -> None:
        nonlocal should_discard_reset
        should_discard_reset = True
        print("Discard reset requested — episode will NOT be saved; environment reset on next step")

    def toggle_loco_mode() -> None:
        if not teleoperation_active:
            return
        if loco_mode == "locomotion":
            enter_static_mode()
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

    kb = RecordKeyboard(
        args_cli.speed,
        args_cli.turn,
        args_cli.strafe,
        on_toggle_mode=toggle_loco_mode,
        on_save_reset=reset_env,
        on_discard_reset=discard_reset_env,
    )

    if args_cli.xr:
        rate_limiter = None
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    print(f"Using teleop device: {teleop_interface}")
    print(_HELP, flush=True)
    if args_cli.xr and not args_cli.auto_start:
        print("Press START in XR to begin recording (or relaunch with --auto_start).")
    elif args_cli.auto_start:
        print("Auto-start enabled: recording/WASD active immediately.", flush=True)

    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                kb.poll()
                if running_recording_instance:
                    step_count += 1
                    hz_window_steps += 1
                    now = time.time()
                    if now - last_hz_t >= 1.0:
                        effective_hz = hz_window_steps / (now - last_hz_t)
                        hz_window_steps = 0
                        last_hz_t = now
                    robot_pos, robot_yaw = robot_root_state(robot)
                    vx = vy = wz = 0.0
                    upper_body_mode: UpperBodyMode | None = None
                    neck_yaw_cmd, neck_pitch_cmd = 0.0, 0.0

                    if teleoperation_active:
                        if loco_mode == "locomotion":
                            vx, vy, wz = kb.command()
                            if frozen_snapshot is None:
                                frozen_snapshot = capture_frozen_upper(robot, last_teleop_upper)
                                pink_hold.refresh(robot)
                            # Wrist targets still recorded for replay; Pink solve is skipped.
                            upper_body = build_upper_from_frozen_snapshot(
                                robot, frozen_snapshot, base_env.device
                            )
                            upper_body_mode = "frozen"
                            last_neck_targets = (0.0, 0.0)
                        else:
                            raw = teleop_interface._get_raw_data()
                            if hold_xy_yaw is None:
                                hold_xy_yaw = capture_hold_pose_from_robot(robot)
                            vx, vy, wz = station_keep_command(
                                robot,
                                hold_xy_yaw,
                                lin_gain=args_cli.station_lin_gain,
                                ang_gain=args_cli.station_ang_gain,
                                max_lin=args_cli.station_max_lin,
                                max_ang=args_cli.station_max_ang,
                                pos_deadzone=0.0,
                                yaw_deadzone=0.0,
                            )
                            upper_body = torch.cat(
                                [r.retarget(raw) for r in teleop_interface._retargeters],
                                dim=-1,
                            )
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
                    env.step(actions)

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
                            f"cmd=({vx:.2f},{vy:.2f},{wz:.2f}) keys={kb.pressed_keys()} "
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
                else:
                    base_env.sim.render()

                if should_reset or should_discard_reset:
                    discard = should_discard_reset
                    if discard:
                        base_env.recorder_manager.cfg.export_in_record_pre_reset = False
                        discard_viewport_video_frames(env)
                    env.reset()
                    if discard:
                        base_env.recorder_manager.cfg.export_in_record_pre_reset = True
                    hold_xy_yaw = None
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

    kb.close()
    env.close()
    print(f"Recording session complete. Demonstrations saved to: {args_cli.dataset_file}")
    if args_cli.video:
        video_folder = Path(args_cli.video_dir) if args_cli.video_dir else Path(output_dir) / "videos"
        print(f"Viewport videos saved under: {video_folder}")

if __name__ == "__main__":
    main()
    simulation_app.close()
