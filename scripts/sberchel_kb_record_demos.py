"""Record Sberchel XR demos with WASD keyboard locomotion.

Upper-body teleop is XR hand tracking; legs use the zest locomotion policy
driven by keyboard velocity commands (like ``demo_sberchel_kb.py``). Press **C**
to toggle locomotion (WASD + frozen upper body) vs static teleop (station-keep
at the latch pose + live XR hands).

Episodes export on RESET/R. Press **X** to discard the current episode and reset
without writing to HDF5.

Usage:
    python scripts/sberchel_kb_record_demos.py \\
        --task Isaac-Demo-Sberchel-XR-v0 \\
        --teleop_device handtracking \\
        --enable_pinocchio \\
        --dataset_file ./datasets/sberchel_kb_demos.hdf5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Callable
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

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.devices.openxr import remove_camera_configs
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
    robot_root_state,
    station_keep_command,
    sync_retargeter_idle_from_robot,
)
from dsynth_isaac.robots.sberchel_cfg import apply_sberchel_xr_teleop_pd_gains
from dsynth_isaac.tasks.manager_based.sberchel_base_env_cfg import (
    LEGS_ACTION_DIM,
    policy_leg_targets,
)
from green_challenge.policy.network import Network

logger = logging.getLogger(__name__)

LocoMode = Literal["locomotion", "static"]

_HELP = """
==================== KB RECORD CONTROLS ====================
  W forward    S backward    A turn left    D turn right
  Q strafe L   E strafe R    (release = stop legs in loco mode)
  C  toggle locomotion (WASD + frozen upper) / static teleop
  R  save episode + reset
  X  discard episode + reset (no HDF5 write)
  XR START / STOP  begin or pause recording
  >>> Click the Isaac Sim window so it captures your keys <<<
===========================================================
"""


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


class RecordKeyboard:
    """WASDQE hold-to-move plus edge-triggered C / R / X via Isaac Sim (carb)."""

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
        self._pressed: set[str] = set()
        self._ok = False
        try:
            self._appwindow = omni.appwindow.get_default_app_window()
            self._input = carb.input.acquire_input_interface()
            self._keyboard = self._appwindow.get_keyboard()
            self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)
            self._ok = True
        except Exception as exc:  # headless / no window -> no keyboard
            print(
                f"[KB] keyboard unavailable ({exc!r}); legs will stand. "
                f"Run WITHOUT --headless and click the Isaac window.",
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
                if name not in self._pressed:
                    if name == "C" and self._on_toggle_mode is not None:
                        self._on_toggle_mode()
                    elif name == "R" and self._on_save_reset is not None:
                        self._on_save_reset()
                    elif name == "X" and self._on_discard_reset is not None:
                        self._on_discard_reset()
                self._pressed.add(name)
            else:
                self._pressed.discard(name)
        except Exception:
            pass
        return True

    def command(self) -> tuple[float, float, float]:
        """Return (lin_vel_x, lin_vel_y, ang_vel_z) from held WASDQE keys."""
        p = self._pressed
        vx = (self._speed if "W" in p else 0.0) - (self._speed if "S" in p else 0.0)
        wz = (self._turn if "A" in p else 0.0) - (self._turn if "D" in p else 0.0)
        vy = (self._strafe if "Q" in p else 0.0) - (self._strafe if "E" in p else 0.0)
        return vx, vy, wz

    def close(self) -> None:
        if self._ok:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            except Exception:
                pass


def setup_output_directories() -> tuple[str, str]:
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir, output_file_name


def exported_demo_count(env) -> int:
    """Total episodes written to HDF5."""
    rm = env.recorder_manager
    return rm.exported_successful_episode_count + rm.exported_failed_episode_count


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
            env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    env_cfg.recorders = SberchelXRRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    env_cfg.recorders.export_in_record_pre_reset = True
    env_cfg.recorders.export_in_close = True
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
) -> tuple[FrozenUpperSnapshot | None, torch.Tensor | None, torch.Tensor]:
    """Reset policy and teleop state after env.reset()."""
    net.reset()
    apply_sberchel_xr_teleop_pd_gains(robot)
    teleop_interface.reset()
    sync_retargeter_idle_from_robot(teleop_interface, robot)
    idle_upper = build_idle_upper_from_robot(robot, env.device)
    return None, None, idle_upper


def main() -> None:
    output_dir, output_file_name = setup_output_directories()
    env_cfg = create_environment_config(output_dir, output_file_name)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.sim.reset()
    env.reset(seed=args_cli.seed)

    net = Network(env, type="zest", device="cpu", closed_kinematics=False)
    net.reset()
    apply_sberchel_xr_teleop_pd_gains(env.scene["robot"])
    robot = env.scene["robot"]
    upper_body_dim = env.action_manager.total_action_dim - LEGS_ACTION_DIM
    print(
        f">>> action dim = {env.action_manager.total_action_dim} "
        f"(legs {LEGS_ACTION_DIM} + upper_body {upper_body_dim})",
        flush=True,
    )

    should_reset = False
    should_discard_reset = False
    teleoperation_active = not args_cli.xr
    running_recording_instance = not args_cli.xr
    step_count = 0
    first_active_logged = False
    loco_mode: LocoMode = "locomotion"
    hold_xy_yaw: tuple[float, float, float] | None = None
    frozen_snapshot: FrozenUpperSnapshot | None = None
    last_teleop_upper: torch.Tensor | None = None

    def reset_env() -> None:
        nonlocal should_reset
        should_reset = True
        print("Reset requested — episode will be saved and environment reset on next step")

    def discard_reset_env() -> None:
        nonlocal should_discard_reset
        should_discard_reset = True
        print("Discard reset requested — episode will NOT be saved; environment reset on next step")

    def toggle_loco_mode() -> None:
        nonlocal loco_mode, hold_xy_yaw, frozen_snapshot
        if not teleoperation_active:
            return
        if loco_mode == "locomotion":
            loco_mode = "static"
            hold_xy_yaw = capture_hold_pose_from_robot(robot)
            sync_retargeter_idle_from_robot(teleop_interface, robot)
            frozen_snapshot = None
            print(
                f"Mode: STATIC teleop (station-keep at "
                f"xy=({hold_xy_yaw[0]:.3f}, {hold_xy_yaw[1]:.3f}) yaw={hold_xy_yaw[2]:.3f})",
                flush=True,
            )
        else:
            loco_mode = "locomotion"
            frozen_snapshot = capture_frozen_upper(robot, last_teleop_upper)
            hold_xy_yaw = None
            print("Mode: LOCOMOTION (WASD + frozen upper body)", flush=True)

    def start_recording_instance() -> None:
        nonlocal running_recording_instance, teleoperation_active, first_active_logged
        nonlocal loco_mode, hold_xy_yaw, frozen_snapshot, last_teleop_upper
        running_recording_instance = True
        teleoperation_active = True
        first_active_logged = False
        loco_mode = "locomotion"
        hold_xy_yaw = None
        frozen_snapshot = None
        last_teleop_upper = None
        sync_retargeter_idle_from_robot(teleop_interface, robot)
        print("Recording started (WASD locomotion + XR upper body; press C for static teleop)")

    def stop_recording_instance() -> None:
        nonlocal running_recording_instance, teleoperation_active
        nonlocal loco_mode, hold_xy_yaw, frozen_snapshot, last_teleop_upper
        running_recording_instance = False
        teleoperation_active = False
        loco_mode = "locomotion"
        hold_xy_yaw = None
        frozen_snapshot = None
        last_teleop_upper = None
        print("Recording paused")

    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_env,
        "START": start_recording_instance,
        "STOP": stop_recording_instance,
        "RESET": reset_env,
    }

    # Head tracking not required for keyboard locomotion.
    teleop_interface = create_sberchel_xr_teleop_interface(
        env_cfg, args_cli.teleop_device, teleoperation_callbacks, require_head_tracking=False
    )
    teleop_interface.reset()
    sync_retargeter_idle_from_robot(teleop_interface, robot)
    idle_upper = build_idle_upper_from_robot(robot, env.device)

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
    if args_cli.xr:
        print("Press START in XR to begin recording.")

    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                if running_recording_instance:
                    step_count += 1
                    robot_pos, robot_yaw = robot_root_state(robot)
                    vx = vy = wz = 0.0
                    upper_body_mode: UpperBodyMode | None = None

                    if teleoperation_active:
                        raw = teleop_interface._get_raw_data()

                        if loco_mode == "locomotion":
                            vx, vy, wz = kb.command()
                            if frozen_snapshot is None:
                                frozen_snapshot = capture_frozen_upper(robot, last_teleop_upper)
                            upper_body = build_upper_from_frozen_snapshot(
                                robot, frozen_snapshot, env.device
                            )
                            upper_body_mode = "frozen"
                        else:
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

                        if upper_body.ndim == 1:
                            upper_body = upper_body.unsqueeze(0)
                        upper_body = upper_body.to(env.device).repeat(env.num_envs, 1)
                    else:
                        upper_body = idle_upper.unsqueeze(0).repeat(env.num_envs, 1)
                        upper_body_mode = "idle"

                    legs = policy_leg_targets(net, vx, vy, wz)
                    legs_tensor = torch.as_tensor(legs, dtype=torch.float32, device=env.device)
                    legs_tensor = legs_tensor.unsqueeze(0).repeat(env.num_envs, 1)
                    actions = torch.cat([legs_tensor, upper_body], dim=-1)
                    stash_locomotion_command(env, vx, vy, wz)
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
                            f"cmd=({vx:.2f},{vy:.2f},{wz:.2f}){hold_dbg}",
                            flush=True,
                        )
                else:
                    env.sim.render()

                if should_reset or should_discard_reset:
                    discard = should_discard_reset
                    if discard:
                        env.recorder_manager.cfg.export_in_record_pre_reset = False
                    env.reset()
                    if discard:
                        env.recorder_manager.cfg.export_in_record_pre_reset = True
                    hold_xy_yaw = None
                    loco_mode = "locomotion"
                    (
                        frozen_snapshot,
                        last_teleop_upper,
                        idle_upper,
                    ) = reset_teleop_state(
                        env=env, net=net, robot=robot, teleop_interface=teleop_interface
                    )
                    should_reset = False
                    should_discard_reset = False
                    first_active_logged = False
                    step_count = 0
                    demo_count = exported_demo_count(env)
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

                if args_cli.num_demos > 0 and exported_demo_count(env) >= args_cli.num_demos:
                    print(f"Recorded {exported_demo_count(env)} demonstrations. Exiting.")
                    break

                if rate_limiter is not None and running_recording_instance:
                    rate_limiter.sleep(env)
        except Exception as exc:
            logger.error("Error during simulation step: %s", exc)
            break

    kb.close()
    env.close()
    print(f"Recording session complete. Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
