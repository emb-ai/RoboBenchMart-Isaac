"""Record Sberchel XR demos with VRD-style v3 head-follow locomotion.

Stages (gate legs + upper body):
  cruise / precision — ZEST chase (breadcrumbs, velocity feed-forward, turn-then-walk);
                     upper body frozen with last teleop finger commands (carry-safe).
  settled — live XR teleop; legs zero inside tight band, soft station-keep if residual
            outside a looser band (no root teleport).

Usage:
    python scripts/sberchel_record_demos_v3.py \\
        --task Isaac-Demo-Sberchel-XR-v0 \\
        --teleop_device handtracking \\
        --enable_pinocchio \\
        --dataset_file ./datasets/sberchel_xr_demos.hdf5
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable

from isaaclab.app import AppLauncher

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from dsynth_isaac.green_challenge_path import ensure_green_challenge_importable  # noqa: E402
from dsynth_isaac.sim_defaults import apply_sberchel_cli_defaults  # noqa: E402

ensure_green_challenge_importable()

from dsynth_isaac.teleop.sberchel_head_follow_v3 import add_head_follow_v3_cli_args  # noqa: E402

parser = argparse.ArgumentParser(
    description="Record Sberchel XR demos (v3 VRD-style cruise/precision/settled)."
)
parser.add_argument("--task", type=str, default="Isaac-Demo-Sberchel-XR-v0", help="Gym task id.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/sberchel_xr_demos.hdf5",
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
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed. Defaults to a fresh entropy-based seed each run (printed at startup); "
    "pass an explicit value to reproduce a session's start poses.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio (required for Pink IK / hand tracking).",
)
add_head_follow_v3_cli_args(parser)
AppLauncher.add_app_launcher_args(parser)
# Sberchel runs on green_challenge's CPU physics by default; --device overrides.
apply_sberchel_cli_defaults(parser)
args_cli = parser.parse_args()

if args_cli.seed is None:
    args_cli.seed = int.from_bytes(os.urandom(4), "little") % (2**31)
    print(f"[INFO] No --seed given; using seed {args_cli.seed} (pass --seed {args_cli.seed} to reproduce this session)")

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

import omni.ui as ui

from isaaclab.devices.device_base import DeviceBase
from dsynth_isaac.robots.sberchel_camera_cfg import strip_camera_configs
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode
from isaaclab_mimic.ui.instruction_display import InstructionDisplay
from isaaclab_tasks.utils import parse_env_cfg

import dsynth_isaac.tasks.manager_based  # noqa: F401

from dsynth_isaac.recorders import (
    SberchelXRRecorderManagerCfg,
    enable_async_episode_export,
    stash_locomotion_command,
)

from dsynth_isaac.teleop.sberchel_head_follow import (
    UpperBodyMode,
    build_idle_upper_from_robot,
    create_sberchel_xr_teleop_interface,
    robot_root_state,
    sync_retargeter_idle_from_robot,
)
from dsynth_isaac.teleop.sberchel_head_follow_v3 import (
    HeadFollowV3Params,
    HeadFollowV3Runtime,
    LocoDebug,
    advance_locomotion,
    on_stage_transition,
    print_debug_block_v3,
    upper_body_for_stage,
    validate_v3_params,
)
from dsynth_isaac.robots.sberchel_cfg import apply_sberchel_xr_teleop_pd_gains
from dsynth_isaac.tasks.manager_based.sberchel_base_env_cfg import (
    LEGS_ACTION_DIM,
    policy_leg_targets,
)
from green_challenge.policy.network import Network

logger = logging.getLogger(__name__)


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


def setup_output_directories() -> tuple[str, str]:
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    # The HDF5 handler opens the dataset with mode "w" (truncate), so reusing a
    # filename would wipe previously recorded demos — auto-suffix instead.
    base_name = output_file_name
    suffix = 0
    while os.path.exists(os.path.join(output_dir, output_file_name + ".hdf5")):
        suffix += 1
        output_file_name = f"{base_name}_{suffix:03d}"
    if suffix:
        print(f"Dataset file {base_name}.hdf5 already exists; writing to {output_file_name}.hdf5 instead")
    return output_dir, output_file_name


def repo_git_commit() -> str:
    """Current commit hash of this repo, with a ``-dirty`` marker for uncommitted changes."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = (
            subprocess.run(
                ["git", "diff", "--quiet", "HEAD"],
                cwd=_repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            != 0
        )
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


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
            env_cfg = strip_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    env_cfg.recorders = SberchelXRRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    env_cfg.recorders.export_in_record_pre_reset = True
    env_cfg.recorders.export_in_close = True
    return env_cfg


def setup_ui(
    env,
    *,
    demo_label_text: str,
) -> InstructionDisplay:
    """XR headset panels + optional desktop labels (matches Isaac Lab record_demos)."""
    instruction_display = InstructionDisplay(args_cli.xr)
    if not args_cli.xr:
        window = EmptyWindow(env, "Sberchel Record")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(demo_label_text)
            instruction_display.set_labels(ui.Label(""), demo_label)
    instruction_display.show_demo(demo_label_text)
    return instruction_display


def main() -> None:
    params = HeadFollowV3Params.from_namespace(args_cli)
    validate_v3_params(params)

    output_dir, output_file_name = setup_output_directories()
    env_cfg = create_environment_config(output_dir, output_file_name)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    enable_async_episode_export(env)
    # Stamp session provenance (seed + code version) into the dataset's env_args attrs.
    session_meta = {"seed": args_cli.seed, "git_commit": repo_git_commit()}
    for handler in (
        env.recorder_manager._dataset_file_handler,
        env.recorder_manager._failed_episode_dataset_file_handler,
    ):
        if handler is not None:
            handler.add_env_args(session_meta)
    env.sim.reset()
    env.reset(seed=args_cli.seed)

    net = Network(env, type="zest", device="cpu", closed_kinematics=True)
    net.reset()
    apply_sberchel_xr_teleop_pd_gains(env.scene["robot"])
    robot = env.scene["robot"]
    upper_body_dim = env.action_manager.total_action_dim - LEGS_ACTION_DIM
    print(
        f">>> action dim = {env.action_manager.total_action_dim} "
        f"(legs {LEGS_ACTION_DIM} + upper_body {upper_body_dim})",
        flush=True,
    )

    runtime = HeadFollowV3Runtime.from_params(params)
    should_reset = False
    teleoperation_active = not args_cli.xr
    running_recording_instance = not args_cli.xr
    need_session_init = False
    step_count = 0
    first_active_logged = False
    loco_dt = 1.0 / float(args_cli.step_hz if args_cli.step_hz > 0 else 30)

    def reset_env() -> None:
        nonlocal should_reset
        should_reset = True
        print("Reset requested — episode will be saved and environment reset on next step")

    def start_recording_instance() -> None:
        nonlocal running_recording_instance, teleoperation_active, first_active_logged, need_session_init
        running_recording_instance = True
        teleoperation_active = True
        first_active_logged = False
        need_session_init = True
        runtime.clear_session()
        sync_retargeter_idle_from_robot(teleop_interface, robot)
        print("Recording started (VRD-style v3 cruise/precision/settled)")

    def stop_recording_instance() -> None:
        nonlocal running_recording_instance, teleoperation_active, need_session_init
        running_recording_instance = False
        teleoperation_active = False
        need_session_init = False
        runtime.clear_session()
        print("Recording paused")

    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_env,
        "START": start_recording_instance,
        "STOP": stop_recording_instance,
        "RESET": reset_env,
    }

    teleop_interface = create_sberchel_xr_teleop_interface(
        env_cfg, args_cli.teleop_device, teleoperation_callbacks
    )
    teleop_interface.reset()
    sync_retargeter_idle_from_robot(teleop_interface, robot)
    idle_upper = build_idle_upper_from_robot(robot, env.device)

    demo_label_text = f"Recorded {exported_demo_count(env)} episode(s)."
    instruction_display = setup_ui(env, demo_label_text=demo_label_text)

    if args_cli.xr:
        rate_limiter = None
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    print(f"Using teleop device: {teleop_interface}")
    print(
        "Legs v3: breadcrumbs + headset-velocity FF + cruise/precision/settled "
        "(deadband-mapped ZEST cmds, turn-then-walk, no root teleport). "
        "Upper body: teleop in settled; frozen (last finger cmds) in cruise/precision. "
        "Press RESET/R to save the current episode and start a new one."
    )
    if args_cli.xr:
        print("Press START in XR to begin recording. Press RESET/R to save and reset.")

    current_recorded_demo_count = exported_demo_count(env)

    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                vx = vy = wz = 0.0
                upper_body_mode: UpperBodyMode | None = None
                loco_debug: LocoDebug | None = None
                stage_dbg = runtime.stage

                if running_recording_instance:
                    step_count += 1
                    robot_pos, robot_yaw = robot_root_state(robot)
                    raw_head_pose = None
                    filtered_head_pose = None

                    if teleoperation_active:
                        raw = teleop_interface._get_raw_data()
                        raw_head_pose = raw.get(DeviceBase.TrackingTarget.HEAD)
                        if raw_head_pose is not None:
                            if need_session_init:
                                filtered_head_pose = runtime.filter.reset(raw_head_pose)
                                runtime.begin_session(filtered_head_pose)
                                need_session_init = False
                            else:
                                filtered_head_pose = runtime.filter.update(raw_head_pose)

                            prev_stage = runtime.stage
                            vx, vy, wz, stage_dbg, loco_debug = advance_locomotion(
                                runtime,
                                robot,
                                filtered_head_pose,
                                params,
                                loco_dt,
                            )
                            if prev_stage != stage_dbg:
                                on_stage_transition(
                                    runtime,
                                    robot,
                                    teleop_interface,
                                    prev_stage,
                                    stage_dbg,
                                )

                            upper_body, upper_body_mode = upper_body_for_stage(
                                runtime,
                                robot,
                                stage_dbg,
                                raw_xr_data=raw,
                                teleop_interface=teleop_interface,
                                device=env.device,
                                idle_upper=idle_upper,
                            )
                            upper_body = upper_body.repeat(env.num_envs, 1)
                        else:
                            upper_body = idle_upper.unsqueeze(0).repeat(env.num_envs, 1)
                            upper_body_mode = "idle"
                            vx = vy = wz = 0.0
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
                    if params.print_every_n > 0:
                        if teleoperation_active and not first_active_logged:
                            should_print = True
                            first_active_logged = True
                        elif step_count % params.print_every_n == 0:
                            should_print = True
                    if should_print:
                        print_debug_block_v3(
                            step_count,
                            active=teleoperation_active,
                            robot_pos=robot_pos,
                            robot_yaw=robot_yaw,
                            raw_head_pose=raw_head_pose,
                            filtered_head_pose=filtered_head_pose,
                            debug=loco_debug,
                            upper_body_mode=upper_body_mode,
                        )
                else:
                    env.sim.render()

                if exported_demo_count(env) > current_recorded_demo_count:
                    current_recorded_demo_count = exported_demo_count(env)
                    demo_label_text = f"Recorded {current_recorded_demo_count} episode(s)."
                    instruction_display.show_demo(demo_label_text)

                if should_reset:
                    env.reset()
                    net.reset()
                    apply_sberchel_xr_teleop_pd_gains(robot)
                    teleop_interface.reset()
                    sync_retargeter_idle_from_robot(teleop_interface, robot)
                    runtime.clear_session()
                    # Re-init loco session on next head frame if teleop is still live.
                    # Previously False left stage/filter/breadcrumbs cleared without
                    # begin_session; post-reset cmds ran but the robot often did not
                    # re-acquire (debug: robot_yaw stuck while dist grew to >3 m).
                    need_session_init = bool(
                        teleoperation_active and running_recording_instance
                    )
                    idle_upper = build_idle_upper_from_robot(robot, env.device)
                    should_reset = False
                    first_active_logged = False
                    step_count = 0
                    demo_count = exported_demo_count(env)
                    demo_label_text = f"Recorded {demo_count} episode(s)."
                    instruction_display.show_demo(demo_label_text)
                    print(f"Environment reset complete. Recorded {demo_count} episode(s).")

                if args_cli.num_demos > 0 and exported_demo_count(env) >= args_cli.num_demos:
                    print(f"Recorded {exported_demo_count(env)} demonstrations. Exiting.")
                    break

                if rate_limiter is not None and running_recording_instance:
                    rate_limiter.sleep(env)
        except Exception as exc:
            logger.error("Error during simulation step: %s", exc)
            break

    env.close()
    print(f"Recording session complete. Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
