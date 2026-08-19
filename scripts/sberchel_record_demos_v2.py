"""Record Sberchel XR demos with v2 Standing/Pursuing head-follow locomotion.

Standing: live XR teleop + legs station-keep to latched robot pose.
Pursuing: legs follow filtered headset XY and headset view yaw; upper body frozen.

Usage:
    python scripts/sberchel_record_demos_v2.py \\
        --task Isaac-Demo-Sberchel-XR-v0 \\
        --teleop_device handtracking \\
        --enable_pinocchio \\
        --dataset_file ./datasets/sberchel_xr_demos.hdf5
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from collections.abc import Callable

from isaaclab.app import AppLauncher

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from dsynth_isaac.green_challenge_path import ensure_green_challenge_importable  # noqa: E402

ensure_green_challenge_importable()

from dsynth_isaac.teleop.sberchel_head_follow_v2 import add_head_follow_v2_cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Record Sberchel XR demos (v2 Standing/Pursuing locomotion).")
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
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio (required for Pink IK / hand tracking).",
)
add_head_follow_v2_cli_args(parser)
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
    build_upper_from_frozen_snapshot,
    capture_frozen_upper,
    create_sberchel_xr_teleop_interface,
    robot_root_state,
    sync_retargeter_idle_from_robot,
    wrap_angle,
)
from dsynth_isaac.teleop.sberchel_head_follow_v2 import (
    HeadFollowV2Params,
    HeadFollowV2Runtime,
    _agent_debug_log,
    head_ref_delta,
    print_debug_block_v2,
    pursuing_command,
    standing_leg_command,
    update_loco_mode,
    validate_v2_params,
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
    params = HeadFollowV2Params.from_namespace(args_cli)
    validate_v2_params(params)

    output_dir, output_file_name = setup_output_directories()
    env_cfg = create_environment_config(output_dir, output_file_name)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    enable_async_episode_export(env)
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

    runtime = HeadFollowV2Runtime.from_params(params)
    should_reset = False
    teleoperation_active = not args_cli.xr
    running_recording_instance = not args_cli.xr
    need_standing_init = False
    step_count = 0
    first_active_logged = False

    def reset_env() -> None:
        nonlocal should_reset
        should_reset = True
        print("Reset requested — episode will be saved and environment reset on next step")

    def start_recording_instance() -> None:
        nonlocal running_recording_instance, teleoperation_active, first_active_logged, need_standing_init
        running_recording_instance = True
        teleoperation_active = True
        first_active_logged = False
        need_standing_init = True
        runtime.clear_session()
        sync_retargeter_idle_from_robot(teleop_interface, robot)
        print("Recording started (Standing/Pursuing v2 locomotion)")

    def stop_recording_instance() -> None:
        nonlocal running_recording_instance, teleoperation_active, need_standing_init
        running_recording_instance = False
        teleoperation_active = False
        need_standing_init = False
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
        "Legs v2: Standing = station-keep latch; Pursuing = filtered headset XY + headset yaw. "
        "Upper body: teleop in Standing, frozen in Pursuing. "
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
                loco_mode_dbg = runtime.mode

                if running_recording_instance:
                    step_count += 1
                    robot_pos, robot_yaw = robot_root_state(robot)
                    raw_head_pose = None
                    filtered_head_pose = None
                    err_x = err_y = err_yaw = dist_xy = 0.0
                    ref_dist_xy: float | None = None
                    ref_yaw_err: float | None = None
                    head_yaw: float | None = None
                    loco_mode_dbg = runtime.mode

                    if teleoperation_active:
                        raw = teleop_interface._get_raw_data()
                        raw_head_pose = raw.get(DeviceBase.TrackingTarget.HEAD)
                        if raw_head_pose is not None:
                            offset_rad = params.head_yaw_offset_rad
                            if need_standing_init:
                                filtered_head_pose = runtime.filter.reset(raw_head_pose)
                                runtime.begin_standing(
                                    robot,
                                    filtered_head_pose,
                                    offset_rad=offset_rad,
                                    latch_robot=True,
                                )
                                need_standing_init = False
                            else:
                                filtered_head_pose = runtime.filter.update(raw_head_pose)

                            if runtime.head_ref_xy_yaw is not None:
                                ref_dist_xy, ref_yaw_err = head_ref_delta(
                                    filtered_head_pose,
                                    runtime.head_ref_xy_yaw,
                                    offset_rad=offset_rad,
                                )
                                raw_ref_dist_xy, raw_ref_yaw_err = head_ref_delta(
                                    raw_head_pose,
                                    runtime.head_ref_xy_yaw,
                                    offset_rad=offset_rad,
                                )
                                prev_mode = runtime.mode
                                new_mode = update_loco_mode(
                                    runtime.mode,
                                    filtered_head_pose,
                                    runtime.head_ref_xy_yaw,
                                    pos_on=params.pursue_pos_on,
                                    pos_off=params.pursue_pos_off,
                                    yaw_on=params.pursue_yaw_on,
                                    yaw_off=params.pursue_yaw_off,
                                    offset_rad=offset_rad,
                                    entry_pose_7d=raw_head_pose,
                                    robot_xy=(robot_pos[0], robot_pos[1]),
                                    robot_yaw=robot_yaw,
                                    arrive_robot_head_xy=params.arrive_robot_head_xy,
                                    arrive_robot_head_yaw=params.arrive_robot_head_yaw,
                                )
                                if prev_mode == "standing" and new_mode == "pursuing":
                                    # region agent log
                                    _agent_debug_log(
                                        "F",
                                        "sberchel_record_demos_v2:mode_edge",
                                        "standing_to_pursuing",
                                        {
                                            "runId": "post-fix",
                                            "step": step_count,
                                            "latch_xy": (
                                                list(runtime.standing_latch_xy_yaw)
                                                if runtime.standing_latch_xy_yaw
                                                else None
                                            ),
                                            "robot_xy": [robot_pos[0], robot_pos[1]],
                                            "filt_ref_dist_xy": ref_dist_xy,
                                            "filt_ref_yaw_err": ref_yaw_err,
                                            "raw_ref_dist_xy": raw_ref_dist_xy,
                                            "raw_ref_yaw_err": raw_ref_yaw_err,
                                            "pos_on": params.pursue_pos_on,
                                            "yaw_on": params.pursue_yaw_on,
                                        },
                                    )
                                    # endregion
                                    runtime.frozen_snapshot = capture_frozen_upper(
                                        robot, runtime.last_teleop_upper
                                    )
                                elif prev_mode == "pursuing" and new_mode == "standing":
                                    latch = runtime.standing_latch_xy_yaw
                                    latch_dist = (
                                        math.hypot(
                                            robot_pos[0] - latch[0],
                                            robot_pos[1] - latch[1],
                                        )
                                        if latch is not None
                                        else None
                                    )
                                    # region agent log
                                    _agent_debug_log(
                                        "A",
                                        "sberchel_record_demos_v2:mode_edge",
                                        "pursuing_to_standing",
                                        {
                                            "step": step_count,
                                            "robot_xy": [robot_pos[0], robot_pos[1]],
                                            "robot_yaw": robot_yaw,
                                            "latch_xy_yaw": list(latch) if latch else None,
                                            "latch_dist": latch_dist,
                                            "ref_dist_xy": ref_dist_xy,
                                            "ref_yaw_err": ref_yaw_err,
                                        },
                                    )
                                    # endregion
                                    sync_retargeter_idle_from_robot(teleop_interface, robot)
                                    runtime.on_enter_standing_from_pursuing(
                                        robot,
                                        filtered_head_pose,
                                        offset_rad=offset_rad,
                                    )
                                    latch = runtime.standing_latch_xy_yaw
                                    latch_dist = (
                                        math.hypot(
                                            robot_pos[0] - latch[0],
                                            robot_pos[1] - latch[1],
                                        )
                                        if latch is not None
                                        else None
                                    )
                                    # region agent log
                                    _agent_debug_log(
                                        "A",
                                        "sberchel_record_demos_v2:mode_edge",
                                        "pursuing_to_standing_post_latch",
                                        {
                                            "runId": "post-fix",
                                            "step": step_count,
                                            "robot_xy": [robot_pos[0], robot_pos[1]],
                                            "latch_xy_yaw": list(latch) if latch else None,
                                            "latch_dist": latch_dist,
                                        },
                                    )
                                    # endregion
                                runtime.mode = new_mode
                                loco_mode_dbg = runtime.mode

                            if runtime.mode == "standing" and runtime.standing_latch_xy_yaw is not None:
                                vx, vy, wz = standing_leg_command(
                                    robot, runtime.standing_latch_xy_yaw, params
                                )
                                hold_x, hold_y, hold_yaw = runtime.standing_latch_xy_yaw
                                err_x = hold_x - robot_pos[0]
                                err_y = hold_y - robot_pos[1]
                                dist_xy = (err_x ** 2 + err_y ** 2) ** 0.5
                                err_yaw = wrap_angle(hold_yaw - robot_yaw)
                                if dist_xy > 0.15 and step_count % 15 == 0:
                                    # region agent log
                                    _agent_debug_log(
                                        "A",
                                        "sberchel_record_demos_v2:standing_cmd",
                                        "standing_large_latch_err",
                                        {
                                            "step": step_count,
                                            "latch_dist": dist_xy,
                                            "latch_xy": [hold_x, hold_y],
                                            "robot_xy": [robot_pos[0], robot_pos[1]],
                                            "vx": vx,
                                            "vy": vy,
                                            "wz": wz,
                                            "err_yaw": err_yaw,
                                        },
                                    )
                                    # endregion
                            elif runtime.mode == "pursuing" and filtered_head_pose is not None:
                                (
                                    vx,
                                    vy,
                                    wz,
                                    err_x,
                                    err_y,
                                    err_yaw,
                                    dist_xy,
                                    head_yaw,
                                ) = pursuing_command(
                                    robot,
                                    filtered_head_pose,
                                    lin_gain=params.follow_lin_gain,
                                    ang_gain=params.follow_ang_gain,
                                    max_lin=params.follow_max_lin,
                                    max_ang=params.follow_max_ang,
                                    pos_deadzone=params.follow_pos_deadzone,
                                    yaw_deadzone=params.follow_yaw_deadzone,
                                    head_yaw_offset_rad=offset_rad,
                                )

                            if runtime.mode == "standing":
                                upper_body = torch.cat(
                                    [r.retarget(raw) for r in teleop_interface._retargeters],
                                    dim=-1,
                                )
                                runtime.last_teleop_upper = (
                                    upper_body.squeeze(0).clone()
                                    if upper_body.ndim > 1
                                    else upper_body.clone()
                                )
                                upper_body_mode = "teleop"
                            else:
                                if runtime.frozen_snapshot is None:
                                    runtime.frozen_snapshot = capture_frozen_upper(
                                        robot, runtime.last_teleop_upper
                                    )
                                upper_body = build_upper_from_frozen_snapshot(
                                    robot, runtime.frozen_snapshot, env.device
                                )
                                upper_body_mode = "frozen"

                            if upper_body.ndim == 1:
                                upper_body = upper_body.unsqueeze(0)
                            upper_body = upper_body.to(env.device).repeat(env.num_envs, 1)
                        else:
                            upper_body = idle_upper.unsqueeze(0).repeat(env.num_envs, 1)
                            upper_body_mode = "idle"
                            vx = vy = wz = 0.0
                    else:
                        upper_body = idle_upper.unsqueeze(0).repeat(env.num_envs, 1)
                        upper_body_mode = "idle"

                    legs = policy_leg_targets(net, vx, vy, wz)
                    print(f'LEGS COMMAND: vx={vx}, vy={vy}, wz={wz}')
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
                        print_debug_block_v2(
                            step_count,
                            active=teleoperation_active,
                            robot_pos=robot_pos,
                            robot_yaw=robot_yaw,
                            raw_head_pose=raw_head_pose,
                            filtered_head_pose=filtered_head_pose,
                            head_ref_xy_yaw=runtime.head_ref_xy_yaw,
                            standing_latch_xy_yaw=runtime.standing_latch_xy_yaw,
                            loco_mode=loco_mode_dbg,
                            err_x=err_x,
                            err_y=err_y,
                            err_yaw=err_yaw,
                            dist_xy=dist_xy,
                            head_yaw=head_yaw,
                            ref_dist_xy=ref_dist_xy,
                            ref_yaw_err=ref_yaw_err,
                            vx=vx,
                            vy=vy,
                            wz=wz,
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
                    need_standing_init = False
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
