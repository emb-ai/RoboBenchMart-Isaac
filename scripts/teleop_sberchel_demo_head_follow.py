"""VR teleop for Sberchel: upper body via OpenXR, legs follow headset pose.

The ONNX RL policy from green_challenge balances the robot while a P-controller
commands body-frame velocity toward the OpenXR headset world XY position and yaw;
XR hand tracking drives the arms and fingers through Pink IK.

Head-follow locomotion activates only after pressing START in XR (same gate as
upper-body teleop).

Usage:
    python scripts/teleop_sberchel_demo_head_follow.py \\
        --task Isaac-Demo-Sberchel-XR-v0 \\
        --teleop_device handtracking \\
        --enable_pinocchio
"""

import argparse
import logging
import os
import sys
from collections.abc import Callable

from isaaclab.app import AppLauncher

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from dsynth_isaac.green_challenge_path import ensure_green_challenge_importable  # noqa: E402
from dsynth_isaac.sim_defaults import apply_sberchel_cli_defaults  # noqa: E402

ensure_green_challenge_importable()

parser = argparse.ArgumentParser(description="Sberchel XR teleop with headset-follow locomotion.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Demo-Sberchel-XR-v0", help="Gym task id.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="handtracking",
    help="Teleop device key from env teleop_devices (e.g. handtracking, manusvive).",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio (required for Pink IK / hand tracking).",
)

from dsynth_isaac.teleop.sberchel_head_follow import add_head_follow_cli_args  # noqa: E402

add_head_follow_cli_args(parser)
AppLauncher.add_app_launcher_args(parser)
# Sberchel runs on green_challenge's CPU physics by default; --device overrides.
apply_sberchel_cli_defaults(parser)
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

from isaaclab.devices.device_base import DeviceBase
from dsynth_isaac.robots.sberchel_camera_cfg import strip_camera_configs
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import parse_env_cfg

sys.path.append(".")
import dsynth_isaac.tasks.manager_based  # noqa: F401

from dsynth_isaac.teleop.sberchel_head_follow import (
    FrozenUpperSnapshot,
    HeadFollowParams,
    UpperBodyMode,
    PursuitMode,
    YawMode,
    build_idle_upper_from_robot,
    build_upper_from_frozen_snapshot,
    capture_frozen_upper,
    capture_hold_pose_from_headset,
    create_sberchel_xr_teleop_interface,
    headset_follow_command,
    print_debug_block,
    robot_root_state,
    station_keep_command,
    sync_retargeter_idle_from_robot,
    update_loco_arrived,
    wrap_angle,
)
from dsynth_isaac.robots.sberchel_cfg import apply_sberchel_xr_teleop_pd_gains
from dsynth_isaac.tasks.manager_based.sberchel_base_env_cfg import (
    LEGS_ACTION_DIM,
    policy_leg_targets,
)
from green_challenge.policy.network import Network

logger = logging.getLogger(__name__)


def main() -> None:
    head_follow = HeadFollowParams.from_namespace(args_cli)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(f"Expected ManagerBasedRLEnvCfg, got {type(env_cfg).__name__}")

    env_cfg.seed = args_cli.seed
    env_cfg.terminations.time_out = None

    if args_cli.xr:
        env_cfg = strip_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
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

    should_reset = False
    teleoperation_active = not args_cli.xr
    step_count = 0
    first_active_logged = False
    yaw_mode: YawMode = "bearing"
    bearing_yaw_active = False
    pursuit_mode: PursuitMode = "long"
    loco_arrived = False
    prev_loco_arrived = False
    hold_xy_yaw: tuple[float, float, float] | None = None
    frozen_snapshot: FrozenUpperSnapshot | None = None
    last_teleop_upper: torch.Tensor | None = None

    def reset_env() -> None:
        nonlocal should_reset
        should_reset = True
        print("Reset triggered - environment will reset on next step")

    def start_teleoperation() -> None:
        nonlocal teleoperation_active, first_active_logged, yaw_mode, bearing_yaw_active, pursuit_mode
        nonlocal loco_arrived, prev_loco_arrived, hold_xy_yaw, frozen_snapshot, last_teleop_upper
        teleoperation_active = True
        first_active_logged = False
        yaw_mode = "bearing"
        bearing_yaw_active = False
        pursuit_mode = "long"
        loco_arrived = False
        prev_loco_arrived = False
        hold_xy_yaw = None
        frozen_snapshot = None
        last_teleop_upper = None
        sync_retargeter_idle_from_robot(teleop_interface, robot)
        print("Teleoperation activated (upper body + head-follow locomotion)")

    def stop_teleoperation() -> None:
        nonlocal teleoperation_active, yaw_mode, bearing_yaw_active, pursuit_mode
        nonlocal loco_arrived, prev_loco_arrived, hold_xy_yaw, frozen_snapshot, last_teleop_upper
        teleoperation_active = False
        yaw_mode = "bearing"
        bearing_yaw_active = False
        pursuit_mode = "long"
        loco_arrived = False
        prev_loco_arrived = False
        hold_xy_yaw = None
        frozen_snapshot = None
        last_teleop_upper = None
        print("Teleoperation deactivated")

    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_env,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_env,
    }

    teleop_interface = create_sberchel_xr_teleop_interface(
        env_cfg, args_cli.teleop_device, teleoperation_callbacks
    )
    teleop_interface.reset()
    sync_retargeter_idle_from_robot(teleop_interface, robot)
    idle_upper = build_idle_upper_from_robot(robot, env.device)

    print(f"Using teleop device: {teleop_interface}")
    print(
        "Legs: locomotion policy follows headset after START. "
        "Upper body: XR teleop when robot matches headset XY+yaw; "
        "arms/fingers frozen body-relative while pursuing."
    )
    if args_cli.xr:
        print("Press START in XR to begin teleoperation and head-follow. Press R to reset.")

    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                step_count += 1
                robot_pos, robot_yaw = robot_root_state(robot)
                head_pose = None
                err_x = err_y = err_yaw = dist_xy = 0.0
                vx = vy = wz = 0.0
                bearing_yaw: float | None = None
                head_yaw: float | None = None
                target_yaw: float | None = None
                yaw_mode_dbg: YawMode | None = None
                yaw_active: bool | None = None
                upper_body_mode: UpperBodyMode | None = None
                loco_arrived_dbg: bool | None = None
                pursuit_mode_dbg: PursuitMode | None = None

                if teleoperation_active:
                    raw = teleop_interface._get_raw_data()
                    head_pose = raw.get(DeviceBase.TrackingTarget.HEAD)
                    if head_pose is not None:
                        (
                            vx,
                            vy,
                            wz,
                            err_x,
                            err_y,
                            err_yaw,
                            dist_xy,
                            head_yaw,
                            bearing_yaw,
                            yaw_mode,
                            target_yaw,
                            yaw_active,
                            bearing_yaw_active,
                            pursuit_mode,
                        ) = headset_follow_command(
                            robot,
                            head_pose,
                            pursuit_mode=pursuit_mode,
                            yaw_mode=yaw_mode,
                            bearing_yaw_active=bearing_yaw_active,
                            yaw_target_mode=head_follow.follow_yaw_target,
                            yaw_on_gate=head_follow.follow_yaw_on_gate,
                            yaw_off_gate=head_follow.follow_yaw_off_gate,
                            short_on_gate=head_follow.follow_short_on_gate,
                            short_off_gate=head_follow.follow_short_off_gate,
                            short_max_lin=head_follow.follow_short_max_lin,
                            lin_gain=head_follow.follow_lin_gain,
                            ang_gain=head_follow.follow_ang_gain,
                            max_lin=head_follow.follow_max_lin,
                            max_ang=head_follow.follow_max_ang,
                            pos_deadzone=head_follow.follow_pos_deadzone,
                            yaw_deadzone=head_follow.follow_yaw_deadzone,
                            head_yaw_offset_rad=head_follow.head_yaw_offset_rad,
                        )
                        yaw_mode_dbg = yaw_mode
                        pursuit_mode_dbg = pursuit_mode
                        alignment_yaw_err = wrap_angle(head_yaw - robot_yaw)
                        new_loco_arrived = update_loco_arrived(
                            dist_xy,
                            alignment_yaw_err,
                            loco_arrived,
                            dist_arrive=head_follow.arrive_xy_tol,
                            dist_leave=head_follow.arrive_xy_leave,
                            yaw_arrive=head_follow.arrive_yaw_tol,
                            yaw_leave=head_follow.arrive_yaw_leave,
                        )
                        if prev_loco_arrived and not new_loco_arrived:
                            frozen_snapshot = capture_frozen_upper(robot, last_teleop_upper)
                        elif not prev_loco_arrived and new_loco_arrived:
                            hold_xy_yaw = capture_hold_pose_from_headset(
                                head_pose,
                                offset_rad=head_follow.head_yaw_offset_rad,
                            )
                            sync_retargeter_idle_from_robot(teleop_interface, robot)
                            frozen_snapshot = None
                        loco_arrived = new_loco_arrived
                        prev_loco_arrived = loco_arrived
                        loco_arrived_dbg = loco_arrived

                        if loco_arrived and hold_xy_yaw is not None:
                            vx, vy, wz = station_keep_command(
                                robot,
                                hold_xy_yaw,
                                lin_gain=head_follow.follow_lin_gain,
                                ang_gain=head_follow.follow_ang_gain,
                                max_lin=head_follow.follow_max_lin,
                                max_ang=head_follow.follow_max_ang,
                                pos_deadzone=0.0,
                                yaw_deadzone=0.0,
                            )

                    use_teleop = head_follow.teleop_while_pursuing or loco_arrived
                    if use_teleop:
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
                    else:
                        if frozen_snapshot is None:
                            frozen_snapshot = capture_frozen_upper(robot, last_teleop_upper)
                        upper_body = build_upper_from_frozen_snapshot(
                            robot, frozen_snapshot, env.device
                        )
                        upper_body_mode = "frozen"
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
                env.step(actions)

                should_print = False
                if head_follow.print_every_n > 0:
                    if teleoperation_active and not first_active_logged:
                        should_print = True
                        first_active_logged = True
                    elif step_count % head_follow.print_every_n == 0:
                        should_print = True
                if should_print:
                    print_debug_block(
                        step_count,
                        active=teleoperation_active,
                        robot_pos=robot_pos,
                        robot_yaw=robot_yaw,
                        head_pose=head_pose,
                        err_x=err_x,
                        err_y=err_y,
                        err_yaw=err_yaw,
                        dist_xy=dist_xy,
                        vx=vx,
                        vy=vy,
                        wz=wz,
                        bearing_yaw=bearing_yaw,
                        head_yaw=head_yaw,
                        target_yaw=target_yaw,
                        yaw_mode=yaw_mode_dbg,
                        yaw_active=yaw_active,
                        upper_body_mode=upper_body_mode,
                        loco_arrived=loco_arrived_dbg,
                        pursuit_mode=pursuit_mode_dbg,
                    )

                if should_reset:
                    env.reset()
                    net.reset()
                    apply_sberchel_xr_teleop_pd_gains(robot)
                    teleop_interface.reset()
                    yaw_mode = "bearing"
                    bearing_yaw_active = False
                    pursuit_mode = "long"
                    loco_arrived = False
                    prev_loco_arrived = False
                    hold_xy_yaw = None
                    frozen_snapshot = None
                    last_teleop_upper = None
                    sync_retargeter_idle_from_robot(teleop_interface, robot)
                    idle_upper = build_idle_upper_from_robot(robot, env.device)
                    should_reset = False
                    first_active_logged = False
                    print("Environment reset complete")
        except Exception as exc:
            logger.error("Error during simulation step: %s", exc)
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
