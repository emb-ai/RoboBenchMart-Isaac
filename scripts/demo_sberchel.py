"""Demo: the Sberchel v3.0a humanoid standing in a generated darkstore,
balanced by the ONNX RL locomotion policy from green_challenge.

The robot's base is FREE — it does not fall because green_challenge's
``Network`` (controller "zest", driving ``body_latest.onnx``) computes leg
joint targets every step that keep it upright. This mirrors green_challenge's
own ``run.sh`` loop, just dropped into RoboBenchMart's darkstore env.

Usage:
    # stand in place (default)
    python scripts/demo_sberchel.py --config_dir_path generated_envs/isaac_sample --episode_len 1000000

    # walk (policy velocity command, m/s and rad/s)
    python scripts/demo_sberchel.py --config_dir_path generated_envs/isaac_sample --vx 0.5

    # no-policy sanity check: pin the base, send zero actions
    python scripts/demo_sberchel.py --config_dir_path generated_envs/isaac_sample --pin
"""

import os
import sys
import argparse
from isaaclab.app import AppLauncher

# Make the sibling green_challenge package importable (for policy.network.Network).
_GC = os.environ.get("GREEN_CHALLENGE_PATH", "/home/alex/projects/green_challenge")
_gc_parent = os.path.dirname(os.path.abspath(_GC))
if _gc_parent not in sys.path:
    sys.path.insert(0, _gc_parent)

parser = argparse.ArgumentParser(description="Sberchel humanoid balanced by the RL policy in a darkstore.")
parser.add_argument("--config_dir_path", type=str, help="Path to the config directory.")
parser.add_argument("--task", type=str, default="Isaac-Dsynth-Sberchel-Base-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed to use.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes.")
parser.add_argument("--episode_len", type=int, default=100, help="Length of each episode (-1 = run until the viewer is closed).")
parser.add_argument("--vx", type=float, default=0.0, help="Forward velocity command (m/s). Negative = backward.")
parser.add_argument("--vy", type=float, default=0.0, help="Sideways velocity command (m/s).")
parser.add_argument("--wz", type=float, default=0.0, help="Turning rate command (rad/s).")
parser.add_argument("--pin", action="store_true", default=False, help="No-policy fallback: pin the base and send zero actions.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from pathlib import Path
import time
import gymnasium as gym

from isaaclab_tasks.utils import parse_env_cfg

sys.path.append('.')
import dsynth_isaac.tasks.manager_based


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.scene.dsynth_scene.spawn.config_dir_path = args_cli.config_dir_path

    # --pin: no-policy fallback. Pin the base before the scene is built.
    if args_cli.pin:
        if env_cfg.scene.robot.spawn.articulation_props is not None:
            env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True

    if args_cli.video:
        env_cfg.viewer.origin_type = 'asset_root'
        env_cfg.viewer.asset_name = 'robot'
        env_cfg.viewer.eye = (2.5, 2.5, 1.5)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.4)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    if args_cli.video:
        new_traj_name = time.strftime("%Y%m%d_%H%M%S")
        video_folder = Path(args_cli.config_dir_path) / 'videos'
        video_folder.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            episode_trigger=lambda episode_id: True,
            video_length=args_cli.episode_len,
            disable_logger=True,
            name_prefix=new_traj_name,
        )

    total_dim = env.unwrapped.action_manager.total_action_dim
    print(f">>> action dim = {total_dim} (expected 41: legs12 + torso13 + wrist4 + fingers12)", flush=True)

    # Build the RL locomotion controller (unless running the pinned sanity check).
    net = None
    if not args_cli.pin:
        env.reset(seed=args_cli.seed)
        from green_challenge.policy.network import Network
        net = Network(env, type="zest", device="cpu")
        print(f">>> policy loaded: zest (cmd vx={args_cli.vx} vy={args_cli.vy} wz={args_cli.wz})", flush=True)

    seed = args_cli.seed
    for _ in range(args_cli.num_episodes):
        print(">>> reset", flush=True)
        env.reset(seed=seed)
        if net is not None:
            net.reset()
        t = 0
        while simulation_app.is_running() and (args_cli.episode_len < 0 or t < args_cli.episode_len):
            with torch.inference_mode():
                if net is None:
                    # pinned sanity check: hold zero action
                    act_tensor = torch.zeros(
                        (args_cli.num_envs, total_dim), dtype=torch.float32, device=env.unwrapped.device
                    )
                else:
                    # mirror green_challenge run.py: command -> obs -> policy -> legs
                    net.set_command_detailed(args_cli.vx, args_cli.vy, args_cli.wz)
                    net.set_joint_overrides({
                        "torso_joint_pos": [0.0] * 13,
                        "wrist_joint_pos": [0.0] * 4,
                        "finger_joint_pos": [0.0] * 12,
                    })
                    loco = net(net.obs())
                    legs = list(loco["joint_pos"][:12])
                    # legs(12, from policy) + torso(13) + wrist(4) + fingers(12), all held at rest
                    act_list = legs + [0.0] * 13 + [0.0] * 4 + [0.0] * 12
                    act_tensor = torch.as_tensor(
                        act_list, dtype=torch.float32, device=env.unwrapped.device
                    ).unsqueeze(0).repeat(args_cli.num_envs, 1)

                _ = env.step(act_tensor)
            t += 1
        seed += 1
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
