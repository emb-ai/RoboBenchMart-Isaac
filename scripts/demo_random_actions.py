from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument(
    "--config_dir_path", type=str, required=True, help="Path to generated scene config directory."
)
parser.add_argument("--task", type=str, default='Dsynth-Fetch-Any-Item-Fetchbot-IK-Rel-v0', help="Registered gym task ID.")
parser.add_argument("--seed", type=int, default=42, help="Seed to use.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes.")
parser.add_argument("--episode_len", type=int, default=100, help="Length of each episode.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Import Pinocchio before Isaac Sim (required for Pink IK / upper-body XR tasks).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


def _task_needs_pinocchio(task: str) -> bool:
    """Registered tasks that use Isaac Lab Pink IK (not generic IK-rel envs)."""
    task_lower = task.lower()
    return any(hint in task_lower for hint in ("upperbody", "gr1t2", "sberchel-xr"))


if args_cli.enable_pinocchio or _task_needs_pinocchio(args_cli.task):
    # Must load before AppLauncher so eigenpy registers std::vector bindings for Pinocchio.
    import pinocchio  # noqa: F401

args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from pathlib import Path
import time
import tqdm
import gymnasium as gym

from isaaclab.envs import ManagerBasedEnv
from isaaclab_tasks.utils import parse_env_cfg

import sys
sys.path.append('.')
import dsynth_isaac.tasks.manager_based
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_env_cfg import FrankaCubeStackEnvCfg


def _pink_hold_action_from_robot(env) -> torch.Tensor | None:
    """Build a Pink IK hold action from the robot's current wrist poses (world frame).

    Env ``idle_action`` wrist targets are calibrated for the XR anchor, not the
    darkstore robot spawn pose, so they pull arms the wrong way in demo scripts.
    """
    unwrapped = env.unwrapped
    action_manager = getattr(unwrapped, "action_manager", None)
    if action_manager is None:
        return None
    try:
        ik_term = action_manager.get_term("upper_body_ik")
    except (AttributeError, KeyError, ValueError):
        return None
    cfg = ik_term.cfg
    robot = unwrapped.scene[cfg.asset_name]
    data = robot.data

    left_link = cfg.target_eef_link_names["left_wrist"]
    right_link = cfg.target_eef_link_names["right_wrist"]
    left_idx = data.body_names.index(left_link)
    right_idx = data.body_names.index(right_link)

    left_wrist = data.body_link_state_w[:, left_idx, :7]
    right_wrist = data.body_link_state_w[:, right_idx, :7]
    hand_pos = data.joint_pos[:, ik_term._hand_joint_ids]
    return torch.cat([left_wrist, right_wrist, hand_pos], dim=-1)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed=args_cli.seed
    env_cfg.scene.scene_config_dir_path = args_cli.config_dir_path

    if args_cli.video:
        env_cfg.viewer.origin_type = 'asset_root'
        env_cfg.viewer.asset_name = 'robot'
        env_cfg.viewer.eye = (1.0, 1.0, 2.0)
        env_cfg.viewer.lookat = (0.4, 0.0, 1.6)
    env_cfg.target_item_name_regex = ".*Nestle Honey.*"
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    
    if args_cli.video:
        new_traj_name = time.strftime("%Y%m%d_%H%M%S")
        video_folder = Path(args_cli.config_dir_path) / 'videos'
        video_folder.mkdir(parents=True, exist_ok=True)
        video_kwargs = {
                "video_folder": str(video_folder),
                "episode_trigger": lambda episode_id: True,
                "video_length": args_cli.episode_len,
                "disable_logger": True,
                'name_prefix': f"{new_traj_name}",
            }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    seed = args_cli.seed
    for n_episode in range(args_cli.num_episodes):
        with torch.inference_mode():
            print(">>> reset")
            env.reset(seed=seed)
            hold_action = _pink_hold_action_from_robot(env)
            for t in tqdm.tqdm(range(args_cli.episode_len), total=args_cli.episode_len, desc="Elapsed steps"):
                if hold_action is not None:
                    action = hold_action
                else:
                    action = env.unwrapped.action_manager.action.clone()
                    if "fetchbot" in args_cli.task.lower():
                        action[:, 2] = 2.0
                obs, reward, terminated, truncated, info = env.step(action)
                print(info)
        seed += 1
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()