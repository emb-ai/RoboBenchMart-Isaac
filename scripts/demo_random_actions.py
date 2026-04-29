import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument(
    "--config_dir_path", type=str, help="Path to the config directory."
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from isaaclab.envs import ManagerBasedEnv
from isaaclab_tasks.utils import parse_env_cfg

import gymnasium as gym
import sys
sys.path.append('.')
from dsynth_isaac.tasks.manager_based.fetch_base_env_cfg import FetchBaseEnvCfg
# from dsynth_isaac.tasks.manager_based.panda_base_env_cfg import PandaBaseEnvCfg
from dsynth_isaac.tasks.manager_based.worker_base_env_cfg import WorkerBaseEnvCfg
import dsynth_isaac.tasks.manager_based

def main():
    env_cfg = WorkerBaseEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed=42
    env_cfg.scene.dsynth_scene.spawn.config_dir_path = args_cli.config_dir_path
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    seed = 42
    env.reset(seed=seed)
    while simulation_app.is_running():
        with torch.inference_mode():
            # joint_efforts = torch.randn_like(env.action_manager.action)
            action = env.action_manager.action
            # action[:, -2] = 10.0
            # action[:, -1] = 10.0
            # action[0, 0] =  0.0
            # action[0, 1] =  -0.002
            # action[0, 2] =  0.0
            # action[0, 3] =  0.0
            # action[0, 4] =  0.0
            # action[0, 5] =  0.0
            # action[0, 6] = 0.0
# tensor([[-0.6635,  0.6894,  0.2776, -0.0860]], device='cuda:0')
            obs, _ = env.step(action)
        count += 1
        if count % 200 == 0:
            seed += 1
            env.reset(seed=seed)
            print(">>> reset")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()