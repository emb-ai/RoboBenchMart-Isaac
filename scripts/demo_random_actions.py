import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedEnv

import sys
sys.path.append('.')
from dsynth_isaac.tasks.manager_based.fetch_base_env_cfg import FetchBaseEnvCfg

def main():
    env_cfg = FetchBaseEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed=42
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    env.reset(seed=42)
    while simulation_app.is_running():
        with torch.inference_mode():
            # joint_efforts = torch.randn_like(env.action_manager.action)
            joint_efforts = env.action_manager.action
            joint_efforts[0, -2] = 10.0
            joint_efforts[0, -1] = 10.0
            obs, _ = env.step(joint_efforts)


    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()