import gymnasium as gym

gym.register(
    id="Isaac-Dsynth-Fetch-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fetch_base_env_cfg:FetchIKBaseEnvCfg",
    },
    disable_env_checker=True,
)
