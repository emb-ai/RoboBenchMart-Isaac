import gymnasium as gym

gym.register(
    id="Isaac-Dsynth-Fetch-Base-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fetch_base_env_cfg:FetchBaseEnvCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Dsynth-Fetch-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fetch_base_env_cfg:FetchIKBaseEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Dsynth-Panda-Base-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_base_env_cfg:PandaBaseEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Dsynth-Worker-Base-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.worker_base_env_cfg:WorkerBaseEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Dsynth-Panda-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_base_env_cfg:PandaIKBaseEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Dsynth-Panda-IKRelMimic-v0",
    entry_point="isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_base_env_cfg:PandaDsynthIKRelMimicEnvCfg",
    },
    disable_env_checker=True,
)
