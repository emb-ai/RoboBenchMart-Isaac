import gymnasium as gym

gym.register(
    id="Dsynth-Fetch-Any-Item-Fetchbot-v0",
    entry_point="dsynth_isaac.tasks.manager_based.fetching.fetching_env:FetchItemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fetchbot_env_cfg:FetchbotAnyItemEnvCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Dsynth-Fetch-Any-Item-Fetchbot-IK-Rel-v0",
    entry_point="dsynth_isaac.tasks.manager_based.fetching.fetching_env:FetchItemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fetchbot_env_cfg:FetchbotIKRelAnyItemEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Dsynth-Fetch-Any-Item-Franka-v0",
    entry_point="dsynth_isaac.tasks.manager_based.fetching.fetching_env:FetchItemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_env_cfg:FrankaFetchAnyItemEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Dsynth-Fetch-Any-Item-Franka-IK-Rel-v0",
    entry_point="dsynth_isaac.tasks.manager_based.fetching.fetching_env:FetchItemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_env_cfg:FrankaIKRelFetchAnyItemEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Dsynth-Fetch-Any-Item-GR1T2-Abs-v0",
    entry_point="dsynth_isaac.tasks.manager_based.fetching.fetching_env:FetchItemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gr1t2_env_cfg:GR1T2FetchAnyItemEnvCfg",
    },
    disable_env_checker=True,
)

# gym.register(
#     id="Isaac-Dsynth-Fetch-IKRelMimic-v0",
#     entry_point="isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env:FrankaCubeStackIKRelMimicEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.fetch_base_env_cfg:FetchDsynthIKRelMimicEnvCfg",
#     },
#     disable_env_checker=True,
# )
gym.register(
    id="Isaac-Dsynth-Sberchel-Base-v0",
    entry_point="dsynth_isaac.tasks.manager_based.fetching.fetching_env:FetchItemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sberchel_base_env_cfg:SberchelBaseEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Dsynth-Fetch-Any-Item-Sberchel-Open-v0",
    entry_point="dsynth_isaac.tasks.manager_based.fetching.fetching_env:FetchItemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sberchel_base_env_cfg:SberchelOpenKinematicsFetchAnyItemEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Sberchel-Open-Kinematics-Demo-v0",
    entry_point="dsynth_isaac.tasks.manager_based.sberdemo_manager_env:SberDemoManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sberchel_base_env_cfg:SberchelOpenKinematicsDemoEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Dsynth-Fetch-Any-Item-Sberchel-Upperbody-XR-v0",
    entry_point="dsynth_isaac.tasks.manager_based.fetching.fetching_env:FetchItemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sberchel_upperbody_xr_env_cfg:UpperbodySberchelXRFetchAnyItemEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Dsynth-Fetch-Any-Item-Sberchel-XR-v0",
    entry_point="dsynth_isaac.tasks.manager_based.fetching.fetching_env:FetchItemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sberchel_xr_env_cfg:SberchelXRFetchAnyItemEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Demo-Sberchel-XR-v0",
    entry_point="dsynth_isaac.tasks.manager_based.sberdemo_manager_env:SberDemoManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sberchel_xr_env_cfg:SberchelXRDemoEnvCfg",
    },
    disable_env_checker=True,
)
