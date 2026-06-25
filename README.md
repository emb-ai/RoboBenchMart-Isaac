# RoboBenchMart-Isaac

This repository is an **Isaac Sim / Isaac Lab** port of [RoboBenchMart](https://emb-ai.github.io/RoboBenchMart/)—retail dark-store scenes and tasks originally built around the ManiSkill stack in [darkstore-synthesizer](https://github.com/emb-ai/RoboBenchMart) (see the upstream [README](https://github.com/emb-ai/RoboBenchMart/blob/main/README.md) for benchmark background, tasks, and citations). Here, scene synthesis produces USD-oriented layouts and configs that load inside Isaac Lab environments.

## Installation

1. **Install [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)** and **install [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)**.

   ```bash
   conda create -n env_isaaclab python=3.11
   conda activate env_isaaclab

   pip install isaaclab[isaacsim,all]==2.3.2.post1 --extra-index-url https://pypi.nvidia.com

   pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
   ```

2. **Install requirements**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download USD assets** used by the Isaac workflows (requires the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) / `hf`):

   ```bash
   hf download emb-ai/RoboBenchMart_assets_usd --repo-type dataset --local-dir assets_usd
   ```

## Generate a scene

Example using the `isaac_sample` Hydra preset (writes under `generated_envs/` by default for this config):

```bash
python scripts/generate_scene_continuous.py ds_continuous=isaac_sample
```

To change the output location, override `ds_continuous.output_dir` (see `conf/ds_continuous/` for other presets).

## Run in Isaac Sim

Launch the sample environment with random actions, pointing at the generated config directory:

```bash
python scripts/demo_random_actions.py --config_dir_path generated_envs/isaac_sample
```

Use Isaac Lab’s usual `AppLauncher` flags (for example `--headless` or device selection) as documented in Isaac Lab; `demo_random_actions.py` forwards those through `AppLauncher.add_app_launcher_args`.

## Run with Green robot with WASDQE keyboard control

```bash
python scripts/demo_sberchel_kb.py --config_dir_path generated_envs/isaac_sample
```

## Citation

If you use RoboBenchMart in research, cite the project as in the [main RoboBenchMart README](https://github.com/emb-ai/RoboBenchMart/blob/main/README.md#citation).
