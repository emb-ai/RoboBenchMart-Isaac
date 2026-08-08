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


## Environments

Registered gym environments from `dsynth_isaac/tasks/manager_based/__init__.py`:

| Env name | Config class | Success criteria | Embodiment | Action space | Scene |
| --- | --- | --- | --- | --- | --- |
| `Dsynth-Fetch-Any-Item-Fetchbot-v0` | `FetchbotAnyItemEnvCfg` | Fetch target item (grasp + pull-out) | `FETCHBOT_CONFIG` | 13D: `arm_actions`(7), `gripper_actions`(1), `body_actions`(1), `head_actions`(2), `base_actions`(2) | `DarkstoreSceneCfg` |
| `Dsynth-Fetch-Any-Item-Fetchbot-IK-Rel-v0` | `FetchbotIKRelAnyItemEnvCfg` | Fetch target item (grasp + pull-out) | `FETCHBOT_CONFIG` | Same groups; arm is relative Diff-IK pose (6) | `DarkstoreSceneCfg` |
| `Dsynth-Fetch-Any-Item-Franka-v0` | `FrankaFetchAnyItemEnvCfg` | Fetch target item (grasp + pull-out) | `FRANKA_PANDA_CFG` | 8D: `arm_action`(7), `gripper_action`(1) | `DarkstoreSceneCfg` |
| `Dsynth-Fetch-Any-Item-Franka-IK-Rel-v0` | `FrankaIKRelFetchAnyItemEnvCfg` | Fetch target item (grasp + pull-out) | `FRANKA_PANDA_HIGH_PD_CFG` | 7D: Diff-IK arm (6) + gripper (1) | `DarkstoreSceneCfg` |
| `Dsynth-Fetch-Any-Item-GR1T2-Abs-v0` | `GR1T2FetchAnyItemEnvCfg` | none | `GR1T2_HIGH_PD_CFG` | 36D: `upper_body_ik` (wrists 7+7 + hands 22) | `DarkstoreSceneCfg` (`GR1T2SceneCfg`) |
| `Isaac-Dsynth-Sberchel-Base-v0` | `SberchelBaseEnvCfg` | none | `SBERCHEL_CONFIG` | 41D: `legs_joint_pos`(12), `torso_joint_pos`(13), `wrist_joint_pos`(4), `finger_joint_pos`(12) | `DarkstoreSceneCfg` |
| `Dsynth-Fetch-Any-Item-Sberchel-Open-v0` | `SberchelOpenKinematicsFetchAnyItemEnvCfg` | none | `SBERCHEL_CONFIG_OPEN_KINEMATICS` | 41D: same groups (serial wrist pitch) | `DarkstoreSceneCfg` |
| `Isaac-Sberchel-Open-Kinematics-Demo-v0` | `SberchelOpenKinematicsDemoEnvCfg` | none | `SBERCHEL_CONFIG_OPEN_KINEMATICS` | 41D (same) | `SberDemoSceneCfg` |
| `Dsynth-Fetch-Any-Item-Sberchel-Upperbody-XR-v0` | `UpperbodySberchelXRFetchAnyItemEnvCfg` | none | `SBERCHEL_UPPERBODY_XR_ROBOT` | 26D: `upper_body_ik` (wrists 7+7 + fingers 12) | `DarkstoreSceneCfg` |
| `Dsynth-Fetch-Any-Item-Sberchel-XR-v0` | `SberchelXRFetchAnyItemEnvCfg` | none | `SBERCHEL_XR_LOCOMOTION_NECK_ROBOT` | 40D: `legs_joint_pos`(12), `neck_joint_pos`(2), `upper_body_ik`(26) | `DarkstoreSceneCfg` |
| `Isaac-Demo-Sberchel-XR-v0` | `SberchelXRDemoEnvCfg` | none | `SBERCHEL_XR_LOCOMOTION_NECK_ROBOT` | 40D (same) | `SberDemoSceneCfg` |

Only envs that inherit `FetchAnyItemEnvCfg` (Fetchbot / Franka) wire grasp+pull-out success. Sberchel and GR1T2 “FetchAnyItem” registrations currently use base timeout terminations only; demo envs have no task success term.

## Robots

Robot articulation configs used by the environments:

- **`FETCHBOT_CONFIG`** (`dsynth_isaac/robots/fetch_cfg.py`): Fetch mobile manipulator from local USD `assets/robots/fetch/fetch.usd` (not Omniverse Nucleus stock).
- **`WORKER_CONFIG`** (`dsynth_isaac/robots/worker_cfg.py`): dual-arm worker from local USD `assets/robots/worker/worker.usd`. Env cfg exists; not registered in `__init__.py`.
- **`FRANKA_PANDA_CFG` / `FRANKA_PANDA_HIGH_PD_CFG`**: Franka Panda from `isaaclab_assets` (Isaac Lab / Omniverse stock).
- **`GR1T2_HIGH_PD_CFG`**: Fourier GR1T2 from `isaaclab_assets.robots.fourier` (Isaac Lab stock).
- **Sberchel** (`dsynth_isaac/robots/sberchel_cfg.py`): ConfigManager JSON + meshes from sibling **`green_challenge`** (`GREEN_CHALLENGE_PATH` or `../green_challenge`); hybrid MJCF is local under `dsynth_isaac/robots/mjcf/`:
  - `SBERCHEL_CONFIG` — closed kinematics (`robot_v3_0a`)
  - `SBERCHEL_CONFIG_OPEN_KINEMATICS` — hybrid open MJCF (serial upper body + policy crank legs)
  - `SBERCHEL_UPPERBODY_XR_ROBOT` — fixed pelvis, XR upper-body PD
  - `SBERCHEL_XR_LOCOMOTION_ROBOT` / `SBERCHEL_XR_LOCOMOTION_NECK_ROBOT` — free-floating locomotion + XR arm PD (neck PD on the `_NECK_` variant)

## Sberchel keyboard control

Drive open-kinematics Sberchel with WASD/QE via the zest locomotion policy (hold-to-move). Requires the Isaac Sim GUI (do not use `--headless`); click the Isaac window so it receives keyboard focus.

```bash
python scripts/demo_sberchel_kb.py \
  --task Dsynth-Fetch-Any-Item-Sberchel-Open-v0 \
  --config_dir_path generated_envs/isaac_sample
```

Keys: `W`/`S` forward/back, `A`/`D` turn, `Q`/`E` strafe; release keys to stand.

## Sberchel teleoperation

Deeper notes live under [`docs/`](docs/) (for example `sberchel_xr_teleop.md`, upperbody / hybrid reports). Below are the usual launch lines.

### CloudXR / OpenXR runtime (Option 2)

Follow [Isaac Lab CloudXR teleoperation — Option 2](https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html#run-isaac-lab-with-the-cloudxr-runtime): run Isaac Lab as a local process and CloudXR Runtime in Docker with a shared `openxr/` directory.

```bash
mkdir -p $(pwd)/openxr

docker run -it --rm --name cloudxr-runtime \
  --user $(id -u):$(id -g) \
  --gpus=all \
  -e "ACCEPT_EULA=Y" \
  --mount type=bind,src=$(pwd)/openxr,dst=/openxr \
  -p 48010:48010 \
  -p 47998:47998/udp \
  -p 47999:47999/udp \
  -p 48000:48000/udp \
  -p 48005:48005/udp \
  -p 48008:48008/udp \
  -p 48012:48012/udp \
  nvcr.io/nvidia/cloudxr-runtime:5.0.1
```

In the Isaac Lab terminal (same shared directory):

```bash
export XDG_RUNTIME_DIR=$(pwd)/openxr/run
export XR_RUNTIME_JSON=$(pwd)/openxr/share/openxr/1/openxr_cloudxr.json
```

In Isaac Sim’s AR panel, set Output Plugin to **OpenXR** and OpenXR Runtime to **System OpenXR Runtime**.

### Upper-body XR teleop

With CloudXR env vars set, from this repo (using Isaac Lab’s `isaaclab.sh`):

```bash
./isaaclab.sh -p scripts/teleop_se3_agent.py \
  --task Dsynth-Fetch-Any-Item-Sberchel-Upperbody-XR-v0 \
  --teleop_device handtracking \
  --config_dir_path generated_envs/isaac_sample \
  --enable_pinocchio
```

Press **START** in the XR client after the sim is ready.

### Demo collection (locomotion + teleop)

```bash
python scripts/sberchel_kb_record_demos.py \
  --task Isaac-Demo-Sberchel-XR-v0 \
  --teleop_device handtracking \
  --enable_pinocchio \
  --dataset_file ./datasets/sberchel_kb_demos.hdf5
```

### Replay trajectories

Pass the same `--dataset_file` used for recording (script defaults for record vs replay differ):

```bash
python scripts/sberchel_replay_demos.py \
  --task Isaac-Demo-Sberchel-XR-v0 \
  --dataset_file ./datasets/sberchel_kb_demos.hdf5 \
  --enable_pinocchio \
  --replay_source policy_loco
```

## Citation

If you use RoboBenchMart in research, cite the project as in the [main RoboBenchMart README](https://github.com/emb-ai/RoboBenchMart/blob/main/README.md#citation).
