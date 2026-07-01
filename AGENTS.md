# RoboBenchMart Agent Notes

## Project Purpose
- This repo implements RoboBenchMart tasks on IsaacLab for darkstore/warehouse manipulation evaluation.
- Scenes are generated from YAML in `conf/`, exported as JSON scene data (for example `generated_envs/isaac_sample`), and instantiated at runtime via `SceneBuilder`.

## Core Runtime Flow
- `DarkstoreManagerBasedRLEnv` creates `DarkstoreInteractiveScene`.
- `DarkstoreInteractiveScene` builds dynamic entities through `SceneBuilder`.
- `SceneBuilder` registers rigid objects and per-env grasp sensors into scene registries.
- Fetching tasks derive targets from scene product metadata and evaluate success via grasp contact + optional pull-out checks.

## Dynamic Entity Contracts
- Dynamic rigid-object keys are preserved as:
  - shelf: `[env_i]_[shelf_name]`
  - item: `[env_i]_[shelf_name]_[item_name]`
- Per-env grasp sensor keys are preserved as:
  - `contact_grasp_[env_i]`

## Env ID Semantics (Critical)
- Global environment IDs come from scene/env manager reset and event APIs.
- Subset rigid objects remap global env IDs to local instance indices for write/reset paths.
- Contact sensor update path uses local sensor indices from IsaacLab (`_update_buffers_impl`); do not remap there.

## SceneBuilder Invariants
- Heterogeneous per-env object sets are expected and supported.
- Use single registration path for dynamic rigid objects to avoid registry drift.
- Robot default pose is derived from chosen `start_shelf_name` once per env.
- Product metadata is read through public helper methods (avoid external reads of private internals).

## Fetching Task Notes
- Target selection regex is matched with `re.search` (partial matches supported).
- Termination maps contacts via sensor filter prim paths to item IDs; do not assume order from private arrays.
- If per-env contact sensor is absent, termination should fail-safe (no success) rather than crash.

## Common Commands
- Smoke test:
  - `/home/kvsoshin/miniconda3/envs/isaacsim/bin/python scripts/demo_random_actions.py --task Dsynth-Fetch-Any-Item-Fetchbot-IK-Rel-v0 --config_dir_path generated_envs/isaac_sample --num_envs 1`
- Syntax check:
  - `python -m compileall dsynth_isaac scripts`

## Editing Guidance
- Preserve backward-compatible dynamic key formats unless explicitly migrating all call sites.
- Prefer public scene/builder interfaces over direct private attribute access.
- Add docstrings when touching scene/env integration logic, especially around env-id mapping and tensor shape assumptions.
