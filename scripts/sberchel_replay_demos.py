# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Replay Sberchel XR demonstrations recorded by ``sberchel_record_demos.py``."""

import argparse
import os
import sys

_DEFAULT_GC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "green_challenge"))
_GC = os.environ.get("GREEN_CHALLENGE_PATH", _DEFAULT_GC)
_gc_parent = os.path.dirname(os.path.abspath(_GC))
if _gc_parent not in sys.path:
    sys.path.insert(0, _gc_parent)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay Sberchel XR demonstrations in Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Demo-Sberchel-XR-v0",
    help="Task used for replay (should match the recording task).",
)
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="Episode indices to replay. Empty replays all episodes in the dataset.",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="datasets/sberchel_xr_demos.hdf5",
    help="Dataset file to replay.",
)
parser.add_argument(
    "--validate_states",
    action="store_true",
    default=False,
    help="Validate recorded states against runtime (single env only).",
)
parser.add_argument(
    "--validate_success_rate",
    action="store_true",
    default=False,
    help="Validate replay success rate using task termination criteria.",
)
parser.add_argument(
    "--validate_report",
    type=str,
    default="/tmp/sberchel_replay_validate_report.txt",
    help="When --validate_states is set, write validation output to this file instead of stdout.",
)
parser.add_argument(
    "--replay_source",
    type=str,
    choices=("policy_loco", "processed_joint_targets", "actions"),
    default="policy_loco",
    help=(
        "Replay command source. 'policy_loco' (default) runs the zest leg policy from recorded "
        "loco_commands and applies recorded upper-body joint targets. "
        "'processed_joint_targets' applies recorded post-IK joint targets directly. "
        "'actions' feeds raw 38D Pink+leg actions through the env (re-solves IK)."
    ),
)
parser.add_argument(
    "--no_state_sync",
    action="store_true",
    default=False,
    help=(
        "Disable per-step scene state sync from recorded post-step states. By default, "
        "processed_joint_targets replay re-applies the previous recorded scene state "
        "before each step to prevent open-loop drift on the floating humanoid."
    ),
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio (required for Pink IK replay).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import contextlib

import gymnasium as gym
import torch

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import dsynth_isaac.tasks.manager_based  # noqa: F401

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from dsynth_isaac.tasks.manager_based.sberchel_base_env_cfg import (
    LEGS_ACTION_DIM,
    POLICY_LEG_JOINT_NAMES,
    policy_leg_targets,
)
from dsynth_isaac.robots.sberchel_cfg import apply_sberchel_xr_teleop_pd_gains
from green_challenge.policy.network import Network

is_paused = False


def play_cb() -> None:
    global is_paused
    is_paused = False


def pause_cb() -> None:
    global is_paused
    is_paused = True


def _flatten_state_tensor(value: torch.Tensor) -> torch.Tensor:
    """Normalize scene-state tensors for element-wise comparison."""
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.reshape(-1).detach().cpu()


def compare_states(state_from_dataset, runtime_state, runtime_env_index) -> tuple[bool, str, float]:
    states_matched = True
    output_log = ""
    max_abs_error = 0.0
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = _flatten_state_tensor(
                    runtime_state[asset_type][asset_name][state_name][runtime_env_index]
                )
                dataset_asset_state = _flatten_state_tensor(
                    state_from_dataset[asset_type][asset_name][state_name]
                )
                if dataset_asset_state.numel() != runtime_asset_state.numel():
                    raise ValueError(
                        f"State shape of {state_name} for asset {asset_name} don't match: "
                        f"dataset {tuple(dataset_asset_state.shape)} vs runtime {tuple(runtime_asset_state.shape)}"
                    )
                for i in range(dataset_asset_state.numel()):
                    abs_error = abs(dataset_asset_state[i] - runtime_asset_state[i]).item()
                    max_abs_error = max(max_abs_error, abs_error)
                    if abs_error > 0.01:
                        states_matched = False
                        output_log += f'\tState ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] don\'t match\r\n'
                        output_log += f"\t  Dataset:\t{dataset_asset_state[i]}\r\n"
                        output_log += f"\t  Runtime: \t{runtime_asset_state[i]}\r\n"
    return states_matched, output_log, max_abs_error


def build_replay_idle_action(env, env_cfg, num_envs: int) -> torch.Tensor:
    """Build a full action-dim idle tensor for Sberchel XR replay.

    Recorded demos use ``[legs(12), upper_body_ik(26)]``. ``env_cfg.idle_action``
    may store only the 26D upper-body Pink targets; leg targets are taken from
    the robot's current joint positions after reset.
    """
    action_dim = env.action_manager.total_action_dim
    idle = torch.zeros(num_envs, action_dim, device=env.device)

    if hasattr(env_cfg, "idle_action"):
        cfg_idle = env_cfg.idle_action.to(env.device).reshape(-1)
        if cfg_idle.numel() == action_dim:
            idle[:] = cfg_idle.unsqueeze(0).repeat(num_envs, 1)
            return idle
        if cfg_idle.numel() == action_dim - LEGS_ACTION_DIM:
            idle[:, LEGS_ACTION_DIM:] = cfg_idle.unsqueeze(0).repeat(num_envs, 1)

    robot = env.scene["robot"]
    name_to_id = {name: idx for idx, name in enumerate(robot.joint_names)}
    for env_id in range(num_envs):
        for leg_idx, joint_name in enumerate(POLICY_LEG_JOINT_NAMES):
            joint_id = name_to_id.get(joint_name)
            if joint_id is not None:
                idle[env_id, leg_idx] = robot.data.joint_pos[env_id, joint_id]
    return idle


def prepare_sberchel_xr_episode_replay(env, env_cfg, num_envs: int) -> torch.Tensor:
    """Re-apply teleop PD gains and rebuild idle actions after ``reset`` / ``reset_to``."""
    apply_sberchel_xr_teleop_pd_gains(env.scene["robot"])
    return build_replay_idle_action(env, env_cfg, num_envs)


def get_next_replay_command(episode_data: EpisodeData, replay_source: str) -> torch.Tensor | None:
    """Return the next replay command tensor, advancing the episode cursor."""
    if replay_source == "processed_joint_targets":
        processed = episode_data.data.get("processed_actions")
        if processed is None:
            if episode_data.is_empty():
                return None
            raise ValueError(
                "Dataset episode is missing 'processed_actions' required for processed_joint_targets replay."
            )
        index = episode_data.next_action_index
        if index >= len(processed):
            return None
        episode_data.next_action_index = index + 1
        return processed[index]
    if replay_source == "policy_loco":
        return None
    return episode_data.get_next_action()


def get_next_policy_loco_step(episode_data: EpisodeData) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return recorded ``(loco_commands, processed_actions)`` for the next step."""
    if episode_data.is_empty():
        return None
    loco_commands = episode_data.data.get("loco_commands")
    processed_actions = episode_data.data.get("processed_actions")
    if loco_commands is None:
        raise ValueError(
            "Dataset episode is missing 'loco_commands' required for policy_loco replay. "
            "Re-record demos with the updated sberchel_record_demos.py."
        )
    if processed_actions is None:
        raise ValueError(
            "Dataset episode is missing 'processed_actions' required for policy_loco replay."
        )
    index = episode_data.next_action_index
    if index >= len(loco_commands):
        return None
    episode_data.next_action_index = index + 1
    return loco_commands[index], processed_actions[index]


def step_hybrid_joint_targets(env, legs: torch.Tensor, upper: torch.Tensor) -> None:
    """Step physics with separate leg and upper-body joint targets, bypassing Pink IK re-solve."""
    robot = env.scene["robot"]
    legs_term = env.action_manager.get_term("legs_joint_pos")
    upper_term = env.action_manager.get_term("upper_body_ik")
    legs_term._processed_actions[:] = legs
    upper_term._processed_actions[:] = upper

    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        legs_term.apply_actions()
        robot.set_joint_position_target(upper, joint_ids=upper_term._controlled_joint_ids)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
            env.sim.render()
        env.scene.update(dt=env.physics_dt)

    env.episode_length_buf += 1
    env.common_step_counter += 1


def step_processed_joint_targets(env, processed_actions: torch.Tensor) -> None:
    """Step physics using recorded joint targets, bypassing Pink IK re-solve."""
    step_hybrid_joint_targets(
        env,
        processed_actions[:, :LEGS_ACTION_DIM],
        processed_actions[:, LEGS_ACTION_DIM:],
    )


def sync_scene_to_recorded_state(env, recorded_state: dict, env_id: int) -> None:
    """Inject a recorded post-step scene state without running env ``reset_to`` managers."""
    env_ids = torch.tensor([env_id], device=env.device)
    env.scene.reset_to(recorded_state, env_ids=env_ids, is_relative=True)
    apply_sberchel_xr_teleop_pd_gains(env.scene["robot"])
    env.scene.write_data_to_sim()
    env.sim.forward()


def main() -> None:
    global is_paused

    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        return

    episode_indices_to_replay = list(args_cli.select_episodes)
    if len(episode_indices_to_replay) == 0:
        episode_indices_to_replay = list(range(episode_count))

    if args_cli.task is not None:
        env_name = args_cli.task.split(":")[-1]
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    num_envs = args_cli.num_envs
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs)

    success_term = None
    if args_cli.validate_success_rate:
        if hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
            env_cfg.terminations.success = None
        else:
            print(
                "No success termination term was found in the environment."
                " Will not be able to mark replayed demos as successful."
            )

    env_cfg.recorders = {}
    env_cfg.terminations = {}

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    policy_net = None
    if args_cli.replay_source == "policy_loco":
        policy_net = Network(env, type="zest", device="cpu", closed_kinematics=False)
        policy_net.reset()

    state_sync_enabled = (
        args_cli.replay_source == "processed_joint_targets" and not args_cli.no_state_sync
    )
    if args_cli.replay_source == "policy_loco":
        print("Replay source: policy_loco (zest leg policy + recorded upper-body joint targets)")
        print("Zest policy: active (device=cpu)")
        print("State sync: disabled (closed-loop leg policy)")
    elif args_cli.replay_source == "processed_joint_targets":
        print("Replay source: processed_joint_targets (direct joint PD targets from recording)")
        print("Zest policy: inactive")
        if state_sync_enabled:
            print("State sync: enabled (inject previous recorded scene state before each step)")
        else:
            print("State sync: disabled (open-loop replay)")
    else:
        print("Replay source: actions (raw Pink IK + leg actions; may diverge from recording)")
        print("Zest policy: inactive")
        print("State sync: disabled")

    teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    teleop_interface.add_callback("N", play_cb)
    teleop_interface.add_callback("B", pause_cb)
    print('Press "B" to pause and "N" to resume the replayed actions.')

    state_validation_enabled = args_cli.validate_states and num_envs == 1
    if args_cli.validate_states and num_envs > 1:
        print("Warning: State validation is only supported with a single environment. Skipping state validation.")

    validate_report_file = None
    if state_validation_enabled:
        validate_report_file = open(args_cli.validate_report, "w", encoding="utf-8")
        validate_report_file.write(
            f"Sberchel XR replay validation report\n"
            f"dataset: {args_cli.dataset_file}\n"
            f"task: {args_cli.task}\n"
            f"device: {args_cli.device}\n"
            f"replay_source: {args_cli.replay_source}\n"
            f"state_sync: {state_sync_enabled}\n"
            f"{'=' * 72}\n"
        )
        validate_report_file.flush()
        print(f"State validation output -> {args_cli.validate_report}")

    env.reset()
    idle_action = prepare_sberchel_xr_episode_replay(env, env_cfg, num_envs)
    teleop_interface.reset()

    episode_names = list(dataset_file_handler.get_episode_names())
    replayed_episode_count = 0
    recorded_episode_count = 0
    current_episode_indices = [None] * num_envs
    failed_demo_ids = []
    validation_matched_count = 0
    validation_mismatched_count = 0
    first_mismatch_step: int | None = None
    first_fall_step: int | None = None
    initial_root_z: float | None = None
    episode_step_counters = [0] * num_envs
    policy_loco_steps: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_envs

    def _write_validation(line: str) -> None:
        if validate_report_file is not None:
            validate_report_file.write(line + "\n")
            validate_report_file.flush()

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
            first_loop = True
            has_next_action = True
            episode_ended = [False] * num_envs
            while has_next_action:
                actions = idle_action.clone()
                has_next_action = False
                for env_id in range(num_envs):
                    env_next_action = None
                    if args_cli.replay_source == "policy_loco":
                        policy_loco_steps[env_id] = get_next_policy_loco_step(env_episode_data_map[env_id])
                        if policy_loco_steps[env_id] is not None:
                            has_next_action = True
                    else:
                        env_next_action = get_next_replay_command(
                            env_episode_data_map[env_id], args_cli.replay_source
                        )
                    step_exhausted = (
                        policy_loco_steps[env_id] is None
                        if args_cli.replay_source == "policy_loco"
                        else env_next_action is None
                    )
                    if step_exhausted:
                        if (
                            success_term is not None
                            and current_episode_indices[env_id] is not None
                            and not episode_ended[env_id]
                        ):
                            if bool(success_term.func(env, **success_term.params)[env_id]):
                                recorded_episode_count += 1
                                plural_trailing_s = "s" if recorded_episode_count > 1 else ""
                                print(
                                    f"Successfully replayed {recorded_episode_count} episode{plural_trailing_s} out"
                                    f" of {replayed_episode_count} demos."
                                )
                            elif (
                                current_episode_indices[env_id] is not None
                                and current_episode_indices[env_id] not in failed_demo_ids
                            ):
                                failed_demo_ids.append(current_episode_indices[env_id])
                            episode_ended[env_id] = True

                        next_episode_index = None
                        while episode_indices_to_replay:
                            next_episode_index = episode_indices_to_replay.pop(0)
                            if next_episode_index < episode_count:
                                episode_ended[env_id] = False
                                break
                            next_episode_index = None

                        if next_episode_index is not None:
                            replayed_episode_count += 1
                            current_episode_indices[env_id] = next_episode_index
                            print(f"{replayed_episode_count:4}: Loading #{next_episode_index} episode to env_{env_id}")
                            episode_data = dataset_file_handler.load_episode(
                                episode_names[next_episode_index], env.device
                            )
                            env_episode_data_map[env_id] = episode_data
                            episode_step_counters[env_id] = 0
                            initial_state = episode_data.get_initial_state()
                            env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)
                            idle_action = prepare_sberchel_xr_episode_replay(env, env_cfg, num_envs)
                            if policy_net is not None:
                                policy_net.reset()
                            initial_root_z = env.scene["robot"].data.root_pos_w[env_id, 2].item()
                            _write_validation(f"Episode {next_episode_index}: loaded (initial root_z={initial_root_z:.4f})")
                            if args_cli.replay_source == "policy_loco":
                                policy_loco_steps[env_id] = get_next_policy_loco_step(
                                    env_episode_data_map[env_id]
                                )
                                if policy_loco_steps[env_id] is not None:
                                    has_next_action = True
                            else:
                                env_next_action = get_next_replay_command(
                                    env_episode_data_map[env_id], args_cli.replay_source
                                )
                                if env_next_action is not None:
                                    has_next_action = True
                        else:
                            continue
                    elif args_cli.replay_source != "policy_loco":
                        has_next_action = True
                        actions[env_id] = env_next_action.to(device=env.device, dtype=actions.dtype)
                if first_loop:
                    first_loop = False
                else:
                    while is_paused:
                        env.sim.render()
                if state_sync_enabled:
                    for env_id in range(num_envs):
                        step_idx = episode_step_counters[env_id]
                        if step_idx > 0:
                            episode_data = env_episode_data_map[env_id]
                            prev_state = episode_data.get_state(step_idx - 1)
                            if prev_state is None:
                                raise ValueError(
                                    f"Episode is missing recorded states required for state sync at step {step_idx}."
                                )
                            sync_scene_to_recorded_state(env, prev_state, env_id)
                if args_cli.replay_source == "policy_loco":
                    legs_batch = torch.zeros(num_envs, LEGS_ACTION_DIM, device=env.device, dtype=torch.float32)
                    upper_batch = torch.zeros(
                        num_envs,
                        env.action_manager.total_action_dim - LEGS_ACTION_DIM,
                        device=env.device,
                        dtype=torch.float32,
                    )
                    for env_id in range(num_envs):
                        step_data = policy_loco_steps[env_id]
                        if step_data is None:
                            continue
                        loco_cmd, processed = step_data
                        if loco_cmd.ndim == 0:
                            loco_values = loco_cmd.reshape(-1)
                        else:
                            loco_values = loco_cmd.reshape(-1)[:3]
                        vx, vy, wz = (float(v) for v in loco_values.tolist())
                        legs = policy_leg_targets(policy_net, vx, vy, wz)
                        legs_batch[env_id] = torch.as_tensor(legs, dtype=torch.float32, device=env.device)
                        processed_upper = processed[LEGS_ACTION_DIM:]
                        if processed_upper.ndim > 1:
                            processed_upper = processed_upper.reshape(-1)
                        upper_batch[env_id] = processed_upper.to(device=env.device, dtype=torch.float32)
                    step_hybrid_joint_targets(env, legs_batch, upper_batch)
                elif args_cli.replay_source == "processed_joint_targets":
                    step_processed_joint_targets(env, actions)
                else:
                    env.step(actions)
                for env_id in range(num_envs):
                    if env_episode_data_map[env_id].next_action_index > episode_step_counters[env_id]:
                        episode_step_counters[env_id] += 1

                robot = env.scene["robot"]
                step_index = env_episode_data_map[0].next_state_index - 1 if state_validation_enabled else -1
                root_z = robot.data.root_pos_w[0, 2].item()
                if state_validation_enabled and initial_root_z is not None and first_fall_step is None:
                    if root_z < initial_root_z - 0.25:
                        first_fall_step = max(step_index, 0)

                if state_validation_enabled:
                    state_from_dataset = env_episode_data_map[0].get_next_state()
                    if state_from_dataset is not None:
                        action_index = env_episode_data_map[0].next_state_index - 1
                        current_runtime_state = env.scene.get_state(is_relative=True)
                        states_matched, comparison_log, max_abs_error = compare_states(
                            state_from_dataset, current_runtime_state, 0
                        )
                        if states_matched:
                            validation_matched_count += 1
                            line = f"step {action_index:4d}: matched (max_err={max_abs_error:.6f}, root_z={root_z:.4f})"
                        else:
                            validation_mismatched_count += 1
                            if first_mismatch_step is None:
                                first_mismatch_step = action_index
                            line = (
                                f"step {action_index:4d}: MISMATCH (max_err={max_abs_error:.6f}, root_z={root_z:.4f})"
                            )
                        _write_validation(line)
                        if not states_matched:
                            _write_validation(comparison_log.rstrip())
            break

    if validate_report_file is not None:
        _write_validation(f"{'=' * 72}")
        _write_validation(
            f"Summary: matched={validation_matched_count} mismatched={validation_mismatched_count} "
            f"first_mismatch_step={first_mismatch_step} first_fall_step={first_fall_step}"
        )
        validate_report_file.close()
        print(
            f"Validation done: matched={validation_matched_count} mismatched={validation_mismatched_count} "
            f"(report: {args_cli.validate_report})"
        )

    plural_trailing_s = "s" if replayed_episode_count > 1 else ""
    print(f"Finished replaying {replayed_episode_count} episode{plural_trailing_s}.")

    if success_term is not None:
        print(f"Successfully replayed: {recorded_episode_count}/{replayed_episode_count}")
        if failed_demo_ids:
            print(f"\nFailed demo IDs ({len(failed_demo_ids)} total):")
            print(f"  {sorted(failed_demo_ids)}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
