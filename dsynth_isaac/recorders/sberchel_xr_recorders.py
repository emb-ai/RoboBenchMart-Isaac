"""Sberchel XR demo recorders for Isaac Lab RecorderManager."""

from __future__ import annotations

import queue
import threading
import traceback

import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.managers.recorder_manager import RecorderTermCfg
from isaaclab.utils import configclass

LOCO_COMMAND_ATTR = "_sberchel_loco_command"
TELEOP_STATE_ATTR = "_sberchel_teleop_state"
ZEST_UPPER_REF_ATTR = "_sberchel_zest_upper_ref"

# Optional dict set on an EpisodeData; each entry is written as an HDF5 attr on
# the demo group at export (e.g. {"target_item": ..., "task_instruction": ...}).
EPISODE_EXTRA_ATTRS_FIELD = "sberchel_extra_attrs"


def stash_episode_extra_attrs(episode, **attrs) -> None:
    """Attach per-demo HDF5 attrs to an in-progress episode (merged on repeat calls)."""
    merged = dict(getattr(episode, EPISODE_EXTRA_ATTRS_FIELD, {}) or {})
    merged.update(attrs)
    setattr(episode, EPISODE_EXTRA_ATTRS_FIELD, merged)


def _write_episode_with_extra_attrs(handler, episode, demo_id=None) -> None:
    """``handler.write_episode`` plus the episode's stashed per-demo attrs.

    The demo group name must be derived *before* the write increments the
    handler's demo counter; the group is absent if the episode was empty.
    """
    group_name = f"demo_{demo_id if demo_id is not None else handler._demo_count}"
    handler.write_episode(episode, demo_id)
    extra = getattr(episode, EPISODE_EXTRA_ATTRS_FIELD, None)
    if not extra:
        return
    group = getattr(handler, "_hdf5_data_group", {}).get(group_name)
    if group is None:
        return
    for key, value in extra.items():
        group.attrs[key] = value


def stash_locomotion_command(env, vx: float, vy: float, wz: float) -> None:
    """Store per-env locomotion velocity commands for the next ``env.step`` recorder tick."""
    setattr(
        env,
        LOCO_COMMAND_ATTR,
        torch.tensor([[vx, vy, wz]], dtype=torch.float32, device=env.device).repeat(env.num_envs, 1),
    )


def stash_teleop_state(env, *, loco_mode: str, squeeze: float, legacy_gains: bool) -> None:
    """Store operator/plant state for the next ``env.step`` recorder tick.

    ``loco_mode`` and ``legacy_gains`` are what let a downstream consumer tell
    which plant produced a segment; ``squeeze`` is the grasp-intent channel that
    the finger targets alone do not disambiguate (a finger blocked by an object
    reads the same whether it is squeezing hard or merely resting).
    """
    setattr(
        env,
        TELEOP_STATE_ATTR,
        torch.tensor(
            [[1.0 if loco_mode == "locomotion" else 0.0, float(squeeze), float(legacy_gains)]],
            dtype=torch.float32,
            device=env.device,
        ).repeat(env.num_envs, 1),
    )


def stash_zest_upper_ref(env, ref: list[float]) -> None:
    """Store the 13-value zest torso-block reference for the next recorder tick.

    Layout: ``[torso_yaw, arms(10, POLICY_ARM_JOINT_NAMES order), neck_yaw,
    neck_pitch]``, absolute radians -- exactly what the recorder fed into
    ``Network.set_joint_overrides`` for this step. Replay needs it to re-run
    zest bit-exact: unlike the velocity command, the reference cannot be
    reconstructed from anything else in the dataset (it is the *previous* step's
    raw Pink IK arm solution, which the processed arm targets no longer hold
    once zest's output overwrites them).
    """
    setattr(
        env,
        ZEST_UPPER_REF_ATTR,
        torch.tensor([list(ref)], dtype=torch.float32, device=env.device).repeat(
            env.num_envs, 1
        ),
    )


class ZestUpperRefRecorder(RecorderTerm):
    """Record the stashed zest upper-body reference (zest_upper sessions only).

    Returns no row when nothing was stashed, so legacy (non ``--zest_upper``)
    sessions keep their exact dataset schema.
    """

    def record_pre_step(self):
        ref = getattr(self._env, ZEST_UPPER_REF_ATTR, None)
        if ref is None:
            return None, None
        return "zest_upper_ref", ref


@configclass
class ZestUpperRefRecorderCfg(RecorderTermCfg):
    """Configuration for the zest upper-body reference recorder term."""

    class_type: type[RecorderTerm] = ZestUpperRefRecorder


class LocomotionCommandRecorder(RecorderTerm):
    """Record ``(vx, vy, wz)`` stashed on the env before each step."""

    def record_pre_step(self):
        command = getattr(self._env, LOCO_COMMAND_ATTR, None)
        if command is None:
            command = torch.zeros(self._env.num_envs, 3, device=self._env.device)
        return "loco_commands", command


@configclass
class LocomotionCommandRecorderCfg(RecorderTermCfg):
    """Configuration for the locomotion command recorder term."""

    class_type: type[RecorderTerm] = LocomotionCommandRecorder


class PreStepRgbCameraObservationsRecorder(RecorderTerm):
    """Record ``observations.rgb_camera`` terms under flat ``obs/<term>`` keys.

    ``ActionStateRecorderManagerCfg`` only writes ``obs_buf["policy"]``. Sberchel puts
    left-head RGB under ``rgb_camera``, so demos need this term or HDF5 exports omit
    video even when ``--enable_cameras`` is set.

    Terms are flattened (``obs/head_img``, not ``obs/rgb_camera/head_img``) so
    Isaac Lab's ``hdf5_to_mp4.py --input_keys head_img`` works without nested paths.
    """

    def record_pre_step(self):
        rgb_obs = self._env.obs_buf.get("rgb_camera")
        if rgb_obs is None or (isinstance(rgb_obs, dict) and len(rgb_obs) == 0):
            return None, None
        # Flat under ``obs/`` (policy recorder also uses ``obs``; keys merge).
        return "obs", rgb_obs


@configclass
class PreStepRgbCameraObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the RGB camera observation recorder term."""

    class_type: type[RecorderTerm] = PreStepRgbCameraObservationsRecorder


class TeleopStateRecorder(RecorderTerm):
    """Record ``(is_locomotion, squeeze, legacy_gains)`` stashed before each step."""

    def record_pre_step(self):
        state = getattr(self._env, TELEOP_STATE_ATTR, None)
        if state is None:
            state = torch.zeros(self._env.num_envs, 3, device=self._env.device)
        return "teleop_state", state


@configclass
class TeleopStateRecorderCfg(RecorderTermCfg):
    """Configuration for the teleop state recorder term."""

    class_type: type[RecorderTerm] = TeleopStateRecorder


class PostStepStoreProductsRecorder(RecorderTerm):
    """Record store product rigid-body poses and velocities each step.

    Store products are plain USD rigid bodies, not registered scene assets, so
    the scene-state recorder misses them. Without these rows a replay can only
    state-sync the robot: objects run open-loop physics and a grasped item
    drifts out of the hand. Replay restores the previous step's rows before
    each step, pinning products to the recorded trajectory.

    Single-env only (teleop always is): the PhysX view spans every env's
    products in one flat list.
    """

    def record_post_step(self):
        if self._env.num_envs != 1:
            return None, None
        capture = getattr(self._env.scene, "capture_store_rigid_body_state", None)
        state = capture(self._env.sim) if capture is not None else None
        if state is None:
            return None, None
        return "store_products", {
            "transforms": state["transforms"].unsqueeze(0),
            "velocities": state["velocities"].unsqueeze(0),
        }


@configclass
class PostStepStoreProductsRecorderCfg(RecorderTermCfg):
    """Configuration for the store product state recorder term."""

    class_type: type[RecorderTerm] = PostStepStoreProductsRecorder


class PostStepJointTargetsRecorder(RecorderTerm):
    """Record full-articulation commanded and achieved joint positions.

    ``processed_actions`` only covers joints owned by an action term, and the
    upper-body term reports Cartesian-derived IK output. green_challenge's VLA
    interface is joint-space throughout (``legs_joint_pos`` / ``torso_joint_pos``
    / ``finger_joint_pos``), so recording the raw articulation vectors -- plus
    the joint-name order in the episode metadata -- is what makes conversion to
    their layout mechanical rather than guesswork.

    Both are needed: the *target* carries grip intent (a blocked finger's
    achieved angle says nothing about how hard it is squeezing), while the
    *achieved* value is the plant-independent physical fact that survives
    relabelling onto a different controller.
    """

    def record_post_step(self):
        robot = self._env.scene["robot"]
        return "joint_targets", {
            "joint_pos_target": robot.data.joint_pos_target.clone(),
            "joint_pos": robot.data.joint_pos.clone(),
            "joint_vel": robot.data.joint_vel.clone(),
        }


@configclass
class PostStepJointTargetsRecorderCfg(RecorderTermCfg):
    """Configuration for the joint target/state recorder term."""

    class_type: type[RecorderTerm] = PostStepJointTargetsRecorder


class AsyncDatasetFileHandler:
    """Run ``write_episode``/``flush`` of a dataset file handler on a background thread.

    ``RecorderManager.export_episodes`` writes the whole episode (gzip-compressed,
    camera frames included) synchronously inside ``env.reset()``, stalling teleop
    for seconds between demos. Export hands each episode buffer off wholesale and
    immediately replaces it with a fresh ``EpisodeData``, so nothing mutates the
    episode after enqueue — a single writer thread can compress and write it while
    the next demo is already being collected. All file access (including attrs via
    ``add_env_args``) goes through the queue so only the worker touches h5py.
    """

    def __init__(self, inner):
        self._inner = inner
        self._queue: queue.Queue = queue.Queue()
        self._closed = False
        # Surfaced to the recording loop: the operator is in the headset and
        # never sees a worker-thread traceback, while RecorderManager's export
        # counter advances at enqueue time regardless of write outcome.
        self.failed_writes = 0
        self.last_write_error: str | None = None
        self._worker = threading.Thread(target=self._run, name="episode-writer", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                fn_name, args, kwargs = item
                self._dispatch(fn_name, args, kwargs)
                if fn_name == "write_episode":
                    print(f"[episode-writer] episode saved (total in file: {self._inner.demo_count})", flush=True)
            except Exception as exc:
                if item is not None and item[0] == "write_episode":
                    self.failed_writes += 1
                self.last_write_error = repr(exc)
                print("[episode-writer] background write failed:", flush=True)
                traceback.print_exc()
            finally:
                self._queue.task_done()

    def wait_idle(self) -> None:
        """Block until every queued write/flush has been processed."""
        self._queue.join()

    def _dispatch(self, fn_name: str, args, kwargs) -> None:
        if fn_name == "write_episode":
            _write_episode_with_extra_attrs(self._inner, *args, **kwargs)
        else:
            getattr(self._inner, fn_name)(*args, **kwargs)

    def _submit(self, fn_name: str, *args, **kwargs) -> None:
        if self._closed:
            self._dispatch(fn_name, args, kwargs)
            return
        self._queue.put((fn_name, args, kwargs))

    def write_episode(self, episode, demo_id=None) -> None:
        self._submit("write_episode", episode, demo_id)

    def flush(self) -> None:
        self._submit("flush")

    def add_env_args(self, env_args: dict) -> None:
        self._submit("add_env_args", env_args)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put(None)
            self._worker.join()
        self._inner.close()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def enable_async_episode_export(env) -> None:
    """Wrap the env recorder manager's file handlers so episode export never blocks the sim loop."""
    recorder_manager = getattr(env, "unwrapped", env).recorder_manager
    for attr in ("_dataset_file_handler", "_failed_episode_dataset_file_handler"):
        handler = getattr(recorder_manager, attr, None)
        if handler is not None and not isinstance(handler, AsyncDatasetFileHandler):
            setattr(recorder_manager, attr, AsyncDatasetFileHandler(handler))


@configclass
class SberchelXRRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Action/state recorder plus locomotion commands, teleop state, joint targets and RGB."""

    record_pre_step_loco_commands = LocomotionCommandRecorderCfg()
    record_pre_step_teleop_state = TeleopStateRecorderCfg()
    record_pre_step_zest_upper_ref = ZestUpperRefRecorderCfg()
    record_post_step_joint_targets = PostStepJointTargetsRecorderCfg()
    record_post_step_store_products = PostStepStoreProductsRecorderCfg()
    record_pre_step_rgb_camera_observations = PreStepRgbCameraObservationsRecorderCfg()
