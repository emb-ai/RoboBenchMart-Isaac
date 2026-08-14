"""Scene config and helpers for the exported small_store_aij USD package."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone, standardize_xform_ops
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext

logger = logging.getLogger(__name__)

# Keep SubUSDs/ next to store.usd (relative refs in the USD).
_STORE_PKG = Path(__file__).resolve().parents[2] / "small_store_aij"
_STORE_PKG_VER2 = Path(__file__).resolve().parents[2] / "small_store_aij_ver2"
STORE_USD = (_STORE_PKG_VER2 / "store.usd").resolve()

# Exported store layout (see small_store_aij/store.usd).
STORE_DSYNTH_SUBPATH = "/World/envs/env_0/dsynth_scene"
STORE_COLLISIONS_SUBPATH = "/World/collisions"
STORE_ROOM_MATERIALS_SUBPATH = "/World/RoomMaterials"


def _spawn_store_usd_subpath_impl(
    prim_path: str,
    cfg: "StoreSubpathUsdCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(cfg.usd_path, Sdf.Path(cfg.usd_subpath))
    standardize_xform_ops(prim, translation=translation, orientation=orientation)
    return prim


@clone
def spawn_store_usd_subpath(
    prim_path: str,
    cfg: "StoreSubpathUsdCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Reference a sub-prim from the store USD (avoids double env/dsynth_scene nesting)."""
    return _spawn_store_usd_subpath_impl(prim_path, cfg, translation=translation, orientation=orientation)


def spawn_store_usd_subpath_global(
    prim_path: str,
    cfg: "StoreSubpathUsdCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Reference a store USD sub-prim once (no per-env clone)."""
    return _spawn_store_usd_subpath_impl(prim_path, cfg, translation=translation, orientation=orientation)


@configclass
class StoreSubpathUsdCfg(SpawnerCfg):
    """Spawn by referencing a sub-prim inside a packaged store USD."""

    func = spawn_store_usd_subpath
    usd_path: str = MISSING
    usd_subpath: str = MISSING


@configclass
class StoreSubpathUsdGlobalCfg(StoreSubpathUsdCfg):
    """Like ``StoreSubpathUsdCfg`` but without env cloning (shared prims like RoomMaterials)."""

    func = spawn_store_usd_subpath_global


def resolve_lab_physics_scene_path(stage: Usd.Stage, physics_prim_path: str = "/physicsScene") -> str:
    """Return Isaac Lab's active physics scene (defaults to ``/physicsScene``)."""
    prim = stage.GetPrimAtPath(physics_prim_path)
    if prim.IsValid() and prim.HasAPI(PhysxSchema.PhysxSceneAPI):
        return physics_prim_path

    for prim in stage.Traverse():
        if not prim.HasAPI(PhysxSchema.PhysxSceneAPI):
            continue
        path = prim.GetPath().pathString
        if path.startswith("/World/envs/"):
            continue
        return path

    raise RuntimeError("No Isaac Lab physics scene found on stage.")


def disable_stray_physics_scenes(stage: Usd.Stage, lab_physics_scene_path: str) -> int:
    """Disable extra PhysicsScene prims (e.g. mistakenly composed from exports)."""
    disabled = 0
    for prim in stage.Traverse():
        if not prim.HasAPI(PhysxSchema.PhysxSceneAPI):
            continue
        path = prim.GetPath().pathString
        if path == lab_physics_scene_path:
            continue
        physx_scene = PhysxSchema.PhysxSceneAPI(prim)
        if physx_scene.GetEnabledAttr().Get():
            physx_scene.GetEnabledAttr().Set(False)
            disabled += 1
            logger.info("[SberDemo] Disabled stray physics scene: %s", path)
    return disabled


def iter_store_rigid_bodies(
    stage: Usd.Stage,
    env_prim_paths: list[str],
    store_root_name: str = "dsynth_scene",
) -> Iterator[tuple[Usd.Prim, bool]]:
    """Yield enabled rigid-body prims under each env's store root as ``(prim, is_kinematic)``."""
    for env_path in env_prim_paths:
        store_root = stage.GetPrimAtPath(f"{env_path}/{store_root_name}")
        if not store_root.IsValid():
            logger.warning("[SberDemo] Store root missing under %s", env_path)
            continue
        for prim in Usd.PrimRange(store_root):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            rb = UsdPhysics.RigidBodyAPI(prim)
            if not rb.GetRigidBodyEnabledAttr().Get():
                continue
            yield prim, rb.GetKinematicEnabledAttr().Get()


def _world_pose_wxyz(prim: Usd.Prim, xform_cache: UsdGeom.XformCache) -> tuple[np.ndarray, np.ndarray]:
    """Return world position (3,) and orientation wxyz (4,) for a prim."""
    matrix = xform_cache.GetLocalToWorldTransform(prim)
    translation = matrix.ExtractTranslation()
    quat = matrix.ExtractRotationQuat()
    imag = quat.GetImaginary()
    position = np.array([translation[0], translation[1], translation[2]], dtype=np.float32)
    orientation = np.array(
        [quat.GetReal(), imag[0], imag[1], imag[2]],
        dtype=np.float32,
    )
    return position, orientation


@dataclass
class StoreRigidBodySnapshot:
    """Initial world poses for unregistered store USD rigid bodies."""

    prim_paths: list[str] = field(default_factory=list)
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    orientations: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    _physx_view: object | None = field(default=None, repr=False)

    def _ensure_view(self, sim: SimulationContext):
        if self._physx_view is None:
            self._physx_view = sim.physics_sim_view.create_rigid_body_view(self.prim_paths)
            if self._physx_view._backend is None:
                raise RuntimeError(
                    "[SberDemo] Failed to create rigid-body view for store products. "
                    "Check that simulation is playing and prim paths are valid."
                )
        return self._physx_view

    def capture_live_state(self, sim: SimulationContext) -> dict | None:
        """Current PhysX poses + velocities of the bodies, for mid-episode checkpoints.

        Unlike the dataclass fields (authoring-time USD poses used by :meth:`restore`),
        this reads the live simulated state, so a later :meth:`restore_live_state`
        puts dropped/knocked-over products back where they were at capture time.
        """
        if len(self.prim_paths) == 0:
            return None
        view = self._ensure_view(sim)
        return {
            "transforms": view.get_transforms().clone(),  # (N, 7) pos + quat xyzw
            "velocities": view.get_velocities().clone(),  # (N, 6)
        }

    def restore_live_state(
        self,
        sim: SimulationContext,
        state: dict | None,
        *,
        only_moved: bool = True,
        pos_tol: float = 0.02,
        ang_tol: float = 4e-3,
        lin_vel_tol: float = 0.25,
    ) -> int:
        """Write a :meth:`capture_live_state` snapshot back to PhysX.

        With ``only_moved`` (default) only bodies genuinely disturbed since the
        snapshot are written: displaced beyond ``pos_tol`` (m), rotated beyond
        ``ang_tol`` (as 1 - |quat dot|; 4e-3 is about 10 degrees), or currently
        translating faster than ``lin_vel_tol`` (m/s, catches a mid-fall body
        near its captured pose). The thresholds are deliberately coarse — well
        above the contact micro-jitter of never-sleeping shelf products (mm /
        mrad / small angular velocities), which must NOT count as moved: a PhysX
        state write wakes the body and drops its contact warm-start, so writing
        jitter-flagged products makes whole shelves visibly re-settle on every
        restore. Angular velocity is ignored for the same reason.

        Returns the number of bodies written.
        """
        if state is None or len(self.prim_paths) == 0:
            return 0

        import torch

        view = self._ensure_view(sim)
        if only_moved:
            current_tf = view.get_transforms()
            current_vel = view.get_velocities()
            dpos = torch.linalg.norm(current_tf[:, :3] - state["transforms"][:, :3], dim=-1)
            quat_dot = (
                (current_tf[:, 3:] * state["transforms"][:, 3:]).sum(dim=-1).abs().clamp(max=1.0)
            )
            moved = (
                (dpos > pos_tol)
                | (quat_dot < 1.0 - ang_tol)
                | (torch.linalg.norm(current_vel[:, :3], dim=-1) > lin_vel_tol)
            )
            indices = torch.nonzero(moved).flatten().to(device=sim.device, dtype=torch.int32)
        else:
            indices = torch.arange(len(self.prim_paths), device=sim.device, dtype=torch.int32)
        if indices.numel() == 0:
            return 0
        view.set_transforms(state["transforms"], indices=indices)
        view.set_velocities(state["velocities"], indices=indices)
        return int(indices.numel())

    def restore(self, sim: SimulationContext) -> int:
        """Write snapshot poses to PhysX and zero velocities. Returns body count."""
        if len(self.prim_paths) == 0:
            return 0

        import torch
        from isaaclab.utils import math as math_utils

        self._ensure_view(sim)

        device = sim.device
        poses_wxyz = torch.cat(
            [
                torch.from_numpy(self.positions).to(device=device, dtype=torch.float32),
                torch.from_numpy(self.orientations).to(device=device, dtype=torch.float32),
            ],
            dim=-1,
        )
        poses_xyzw = poses_wxyz.clone()
        poses_xyzw[:, 3:] = math_utils.convert_quat(poses_wxyz[:, 3:], to="xyzw")
        indices = torch.arange(len(self.prim_paths), device=device, dtype=torch.int32)
        self._physx_view.set_transforms(poses_xyzw, indices=indices)
        self._physx_view.set_velocities(
            torch.zeros(len(self.prim_paths), 6, device=device, dtype=torch.float32),
            indices=indices,
        )
        return len(self.prim_paths)


def capture_store_rigid_body_snapshot(
    scene: InteractiveScene,
    *,
    store_root_name: str = "dsynth_scene",
    dynamic_only: bool = True,
) -> StoreRigidBodySnapshot:
    """Capture authoring-time world poses for store rigid bodies from USD."""
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    prim_paths: list[str] = []
    positions: list[np.ndarray] = []
    orientations: list[np.ndarray] = []

    for prim, is_kinematic in iter_store_rigid_bodies(scene.stage, scene.env_prim_paths, store_root_name):
        if dynamic_only and is_kinematic:
            continue
        position, orientation = _world_pose_wxyz(prim, xform_cache)
        prim_paths.append(prim.GetPath().pathString)
        positions.append(position)
        orientations.append(orientation)

    if prim_paths:
        pos_array = np.stack(positions, axis=0)
        ori_array = np.stack(orientations, axis=0)
    else:
        pos_array = np.zeros((0, 3), dtype=np.float32)
        ori_array = np.zeros((0, 4), dtype=np.float32)

    logger.info("[SberDemo] Captured initial poses for %d store rigid bodies", len(prim_paths))
    return StoreRigidBodySnapshot(
        prim_paths=prim_paths,
        positions=pos_array,
        orientations=ori_array,
    )


def rebind_store_rigid_bodies(
    stage: Usd.Stage,
    env_prim_paths: list[str],
    lab_physics_scene_path: str,
    store_root_name: str = "dsynth_scene",
    *,
    include_kinematic: bool = True,
) -> tuple[int, int]:
    """Rebind store rigid bodies (products + kinematic shelves/floor) to the lab physics scene."""
    lab_scene_prim = stage.GetPrimAtPath(lab_physics_scene_path)
    if not lab_scene_prim.IsValid():
        raise RuntimeError(f"Lab physics scene prim not found: {lab_physics_scene_path}")

    rebound_dynamic = 0
    rebound_kinematic = 0
    for prim, is_kinematic in iter_store_rigid_bodies(stage, env_prim_paths, store_root_name):
        if is_kinematic and not include_kinematic:
            continue
        rb = UsdPhysics.RigidBodyAPI(prim)
        rb.GetSimulationOwnerRel().SetTargets([lab_scene_prim.GetPath()])
        if is_kinematic:
            rebound_kinematic += 1
        else:
            rebound_dynamic += 1
    return rebound_dynamic, rebound_kinematic


def _load_store_room_material_map(store_usd: Path = STORE_USD) -> dict[str, str]:
    """Map dsynth_scene room prim names to ``/World/RoomMaterials/*`` paths in the export."""
    stage = Usd.Stage.Open(str(store_usd))
    root = stage.GetPrimAtPath(STORE_DSYNTH_SUBPATH)
    if not root.IsValid():
        raise RuntimeError(f"Store dsynth_scene root missing in {store_usd}")

    material_map: dict[str, str] = {}
    for child in root.GetChildren():
        name = child.GetName()
        if not (name.startswith("wall_") or name in ("floor", "ceiling")):
            continue
        material = UsdShade.MaterialBindingAPI(child).GetDirectBinding().GetMaterial()
        if material:
            material_map[name] = material.GetPath().pathString
    return material_map


def fix_store_room_material_bindings(
    scene: InteractiveScene,
    *,
    store_usd: Path = STORE_USD,
    store_root_name: str = "dsynth_scene",
) -> None:
    """Re-apply room material bindings after subpath USD references.

    When ``dsynth_scene`` is referenced as a sub-prim, USD ignores xform bindings that
    target ``/World/RoomMaterials/*`` (outside the reference scope). Meshes then fall
    back to the baked ``bricks3`` geometry materials. Rebind on the live stage with
    ``strongerThanDescendants`` so plaster/tiles win over mesh-local shaders.
    """
    material_map = _load_store_room_material_map(store_usd)
    stage = scene.stage
    rebound = 0
    skipped = 0
    for env_path in scene.env_prim_paths:
        store_root = stage.GetPrimAtPath(f"{env_path}/{store_root_name}")
        if not store_root.IsValid():
            logger.warning("[SberDemo] Store root missing under %s", env_path)
            continue
        for child in store_root.GetChildren():
            mat_path = material_map.get(child.GetName())
            if mat_path is None:
                continue
            mat_prim = stage.GetPrimAtPath(mat_path)
            if not mat_prim.IsValid():
                logger.warning("[SberDemo] Room material missing on stage: %s", mat_path)
                skipped += 1
                continue
            binding_api = UsdShade.MaterialBindingAPI.Apply(child)
            binding_api.Bind(
                UsdShade.Material(mat_prim),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
            rebound += 1
    print(f"[INFO] SberDemo room materials: rebound={rebound}, skipped={skipped}")


def fix_store_usd_physics_for_isaac_lab(
    scene: InteractiveScene,
    *,
    physics_prim_path: str = "/physicsScene",
    store_root_name: str = "dsynth_scene",
) -> None:
    """Bind all store rigid bodies to Isaac Lab's physics scene (call after sim play/reset)."""
    stage = scene.stage
    lab_scene_path = resolve_lab_physics_scene_path(stage, physics_prim_path)
    disabled = disable_stray_physics_scenes(stage, lab_scene_path)
    rebound_dynamic, rebound_kinematic = rebind_store_rigid_bodies(
        stage, scene.env_prim_paths, lab_scene_path, store_root_name
    )
    print(
        f"[INFO] SberDemo store physics: lab scene={lab_scene_path}, "
        f"disabled_stray_scenes={disabled}, "
        f"rebound_dynamic={rebound_dynamic}, rebound_kinematic={rebound_kinematic}"
    )


@configclass
class SberDemoSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    # Set by demo env cfgs; ``None`` is skipped by InteractiveScene.
    left_head_camera: CameraCfg | None = None

    dsynth_scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dsynth_scene",
        spawn=StoreSubpathUsdCfg(
            usd_path=str(STORE_USD),
            usd_subpath=STORE_DSYNTH_SUBPATH,
        ),
    )

    store_collisions = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/store_collisions",
        spawn=StoreSubpathUsdCfg(
            usd_path=str(STORE_USD),
            usd_subpath=STORE_COLLISIONS_SUBPATH,
        ),
    )

    # Wall/floor/ceiling xforms bind to /World/RoomMaterials/*; compose it once globally.
    store_room_materials = AssetBaseCfg(
        prim_path="/World/RoomMaterials",
        spawn=StoreSubpathUsdGlobalCfg(
            usd_path=str(STORE_USD),
            usd_subpath=STORE_ROOM_MATERIALS_SUBPATH,
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.35, 0.35, 0.35)),
    )


class SberDemoInteractiveScene(InteractiveScene):
    """InteractiveScene for the exported store USD package."""

    def __init__(self, cfg: SberDemoSceneCfg):
        super().__init__(cfg)
        fix_store_room_material_bindings(self)
        self._store_rigid_snapshot = capture_store_rigid_body_snapshot(self, dynamic_only=True)

    def reset_store_rigid_bodies(self, sim: SimulationContext) -> int:
        """Restore dynamic store products to their initial USD poses."""
        count = self._store_rigid_snapshot.restore(sim)
        if count > 0:
            logger.info("[SberDemo] Reset %d store rigid bodies to initial poses", count)
        return count

    def capture_store_rigid_body_state(self, sim: SimulationContext) -> dict | None:
        """Live poses + velocities of dynamic store products (mid-episode checkpoint)."""
        return self._store_rigid_snapshot.capture_live_state(sim)

    def restore_store_rigid_body_state(self, sim: SimulationContext, state: dict | None) -> int:
        """Put dynamic store products back to a captured live state."""
        return self._store_rigid_snapshot.restore_live_state(sim, state)
