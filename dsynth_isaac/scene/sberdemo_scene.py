"""Scene config and helpers for the exported small_store_aij USD package."""

from __future__ import annotations

import logging
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone, standardize_xform_ops
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass
from pxr import PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade

logger = logging.getLogger(__name__)

# Keep SubUSDs/ next to store.usd (relative refs in the USD).
_STORE_PKG = Path(__file__).resolve().parents[2] / "small_store_aij"
STORE_USD = (_STORE_PKG / "store.usd").resolve()

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
            is_kinematic = rb.GetKinematicEnabledAttr().Get()
            if is_kinematic and not include_kinematic:
                continue
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
