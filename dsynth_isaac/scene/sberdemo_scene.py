"""Scene config and helpers for the exported small_store_aij USD package."""

from __future__ import annotations

import logging
import math
import re
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
# ver3 = ver2 with aggressive convex-decomposition colliders (error 3%,
# 64 hulls; too slow with all store products). ver4 = ver2 with product
# colliders at errorPercentage=4/maxConvexHulls=24: tighter hulls at
# ver2 sim cost (note: shrinkWrap is a cooking no-op in Isaac Sim 5.x).
_STORE_PKG_VER3 = Path(__file__).resolve().parents[2] / "small_store_aij_ver3"
_STORE_PKG_VER4 = Path(__file__).resolve().parents[2] / "small_store_aij_ver4"
# ver5 = ver2 de-branded: identical layout, poses, masses and 2228 item prims,
# with synthetic packaging art (see the package's DEBRAND_AUDIT.md). Built by
# scripts/debrand_store/. Product prim/asset ids differ from ver2
# (energy_can_02, not redbull_1_rotated).
# NOTE: these constants resolve against the REPO root, so each package needs a
# symlink here pointing at ../<package> - as ver2/ver3/ver4 have.
_STORE_PKG_VER5 = Path(__file__).resolve().parents[2] / "small_store_aij_ver5_debranded"
_STORE_PKG_VER6 = Path(__file__).resolve().parents[2] / "small_store_aij_ver6_blurred"
_STORE_PKG_VER7 = Path(__file__).resolve().parents[2] / "small_store_aij_ver7"
_STORE_PKG_VER8 = Path(__file__).resolve().parents[2] / "small_store_aij_ver8_monotone"
STORE_USD = (_STORE_PKG_VER8 / "store.usd").resolve()

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


# Store item prims are named ``r{row}_s{unit}_b{board}_item_{idx}`` under each
# shelf unit's ``visual_items`` scope; group(1) is the board label.
_STORE_ITEM_NAME_RE = re.compile(r"^(r\d+_s\d+_b\d+)_item_\d+$")


def deactivate_off_board_products(
    stage: Usd.Stage,
    env_prim_paths: list[str],
    keep_boards: list[str],
    store_root_name: str = "dsynth_scene",
) -> tuple[int, int]:
    """Deactivate store item prims that are not on the given boards.

    ``keep_boards`` entries are board labels as encoded in the item prim names
    (``"r01_s02_b2"`` keeps every ``r01_s02_b2_item_*``). Must run before the
    first ``sim.reset()``: PhysX never parses inactive prims, so stripped
    products cost nothing to simulate or render. Returns (kept, removed);
    raises if a requested board matches no item (typo protection — a lean
    recording session against an empty board would be wasted).
    """
    keep = set(keep_boards)
    matched: set[str] = set()
    to_deactivate: list[Usd.Prim] = []
    kept = 0
    for env_path in env_prim_paths:
        store_root = stage.GetPrimAtPath(f"{env_path}/{store_root_name}")
        if not store_root.IsValid():
            logger.warning("[SberDemo] Store root missing under %s", env_path)
            continue
        it = iter(Usd.PrimRange(store_root))
        for prim in it:
            match = _STORE_ITEM_NAME_RE.match(prim.GetName())
            if match is None or prim.GetParent().GetName() != "visual_items":
                continue
            it.PruneChildren()
            if match.group(1) in keep:
                matched.add(match.group(1))
                kept += 1
            else:
                to_deactivate.append(prim)
    unmatched = keep - matched
    if unmatched:
        raise ValueError(
            f"[SberDemo] keep_product_boards entries matched no store items: {sorted(unmatched)}. "
            "Board labels come from item prim names, e.g. 'r01_s02_b2'."
        )
    for prim in to_deactivate:
        prim.SetActive(False)
    return kept, len(to_deactivate)


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
class FirstRowJitterCfg:
    """Per-reset pose jitter for aisle-facing first-row products.

    Magnitudes are bounded by the measured packing of small_store_aij_ver2:
    neighbor gaps along a row are 2.5cm at the 5th percentile (so +/-1cm keeps
    worst-case approach under the gap), the second row touches the first
    (median 1.2cm, quarter of items <=5mm — never jitter inward), and ~3cm of
    shelf board remains in front of the first row (outward pull <=1.5cm keeps
    items supported). Yaw beyond ~5 deg sweeps box corners into the touching
    row behind; raise it only for stores of mostly cylindrical items.
    """

    enabled: bool = True
    along_row_max: float = 0.01
    """Independent per-item shift along the row (x), +/- meters."""
    row_shift_max: float = 0.02
    """Coherent whole-row shift along the row (x), +/- meters (gaps unchanged)."""
    outward_max: float = 0.015
    """Pull toward the aisle (facing y direction), 0..max meters. Never inward."""
    yaw_max_deg: float = 5.0
    """Yaw about the item's own center, +/- degrees."""


@dataclass
class StoreRigidBodySnapshot:
    """Initial world poses for unregistered store USD rigid bodies."""

    prim_paths: list[str] = field(default_factory=list)
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    orientations: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    # Filled by classify_first_row_products(): indices into the arrays above for
    # aisle-facing first-row items, their facing sign along y, and a row id per
    # item so whole rows can be shifted coherently.
    first_row_indices: np.ndarray | None = field(default=None, repr=False)
    first_row_facing: np.ndarray | None = field(default=None, repr=False)
    first_row_row_ids: np.ndarray | None = field(default=None, repr=False)
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
        ang_vel_tol: float = 0.5,
        body_indices=None,
    ) -> int:
        """Write a :meth:`capture_live_state` snapshot back to PhysX.

        With ``only_moved`` (default) only bodies genuinely disturbed since the
        snapshot are written: displaced beyond ``pos_tol`` (m), rotated beyond
        ``ang_tol`` (as 1 - |quat dot|; 4e-3 is about 10 degrees), currently
        translating faster than ``lin_vel_tol`` (m/s, catches a mid-fall body
        near its captured pose), or spinning faster than ``ang_vel_tol``
        (rad/s, catches a body rocking/swinging about its rest pose — such an
        oscillation can stay under all three pose/linear gates forever). The
        thresholds are deliberately coarse — well above the contact
        micro-jitter of never-sleeping shelf products (mm / mrad / mrad-per-s
        angular velocities), which must NOT count as moved: a PhysX state
        write wakes the body and drops its contact warm-start, so writing
        jitter-flagged products makes whole shelves visibly re-settle on every
        restore.

        ``body_indices`` (integer tensor of indices into this snapshot's prim
        order) declares which bodies the state rows describe, row ``k`` of the
        state mapping to body ``body_indices[k]`` — for states captured on a
        scene with fewer products (lean recording via ``keep_product_boards``)
        replayed against the full store. ``None`` means the state covers every
        body in snapshot order.

        Returns the number of bodies written.
        """
        if state is None or len(self.prim_paths) == 0:
            return 0

        import torch

        view = self._ensure_view(sim)
        state_tf = state["transforms"]
        state_vel = state["velocities"]
        if body_indices is not None:
            body_idx = torch.as_tensor(body_indices, device=sim.device, dtype=torch.long)
            # Scatter the recorded rows into full-view-sized buffers (the PhysX
            # tensor API indexes data buffers by view row); unmapped rows keep
            # their current state and are never selected below.
            full_tf = view.get_transforms().clone()
            full_vel = view.get_velocities().clone()
            full_tf[body_idx] = state_tf.to(dtype=full_tf.dtype, device=full_tf.device)
            full_vel[body_idx] = state_vel.to(dtype=full_vel.dtype, device=full_vel.device)
            state_tf, state_vel = full_tf, full_vel
        else:
            body_idx = None
        if only_moved:
            current_tf = view.get_transforms()
            current_vel = view.get_velocities()
            dpos = torch.linalg.norm(current_tf[:, :3] - state_tf[:, :3], dim=-1)
            quat_dot = (
                (current_tf[:, 3:] * state_tf[:, 3:]).sum(dim=-1).abs().clamp(max=1.0)
            )
            moved = (
                (dpos > pos_tol)
                | (quat_dot < 1.0 - ang_tol)
                | (torch.linalg.norm(current_vel[:, :3], dim=-1) > lin_vel_tol)
                | (torch.linalg.norm(current_vel[:, 3:], dim=-1) > ang_vel_tol)
            )
            if body_idx is not None:
                mapped = torch.zeros_like(moved)
                mapped[body_idx] = True
                moved &= mapped
            indices = torch.nonzero(moved).flatten().to(device=sim.device, dtype=torch.int32)
        elif body_idx is not None:
            indices = body_idx.to(device=sim.device, dtype=torch.int32)
        else:
            indices = torch.arange(len(self.prim_paths), device=sim.device, dtype=torch.int32)
        if indices.numel() == 0:
            return 0
        view.set_transforms(state_tf, indices=indices)
        view.set_velocities(state_vel, indices=indices)
        return int(indices.numel())

    def restore(self, sim: SimulationContext, jitter: FirstRowJitterCfg | None = None) -> int:
        """Write snapshot poses to PhysX and zero velocities. Returns body count.

        With ``jitter`` enabled and first-row items classified, the front row of
        each shelf gets a bounded random offset (along-row, outward toward the
        aisle, and yaw about its own center) on top of the snapshot pose. The
        store layout has rows running along x and shelf fronts facing +/-y.
        """
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
        if (
            jitter is not None
            and jitter.enabled
            and self.first_row_indices is not None
            and len(self.first_row_indices) > 0
        ):
            idx = torch.as_tensor(self.first_row_indices, device=device, dtype=torch.long)
            face = torch.as_tensor(self.first_row_facing, device=device, dtype=torch.float32)
            rows = torch.as_tensor(self.first_row_row_ids, device=device, dtype=torch.long)
            num = idx.numel()
            along = (torch.rand(num, device=device) * 2.0 - 1.0) * jitter.along_row_max
            if jitter.row_shift_max > 0.0:
                row_shift = (
                    torch.rand(int(rows.max()) + 1, device=device) * 2.0 - 1.0
                ) * jitter.row_shift_max
                along = along + row_shift[rows]
            poses_wxyz[idx, 0] += along
            poses_wxyz[idx, 1] += torch.rand(num, device=device) * jitter.outward_max * face
            if jitter.yaw_max_deg > 0.0:
                yaw = (torch.rand(num, device=device) * 2.0 - 1.0) * math.radians(jitter.yaw_max_deg)
                zeros = torch.zeros_like(yaw)
                poses_wxyz[idx, 3:7] = math_utils.quat_mul(
                    math_utils.quat_from_euler_xyz(zeros, zeros, yaw), poses_wxyz[idx, 3:7]
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


def classify_first_row_products(
    stage: Usd.Stage,
    snapshot: StoreRigidBodySnapshot,
    *,
    unit_cluster_gap: float = 0.4,
    unit_depth_max: float = 0.6,
    back_to_back_max: float = 0.7,
    first_row_band: float = 0.05,
) -> int:
    """Mark aisle-facing first-row products on the snapshot; returns the count.

    Assumes the small_store_aij layout: shelf units cluster along y, rows of
    items run along x, and unit fronts face +/-y into the aisles. Per env,
    products are clustered by bbox-center y (splitting on gaps larger than
    ``unit_cluster_gap``). A cluster whose nearest neighbor cluster is within
    ``back_to_back_max`` faces away from it (back-to-back pair); otherwise it
    faces toward the nearest cluster (across the aisle). Within each (cluster,
    shelf-level) group the first row is every item whose bbox-center y is
    within ``first_row_band`` of the frontmost one.
    """
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    count = len(snapshot.prim_paths)
    center_y = np.empty(count, dtype=np.float64)
    bottom_z = np.empty(count, dtype=np.float64)
    env_keys: list[str] = []
    for i, path in enumerate(snapshot.prim_paths):
        rng = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
        low, high = rng.GetMin(), rng.GetMax()
        center_y[i] = 0.5 * (low[1] + high[1])
        bottom_z[i] = low[2]
        env_keys.append(path.rsplit("/dsynth_scene", 1)[0])

    indices: list[int] = []
    facing: list[float] = []
    row_ids: list[int] = []
    next_row = 0
    for env_key in dict.fromkeys(env_keys):
        env_idx = np.array([i for i, k in enumerate(env_keys) if k == env_key])
        order = env_idx[np.argsort(center_y[env_idx])]
        clusters: list[list[int]] = [[int(order[0])]]
        for i in order[1:]:
            if center_y[i] - center_y[clusters[-1][-1]] > unit_cluster_gap:
                clusters.append([])
            clusters[-1].append(int(i))
        # A back-to-back unit pair has near-touching products, so it arrives as a
        # single cluster about twice a unit's depth: split it at the midpoint.
        split_clusters: list[list[int]] = []
        for cluster in clusters:
            values = center_y[np.array(cluster)]
            if values.max() - values.min() > unit_depth_max:
                mid = 0.5 * (values.max() + values.min())
                split_clusters.append([i for i in cluster if center_y[i] < mid])
                split_clusters.append([i for i in cluster if center_y[i] >= mid])
            else:
                split_clusters.append(cluster)
        clusters = split_clusters
        centers = np.array([center_y[c].mean() for c in clusters])
        if len(clusters) < 2:
            continue  # single unit: no aisle information
        for k, cluster in enumerate(clusters):
            deltas = centers - centers[k]
            deltas[k] = np.inf
            nearest = int(np.argmin(np.abs(deltas)))
            toward = math.copysign(1.0, centers[nearest] - centers[k])
            face = -toward if abs(centers[nearest] - centers[k]) < back_to_back_max else toward
            cluster_arr = np.array(cluster)
            levels = np.round(bottom_z[cluster_arr], 1)
            for level in np.unique(levels):
                group = cluster_arr[levels == level]
                front = face * center_y[group]
                selected = group[front >= front.max() - first_row_band]
                indices.extend(int(i) for i in selected)
                facing.extend([face] * len(selected))
                row_ids.extend([next_row] * len(selected))
                next_row += 1

    snapshot.first_row_indices = np.array(indices, dtype=np.int64)
    snapshot.first_row_facing = np.array(facing, dtype=np.float32)
    snapshot.first_row_row_ids = np.array(row_ids, dtype=np.int64)
    return len(indices)


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

    # Record-time lean store: board labels (``rXX_sYY_bZ`` from the item prim
    # names) whose products stay in the scene; every other store item prim is
    # deactivated before PhysX parses the stage. The rigid-body snapshot, the
    # recorded store_products rows and the ``.store_prims.json`` sidecar then
    # cover only the kept boards; replay maps the rows back onto the fully
    # populated store by prim path. ``None`` keeps the full store.
    keep_product_boards: list[str] | None = None

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
        # InteractiveScene treats every non-None, non-keyword cfg field as an
        # asset cfg to spawn, so lift the plain-data board filter off while the
        # base class runs and put it back afterwards.
        keep_boards = getattr(cfg, "keep_product_boards", None)
        if keep_boards is not None:
            cfg.keep_product_boards = None
        try:
            super().__init__(cfg)
        finally:
            if keep_boards is not None:
                cfg.keep_product_boards = keep_boards
        fix_store_room_material_bindings(self)
        if keep_boards:
            kept, removed = deactivate_off_board_products(self.stage, self.env_prim_paths, keep_boards)
            print(
                f"[INFO] SberDemo lean store: kept {kept} products on boards "
                f"{sorted(keep_boards)}, deactivated {removed} others"
            )
        self._store_rigid_snapshot = capture_store_rigid_body_snapshot(self, dynamic_only=True)
        self.first_row_jitter = FirstRowJitterCfg()
        try:
            n_first_row = classify_first_row_products(self.stage, self._store_rigid_snapshot)
            print(f"[INFO] SberDemo first-row jitter: {n_first_row} products classified as first-row")
        except Exception as exc:
            self.first_row_jitter.enabled = False
            logger.warning("[SberDemo] First-row classification failed (%r); jitter disabled", exc)

    def reset_store_rigid_bodies(self, sim: SimulationContext) -> int:
        """Restore dynamic store products to their initial USD poses (first row jittered)."""
        count = self._store_rigid_snapshot.restore(sim, jitter=self.first_row_jitter)
        if count > 0:
            logger.info("[SberDemo] Reset %d store rigid bodies to initial poses", count)
        return count

    def capture_store_rigid_body_state(self, sim: SimulationContext) -> dict | None:
        """Live poses + velocities of dynamic store products (mid-episode checkpoint)."""
        return self._store_rigid_snapshot.capture_live_state(sim)

    def restore_store_rigid_body_state(self, sim: SimulationContext, state: dict | None, **kwargs) -> int:
        """Put dynamic store products back to a captured live state."""
        return self._store_rigid_snapshot.restore_live_state(sim, state, **kwargs)
