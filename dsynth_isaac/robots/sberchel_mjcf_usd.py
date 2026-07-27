"""Prepare a Sberchel MJCF USD template for Pink IK / URDF export.

``MjcfConverter`` alone produces a USD with duplicate articulation roots and
closed kinematic loops. ``spawn_from_mjcf`` patches those before sim spawn;
this module applies the articulation fixes needed for URDF export.

Equality ``connect`` constraints are intentionally **not** added here: they
close kinematic loops and make ``UsdToUrdf`` fail. Runtime spawn still adds
them inside ``spawn_from_mjcf``.
"""

from __future__ import annotations

import logging
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.sim.converters import MjcfConverter

logger = logging.getLogger(__name__)

# Bump when Pink IK URDF export logic changes so cached files are regenerated.
PINK_URDF_CACHE_VERSION = "v8"
# Bump when sim spawn settings change and the MJCF USD cache must be regenerated.
MJCF_USD_CACHE_VERSION = "upperbody_fix_root_v1"


def _find_link_prim_path(stage, link_name: str) -> str:
    """Return the body prim path for a named MJCF link (skip visuals/collisions/sites)."""
    matches: list[str] = []
    for prim in stage.Traverse():
        if prim.GetName() != link_name:
            continue
        path = prim.GetPath().pathString
        if any(token in path for token in ("/visuals/", "/collisions/", "/sites/")):
            continue
        matches.append(path)
    if not matches:
        raise ValueError(f"[sberchel_mjcf_usd] Could not find link prim '{link_name}' in USD stage")
    if len(matches) > 1:
        matches.sort(key=len)
    return matches[0]


def _remove_mjcf_equality_constraints_from_usd(stage, default_prim, original_mjcf: str) -> None:
    """Drop MJCF connect constraints so UsdToUrdf sees a kinematic tree."""
    from green_challenge.assets.robot_v3_0a.interface.utils.mjcf import get_mjcf_equality_constraints

    constraints = get_mjcf_equality_constraints(original_mjcf)
    joints_scope = stage.GetPrimAtPath(default_prim.GetPath().AppendChild("joints"))
    if not joints_scope or not joints_scope.IsValid():
        return
    for constraint_name in constraints:
        constraint_prim = joints_scope.GetChild(constraint_name)
        if constraint_prim and constraint_prim.IsValid():
            stage.RemovePrim(constraint_prim.GetPath())



def _collect_mjcf_root_hierarchy_skip_edges(usd_path: str | Path) -> list[str]:
    """Collect USD hierarchy edges from the MJCF ``root`` body that bypass joints.

    MjcfConverter also emits parent/child Xform edges from the floating ``root``
    body to every descendant link. ``UsdToUrdf.trim_joint_skipping_edges`` then
    prefers those edges over the real revolute joints (e.g. ``neck_yaw_joint``),
    which drops whole kinematic branches from the Pink IK URDF.
    """
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.asset.exporter.urdf")
    from nvidia.srl.from_usd.transform_graph import TransformGraph
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    graph = TransformGraph.init_from_stage(stage)

    # Graph node name for the MJCF ``root`` body (not the USD default prim).
    mjcf_root_node = None
    for name in graph.list_node_names():
        if name.endswith("_root") or name == "root":
            # Prefer the body under the articulation, e.g. tmp_..._root
            node = graph.get_node(name)
            path = getattr(node, "path", "") or ""
            if path.endswith("/root") or name.endswith("_root"):
                mjcf_root_node = name
                break
    if mjcf_root_node is None:
        logger.warning("[sberchel_mjcf_usd] Could not find MJCF root body node; skip-edge cleanup disabled")
        return []

    joint_connected_links: set[str] = set()
    for edge in graph.edges:
        left, _, right = edge.name.partition("<-")
        if not right:
            continue
        if left.endswith("_joint") or right.endswith("_joint"):
            if not left.endswith("_joint"):
                joint_connected_links.add(left)
            if not right.endswith("_joint"):
                joint_connected_links.add(right)

    root_prefix = f"{mjcf_root_node}<-"
    skip_edges = [
        edge.name
        for edge in graph.edges
        if edge.name.startswith(root_prefix) and edge.name[len(root_prefix) :] in joint_connected_links
    ]
    logger.info(
        "[sberchel_mjcf_usd] removing %d MJCF root hierarchy skip edges for UsdToUrdf (root=%s)",
        len(skip_edges),
        mjcf_root_node,
    )
    return skip_edges


def patch_mjcf_usd_for_urdf(usd_path: str, preprocessed_mjcf: str, original_mjcf: str) -> str:
    """Apply spawn_from_mjcf USD fixes without equality constraints (tree-safe for URDF)."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from green_challenge.assets.robot_v3_0a.interface.utils.mjcf import (
        _apply_physx_mimic_joints,
        _fix_duplicate_articulation_roots,
        get_mjcf_articulated_joints,
        get_mjcf_mimic_info,
    )

    mimic_info = get_mjcf_mimic_info(original_mjcf)

    stage = Usd.Stage.Open(usd_path)
    default_prim = stage.GetPrimAtPath("/").GetChildren()[0]
    if not default_prim or not default_prim.IsValid():
        raise ValueError(f"[sberchel_mjcf_usd] Invalid root prim in {usd_path}")

    stage.SetDefaultPrim(default_prim)
    _fix_duplicate_articulation_roots(default_prim)

    articulated_joints = set(get_mjcf_articulated_joints(preprocessed_mjcf))
    for prim in stage.Traverse():
        if "RevoluteJoint" in prim.GetTypeName() and prim.GetName() not in articulated_joints:
            prim.RemoveAPI(UsdPhysics.DriveAPI, "X")
            prim.RemoveAPI(UsdPhysics.DriveAPI, "Y")
            prim.RemoveAPI(UsdPhysics.DriveAPI, "Z")

    _apply_physx_mimic_joints(default_prim, mimic_info)
    _remove_mjcf_equality_constraints_from_usd(stage, default_prim, original_mjcf)

    world_body_prim = default_prim.GetChild("worldBody")
    if world_body_prim and world_body_prim.IsValid():
        world_body_prim.SetActive(False)
        world_body_prim.SetHidden(True)

    joints_scope = stage.GetPrimAtPath(default_prim.GetPath().AppendChild("joints"))
    root_joint_prim = None
    if joints_scope and joints_scope.IsValid():
        for child in joints_scope.GetChildren():
            if UsdPhysics.Joint(child) and child.GetName().startswith("rootJoint_"):
                root_joint_prim = child
                break

    if root_joint_prim and root_joint_prim.IsValid():
        joint = UsdPhysics.Joint(root_joint_prim)
        if joint:
            xfc = UsdGeom.XformCache()
            b0_targets = joint.GetBody0Rel().GetTargets()
            b1_targets = joint.GetBody1Rel().GetTargets()
            if b1_targets:
                b1_prim = stage.GetPrimAtPath(b1_targets[0])
                if b1_prim and b1_prim.IsValid():
                    t1_matrix = Gf.Matrix4d(xfc.GetLocalToWorldTransform(b1_prim))
                    pos1 = joint.GetLocalPos1Attr().Get() or Gf.Vec3f(0.0, 0.0, 0.0)
                    rot1 = joint.GetLocalRot1Attr().Get() or Gf.Quatf(1.0, 0.0, 0.0, 0.0)
                    tf1 = Gf.Transform()
                    q1i = rot1.GetImaginary()
                    q1d = Gf.Quatd(float(rot1.GetReal()), float(q1i[0]), float(q1i[1]), float(q1i[2]))
                    tf1.SetRotation(Gf.Rotation(q1d))
                    tf1.SetTranslation(Gf.Vec3d(pos1[0], pos1[1], pos1[2]))
                    a_matrix = (t1_matrix * tf1.GetMatrix().RemoveScaleShear()).RemoveScaleShear()
                    if b0_targets:
                        b0_prim = stage.GetPrimAtPath(b0_targets[0])
                        t0_matrix = (
                            Gf.Matrix4d(xfc.GetLocalToWorldTransform(b0_prim))
                            if b0_prim and b0_prim.IsValid()
                            else Gf.Matrix4d(1.0)
                        )
                        x0_new = t0_matrix.GetInverse() * a_matrix
                        p0 = Gf.Vec3f(x0_new.ExtractTranslation())
                        q0d = x0_new.ExtractRotation().GetQuat()
                    else:
                        p0 = Gf.Vec3f(a_matrix.ExtractTranslation())
                        q0d = a_matrix.ExtractRotation().GetQuat()
                    qi = q0d.GetImaginary()
                    q0 = Gf.Quatf(float(q0d.GetReal()), float(qi[0]), float(qi[1]), float(qi[2]))
                    joint.CreateLocalPos0Attr().Set(p0)
                    joint.CreateLocalRot0Attr().Set(q0)

    stage.Save()
    return usd_path


def resolve_sberchel_mjcf_usd(spawn_cfg) -> tuple[str, str]:
    """Convert Sberchel MJCF to a patched USD usable by Pink IK / URDF export."""
    from green_challenge.assets.robot_v3_0a.interface.utils.mjcf import preprocess_robot_menagerie_mjcf

    original_mjcf = str(Path(spawn_cfg.asset_path).resolve())
    usd_dir = spawn_cfg.usd_dir or str(Path(original_mjcf).parent / "usd_cache")
    usd_file_name = spawn_cfg.usd_file_name or f"{Path(original_mjcf).stem}.usd"

    spawn_cfg.usd_dir = usd_dir
    spawn_cfg.usd_file_name = usd_file_name

    preprocessed_mjcf = preprocess_robot_menagerie_mjcf(original_mjcf)
    resolved_usd = Path(usd_dir) / usd_file_name
    mjcf_cache_version_path = Path(usd_dir) / ".mjcf_spawn_cache_version"

    usd_path = getattr(spawn_cfg, "usd_path", None)
    if not (isinstance(usd_path, str) and usd_path and Path(usd_path).exists()):
        converter_cfg = spawn_cfg.replace(asset_path=preprocessed_mjcf)
        cache_is_current = (
            mjcf_cache_version_path.exists()
            and mjcf_cache_version_path.read_text().strip() == MJCF_USD_CACHE_VERSION
        )
        if (
            not cache_is_current
            or (resolved_usd.exists() and resolved_usd.stat().st_size < 4096)
        ):
            converter_cfg = converter_cfg.replace(force_usd_conversion=True)
        usd_path = MjcfConverter(converter_cfg).usd_path
        mjcf_cache_version_path.write_text(MJCF_USD_CACHE_VERSION)
    else:
        usd_path = str(resolved_usd)

    export_usd = resolved_usd.with_name(f"{resolved_usd.stem}_ik_export.usd")
    export_usd.unlink(missing_ok=True)
    shutil.copy2(usd_path, export_usd)
    patch_mjcf_usd_for_urdf(str(export_usd), preprocessed_mjcf, original_mjcf)
    return str(export_usd), preprocessed_mjcf


def convert_sberchel_usd_to_urdf(usd_path: str | Path, cache_dir: str | Path) -> tuple[str, str]:
    """Export a tree URDF from the patched Sberchel USD for Pink IK."""
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.asset.exporter.urdf")
    from nvidia.srl.from_usd.to_urdf import UsdToUrdf

    usd_path = Path(usd_path)
    cache_dir = Path(cache_dir)
    urdf_output_dir = cache_dir / "urdf"
    mesh_dir = cache_dir / "meshes"
    urdf_output_path = urdf_output_dir / f"{usd_path.stem}.urdf"
    urdf_output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    # Root at pelvis so legs + torso + arms stay in one tree. Hierarchy skip edges
    # from the MJCF ``root`` body must be removed first, otherwise UsdToUrdf drops
    # real joints such as ``neck_yaw_joint`` in favor of cosmetic Xform parents.
    urdf_root_path = _find_link_prim_path(stage, "pelvis_link")
    edge_names_to_remove = _collect_mjcf_root_hierarchy_skip_edges(usd_path)

    usd_to_urdf = UsdToUrdf.init_from_file(
        str(usd_path),
        root=urdf_root_path,
        edge_names_to_remove=edge_names_to_remove or None,
        log_level=logging.ERROR,
    )
    usd_to_urdf.save_to_file(
        urdf_output_path=str(urdf_output_path),
        visualize_collision_meshes=False,
        mesh_dir=str(mesh_dir),
        mesh_path_prefix="",
    )

    content = urdf_output_path.read_text().replace("inf", "0.")
    urdf_output_path.write_text(content)

    _normalize_sberchel_urdf_names(urdf_output_path)
    _sanitize_mjcf_export_urdf_for_pink(urdf_output_path)
    return str(urdf_output_path), str(mesh_dir)


def _normalize_sberchel_urdf_names(urdf_path: Path) -> None:
    """Strip USD export prefixes so Pink IK frame names match the sim articulation."""
    prefixes = (
        "tmp_v3_0_alpha_locomanipulator_open_root_",
        "tmp_v3_0_alpha_locomanipulator_root_",
        "joints_",
    )
    content = urdf_path.read_text()
    for prefix in prefixes:
        content = content.replace(f'name="{prefix}', 'name="')
        content = content.replace(f'link="{prefix}', 'link="')
    urdf_path.write_text(content)


def _sanitize_mjcf_export_urdf_for_pink(urdf_path: Path) -> None:
    """Patch MJCF→USD→URDF exports so Pinocchio can load them for Pink IK.

    The Isaac exporter often emits:
      - duplicate ``PhysicsFixedJoint`` names (e.g. joint and link both named
        ``left_hand_base_link``), which makes ``pin.getJointId`` ambiguous;
      - a ``floating_base`` joint absent from Isaac Lab articulation joint lists;
      - ``continuous`` joints (Pinocchio uses 2D cos/sin ``nq``, breaking the
        1:1 map against Isaac Lab joint positions);
      - ``±inf`` / huge position limits on passive rod / equality joints.

    Passive equality joints must stay as named revolute entries (with finite
    limits), not be collapsed to ``fixed``.
    """
    import math

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    changed = False

    link_names = {link.get("name") for link in root.findall("link") if link.get("name")}
    seen_joint_names: dict[str, int] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if not name:
            continue
        if name in link_names:
            suffix = seen_joint_names.get(name, 0)
            new_name = f"{name}_{suffix}" if suffix else f"{name}_fixed"
            while new_name in link_names or new_name in seen_joint_names:
                suffix += 1
                new_name = f"{name}_{suffix}"
            joint.set("name", new_name)
            name = new_name
            changed = True
        idx = seen_joint_names.get(name, 0)
        if idx > 0:
            joint.set("name", f"{name}_{idx}")
            changed = True
        seen_joint_names[name] = idx + 1

        # Pinocchio models continuous joints with nq=2 (cos/sin). Isaac Lab
        # joint_pos is always 1 value per joint, so convert to revolute.
        if joint.get("type") == "continuous":
            joint.set("type", "revolute")
            if joint.find("limit") is None:
                ET.SubElement(
                    joint,
                    "limit",
                    lower="-3.141593",
                    upper="3.141593",
                    effort="1000.0",
                    velocity="100.0",
                )
            changed = True

    def _finite_limit(raw: str | None, fallback: float) -> str:
        if raw is None:
            return f"{fallback:.6f}"
        token = raw.strip().lower()
        if token in {"inf", "+inf", "-inf", "nan"}:
            return f"{fallback:.6f}"
        try:
            value = float(raw)
        except ValueError:
            return f"{fallback:.6f}"
        if not math.isfinite(value) or abs(value) > 1.0e6:
            return f"{fallback:.6f}"
        return f"{value:.6f}"

    for limit in root.iter("limit"):
        lower = limit.get("lower")
        upper = limit.get("upper")
        new_lower = _finite_limit(lower, -3.141593)
        new_upper = _finite_limit(upper, 3.141593)
        if new_lower != lower:
            limit.set("lower", new_lower)
            changed = True
        if new_upper != upper:
            limit.set("upper", new_upper)
            changed = True
        for attr, fallback in (("effort", 1000.0), ("velocity", 100.0)):
            val = limit.get(attr)
            new_val = _finite_limit(val, fallback)
            if val is not None and new_val != val:
                limit.set(attr, new_val)
                changed = True

    if changed:
        tree.write(urdf_path, encoding="unicode", xml_declaration=True)

    content = urdf_path.read_text()
    content = content.replace(
        '<joint name="floating_base" type="floating">',
        '<joint name="floating_base" type="fixed">',
    )
    urdf_path.write_text(content)


def _strip_urdf_mesh_geometry_for_pink(urdf_path: Path, output_path: Path) -> Path:
    """Drop mesh geometry from URDF; Pink IK only needs link/joint kinematics."""
    content = urdf_path.read_text()
    content = re.sub(r"<collision>.*?</collision>\s*", "", content, flags=re.DOTALL)
    content = re.sub(r"<visual>.*?</visual>\s*", "", content, flags=re.DOTALL)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    return output_path


def _triangulate_obj_meshes(mesh_dir: Path) -> None:
    """Triangulate quad faces in exported OBJ meshes for Pinocchio compatibility."""
    for obj_path in mesh_dir.glob("*.obj"):
        triangulated_lines: list[str] = []
        for line in obj_path.read_text().splitlines():
            if line.startswith("f "):
                face_vertices = line.split()[1:]
                if len(face_vertices) == 4:
                    triangulated_lines.append(f"f {face_vertices[0]} {face_vertices[1]} {face_vertices[2]}")
                    triangulated_lines.append(f"f {face_vertices[0]} {face_vertices[2]} {face_vertices[3]}")
                    continue
            triangulated_lines.append(line)
        obj_path.write_text("\n".join(triangulated_lines) + "\n")


def resolve_sberchel_spawn_usd_path(spawn_cfg) -> str:
    """Resolve a patched Sberchel MJCF USD path for Pink IK / URDF export."""
    usd_path, _ = resolve_sberchel_mjcf_usd(spawn_cfg)
    return usd_path


def prepare_sberchel_pink_ik_urdf(
    usd_path: str | Path, force_conversion: bool = False
) -> tuple[str, str, str]:
    """Convert patched Sberchel USD to cached URDF/meshes usable by Pink IK."""
    usd_path = Path(usd_path)
    cache_dir = usd_path.parent / "pink_ik_cache"
    version_path = cache_dir / ".cache_version"
    raw_urdf_path = cache_dir / "urdf" / f"{usd_path.stem}.urdf"
    pink_urdf_path = cache_dir / "urdf" / f"{usd_path.stem}_pink.urdf"
    mesh_dir = cache_dir / "meshes"

    cache_is_current = version_path.exists() and version_path.read_text().strip() == PINK_URDF_CACHE_VERSION
    needs_export = (
        force_conversion
        or not cache_is_current
        or not raw_urdf_path.exists()
        or not mesh_dir.exists()
    )
    if needs_export:
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw_urdf_path, mesh_dir = convert_sberchel_usd_to_urdf(usd_path, cache_dir)
        raw_urdf_path = Path(raw_urdf_path)
        mesh_dir = Path(mesh_dir)
        version_path.write_text(PINK_URDF_CACHE_VERSION)
    else:
        raw_urdf_path = Path(raw_urdf_path)
        mesh_dir = Path(mesh_dir)

    # Idempotent; also repairs legacy caches that skipped triangulation on hit.
    _triangulate_obj_meshes(mesh_dir)
    _strip_urdf_mesh_geometry_for_pink(raw_urdf_path, pink_urdf_path)

    return str(pink_urdf_path), str(mesh_dir), str(raw_urdf_path)
