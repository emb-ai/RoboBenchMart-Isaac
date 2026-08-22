"""MuJoCo-faithful self-collision filtering for the Sberchel articulation.

green_challenge's interface cfg spawns the robot with
``enabled_self_collisions=True``, and the MJCF->USD importer applies
``UsdPhysics.FilteredPairsAPI`` to every link but leaves every target list
*empty* -- none of the MJCF's ``contype``/``conaffinity`` masks survive the
import. PhysX therefore collides every pair of links that is not a direct
parent/child, while MuJoCo -- the plant these meshes were authored against, and
the one green_challenge's policies were trained on -- collides far fewer.

Outside the hands the difference is nil: every body capsule carries the default
``contype=1 conaffinity=1`` and collides with everything either way. Inside the
hands it is decisive. The palm and all the proximal segments carry
``conaffinity=2`` with ``contype`` 1 or 5, so ``contype & conaffinity`` is zero
in both directions and MuJoCo never collides the thumb chain with the palm or
with the other proximal segments. Only the distal/intermediate segments
(``contype=2``, ``conaffinity=1``) collide with the palm and the thumb proximal
-- exactly the pairs that close on a grasped object.

Under PhysX those filtered pairs come back, and the open-hand pose sits inside
them: the reset pose drives ``*_thumb_proximal_yaw_joint`` to its lower limit,
the thumb assembly penetrates the palm, and the finger drive saturates against
a contact it can never clear. The joint then alternates one step in, one step
out, forever -- a limit cycle at exactly half the sim rate, recorded as both
thumbs trembling for the whole trajectory.

This module reads the masks back out of the MJCF and writes the pairs MuJoCo
excludes into the spawned USD as real ``FilteredPairsAPI`` targets, restoring
MuJoCo's contact set without touching ``enabled_self_collisions``.
"""

from __future__ import annotations

import functools
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

# MJCF geom defaults when the attribute is absent (MuJoCo XML reference).
_DEFAULT_CONTYPE = 1
_DEFAULT_CONAFFINITY = 1


def _body_collision_masks(mjcf_path: str | Path) -> tuple[dict[str, list[tuple[int, int]]], dict[str, str | None]]:
    """Per-body ``(contype, conaffinity)`` of every collidable geom, plus parents.

    Geoms with ``contype=0 conaffinity=0`` are MuJoCo's visual-only convention
    and are dropped. Bodies left with no collidable geom are dropped with them.
    """
    root = ET.parse(str(mjcf_path)).getroot()

    # A <default> block (or a geom naming a class) would move contype /
    # conaffinity out of the geom element and silently invalidate every mask
    # read below. This MJCF has neither; fail loudly if that ever changes.
    if root.find("default") is not None:
        raise RuntimeError(
            f"{mjcf_path} now has a <default> block: geom contype/conaffinity may be "
            "inherited from a class, which this parser does not resolve."
        )

    masks: dict[str, list[tuple[int, int]]] = {}
    parents: dict[str, str | None] = {}

    def walk(body: ET.Element, parent: str | None) -> None:
        name = body.get("name")
        if name is None:
            return
        parents[name] = parent
        geoms = []
        for geom in body.findall("geom"):
            if geom.get("class") is not None:
                raise RuntimeError(
                    f"{mjcf_path}: geom in body '{name}' uses class='{geom.get('class')}'; "
                    "contype/conaffinity may come from a default class."
                )
            contype = int(geom.get("contype", _DEFAULT_CONTYPE))
            conaffinity = int(geom.get("conaffinity", _DEFAULT_CONAFFINITY))
            if contype == 0 and conaffinity == 0:
                continue  # visual-only
            geoms.append((contype, conaffinity))
        if geoms:
            masks[name] = geoms
        for child in body.findall("body"):
            walk(child, name)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"{mjcf_path} has no <worldbody>")
    for body in worldbody.findall("body"):
        walk(body, None)
    return masks, parents


def mjcf_filtered_body_pairs(mjcf_path: str | Path) -> list[tuple[str, str]]:
    """Body pairs MuJoCo never collides but PhysX would.

    MuJoCo collides two geoms when ``(contype_a & conaffinity_b)`` or
    ``(contype_b & conaffinity_a)`` is non-zero. Two bodies are filtered when no
    geom pair between them passes that test. Direct parent/child pairs are left
    out: PhysX already excludes those inside an articulation.
    """
    masks, parents = _body_collision_masks(mjcf_path)
    names = list(masks)

    def collides(a: str, b: str) -> bool:
        return any(
            (ct_a & ca_b) or (ct_b & ca_a)
            for ct_a, ca_a in masks[a]
            for ct_b, ca_b in masks[b]
        )

    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if collides(a, b):
                continue
            if parents.get(a) == b or parents.get(b) == a:
                continue
            pairs.append((a, b))
    return pairs


def apply_mjcf_self_collision_filters(prim_path_expr: str, mjcf_path: str | Path) -> int:
    """Write the MJCF's filtered pairs onto every spawned robot under ``prim_path_expr``.

    Applied per matched root with the targets rewritten into that root, so it is
    correct for a cloned multi-env scene as well as a single robot. Returns the
    number of ``(root, pair)`` relationships written.
    """
    from isaaclab.sim.utils import find_matching_prims
    from pxr import Sdf, Usd, UsdPhysics

    pairs = mjcf_filtered_body_pairs(mjcf_path)
    roots = find_matching_prims(prim_path_expr)
    if not roots:
        raise RuntimeError(f"no prims match '{prim_path_expr}'; nothing to filter")

    written = 0
    for root in roots:
        stage = root.GetStage()
        # Body names are unique in the MJCF but the USD nests them under a
        # per-robot subtree, so resolve each name inside this root's subtree --
        # in a cloned scene the same name exists once per environment.
        by_name: dict[str, Sdf.Path] = {}
        for prim in Usd.PrimRange(root):
            by_name.setdefault(prim.GetName(), prim.GetPath())

        # Collect first, author once: FilteredPairsAPI targets are a single
        # relationship, so appending one pair at a time would rewrite it N times.
        targets: dict[Sdf.Path, list[Sdf.Path]] = {}
        for name_a, name_b in pairs:
            path_a, path_b = by_name.get(name_a), by_name.get(name_b)
            if path_a is None or path_b is None:
                continue
            targets.setdefault(path_a, []).append(path_b)

        for path, others in targets.items():
            prim = stage.GetPrimAtPath(path)
            api = (
                UsdPhysics.FilteredPairsAPI(prim)
                if prim.HasAPI(UsdPhysics.FilteredPairsAPI)
                else UsdPhysics.FilteredPairsAPI.Apply(prim)
            )
            rel = api.CreateFilteredPairsRel()
            existing = list(rel.GetTargets())
            rel.SetTargets(existing + [p for p in others if p not in existing])
            written += len(others)

    print(
        f"[sberchel_self_collision] filtered {len(pairs)} MJCF-excluded body pairs "
        f"on {len(roots)} robot(s) ({written} relationships)",
        flush=True,
    )
    return written


def with_mjcf_self_collision_filters(spawn_func: Callable, mjcf_path: str | Path) -> Callable:
    """Wrap a spawner so the MJCF's collision masks are restored after spawning.

    The wrapped ``spawn_from_mjcf`` is ``@clone``-decorated: it spawns the source
    prim and clones the rest before returning, so by the time this runs every
    robot in the scene exists and can be filtered.
    """

    @functools.wraps(spawn_func)
    def _spawn(prim_path, cfg, *args, **kwargs):
        prim = spawn_func(prim_path, cfg, *args, **kwargs)
        apply_mjcf_self_collision_filters(str(prim_path), mjcf_path)
        return prim

    return _spawn
