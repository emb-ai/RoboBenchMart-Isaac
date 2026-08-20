#!/usr/bin/env python3
"""Inventory an exported RoboBenchMart store package for de-branding.

Walks ``store.usd`` for item references/masses and every ``SubUSDs/<asset>.usd``
wrapper for its instanceable mesh, GeomSubsets, materials and file-backed
textures, then writes ``inventory.json``.

The wrappers are ``gltf/pbr.mdl`` materials: the surface shader carries
``base_color_texture`` / ``metallic_roughness_texture`` / ``normal_texture``
inputs connected to ``gltf_texture_lookup`` shaders whose ``inputs:texture``
holds the PNG path.  Only textures reachable through a *bound* subset actually
render; the glTF import left unbound ``Material`` prims behind, and those
textures can simply be dropped from the de-branded package instead of repainted.

Usage:
    python scripts/debrand_store/inventory.py /home/soshin/repo/small_store_aij_ver2 \
        --out /home/soshin/repo/RoboBenchMart-Isaac-Private/conf/debrand/inventory.json
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdShade

# Fixture assets referenced from store.usd that are not sellable products.
FIXTURES = {
    "shelf_metal", "packing_table", "door_yellow", "square_light_led_panel",
    "SM_BarCodeScanner", "SM_PoiMonitor",
}

TEXTURE_ROLES = ("albedo", "orm", "normal", "occlusion", "emissive")

_MESH_REF = re.compile(r"@\./(instanceable_meshes(?:_\d+)?\.usd)@</([^/]+)/geometry>")


def scan_store(store_usd: Path) -> dict:
    """Per-asset item counts, masses and prim paths from store.usd."""
    stage = Usd.Stage.Open(str(store_usd))
    per_asset = defaultdict(lambda: {"instances": 0, "masses": [], "item_paths": []})
    for prim in stage.Traverse():
        refs = prim.GetMetadata("references")
        if not refs:
            continue
        for item in refs.GetAddedOrExplicitItems():
            path = item.assetPath
            if not path.startswith("./SubUSDs/"):
                continue
            asset = path[len("./SubUSDs/"):-len(".usd")]
            rec = per_asset[asset]
            rec["instances"] += 1
            rec["item_paths"].append(prim.GetPath().pathString)
            mass = prim.GetAttribute("physics:mass")
            if mass and mass.Get() is not None:
                rec["masses"].append(float(mass.Get()))
    out = {}
    for asset, rec in per_asset.items():
        out[asset] = {
            "instances": rec["instances"],
            "mass_median": statistics.median(rec["masses"]) if rec["masses"] else None,
            "item_paths": rec["item_paths"],
        }
    return out


def _texture_of(shader: UsdShade.Shader, input_name: str):
    """Follow a gltf_material input to the PNG behind its texture-lookup shader."""
    inp = shader.GetInput(input_name)
    if not inp:
        return None
    source = inp.GetConnectedSource()
    if not source:
        return None
    tex_shader = UsdShade.Shader(source[0].GetPrim())
    asset = tex_shader.GetInput("texture")
    value = asset.Get() if asset else None
    return value.path if value is not None else None


def scan_asset(sub_usds: Path, asset: str) -> dict | None:
    """Mesh layer, bbox and per-subset material/texture bindings for one wrapper."""
    wrapper = sub_usds / f"{asset}.usd"
    if not wrapper.exists():
        return None
    match = _MESH_REF.search(Sdf.Layer.FindOrOpen(str(wrapper)).ExportToString())
    if not match:
        return None
    mesh_layer, root_prim = match.group(1), match.group(2)

    stage = Usd.Stage.Open(str(sub_usds / mesh_layer))
    mesh_prim = next(
        (p for p in Usd.PrimRange.AllPrims(stage.GetPseudoRoot())
         if p.GetTypeName() == "Mesh"), None)
    if mesh_prim is None:
        return None
    mesh = UsdGeom.Mesh(mesh_prim)

    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    bbox = (points.max(axis=0) - points.min(axis=0)).tolist() if len(points) else None
    n_faces = len(mesh.GetFaceVertexCountsAttr().Get() or [])

    # One entry per bound material: a GeomSubset each, or the whole mesh when
    # the binding sits directly on the Mesh prim.
    subsets = [c for c in mesh_prim.GetAllChildren() if c.GetTypeName() == "GeomSubset"]
    targets = [(s.GetName(), s) for s in subsets] or [(None, mesh_prim)]

    materials = []
    for subset_name, prim in targets:
        bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if not bound:
            continue
        surface = bound.ComputeSurfaceSource("mdl")[0] or bound.ComputeSurfaceSource()[0]
        if not surface:
            continue
        indices = (UsdGeom.Subset(prim).GetIndicesAttr().Get() if subset_name else None)
        materials.append({
            "subset": subset_name,
            "material": bound.GetPrim().GetName(),
            "material_path": bound.GetPrim().GetPath().pathString,
            "n_faces": len(indices) if indices is not None else n_faces,
            "albedo": _texture_of(surface, "base_color_texture"),
            "orm": _texture_of(surface, "metallic_roughness_texture"),
            "normal": _texture_of(surface, "normal_texture"),
            "occlusion": _texture_of(surface, "occlusion_texture"),
            "emissive": _texture_of(surface, "emissive_texture"),
        })

    all_materials = []
    looks = stage.GetPrimAtPath(f"/{root_prim}/geometry/Looks")
    if looks:
        all_materials = [c.GetName() for c in looks.GetAllChildren()
                         if c.GetTypeName() == "Material"]

    return {
        "wrapper": f"SubUSDs/{asset}.usd",
        "all_materials": all_materials,
        "mesh_layer": mesh_layer,
        "root_prim": root_prim,
        "mesh_path": mesh_prim.GetPath().pathString,
        "n_faces": n_faces,
        "bbox": bbox,
        "materials": materials,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("package", help="Exported store package root (contains store.usd).")
    ap.add_argument("--out", required=True, help="Where to write inventory.json.")
    args = ap.parse_args()

    package = Path(args.package).resolve()
    sub_usds = package / "SubUSDs"

    store = scan_store(package / "store.usd")
    print(f"store.usd: {len(store)} referenced assets, "
          f"{sum(r['instances'] for r in store.values())} item prims")

    assets, failed = {}, []
    for asset in sorted(store):
        rec = scan_asset(sub_usds, asset)
        if rec is None:
            failed.append(asset)
            continue
        rec.update(instances=store[asset]["instances"],
                   mass_median=store[asset]["mass_median"],
                   is_fixture=asset in FIXTURES)
        assets[asset] = rec

    # Texture reachability: every PNG on disk vs those bound to a rendered material.
    tex_dir = sub_usds / "textures" / "instanceable_meshes"
    on_disk = {p.name for p in tex_dir.iterdir()} if tex_dir.is_dir() else set()
    reachable = defaultdict(lambda: {"roles": set(), "assets": set()})
    for asset, rec in assets.items():
        for mat in rec["materials"]:
            for role in TEXTURE_ROLES:
                path = mat[role]
                if not path:
                    continue
                entry = reachable[Path(path).name]
                entry["roles"].add(role)
                entry["assets"].add(asset)

    textures = {
        name: {"roles": sorted(v["roles"]), "assets": sorted(v["assets"])}
        for name, v in reachable.items()
    }
    orphaned = sorted(on_disk - set(textures))

    products = {a: r for a, r in assets.items() if not r["is_fixture"]}
    payload = {
        "package": str(package),
        "counts": {
            "referenced_assets": len(store),
            "products": len(products),
            "fixtures": len(assets) - len(products),
            "product_item_prims": sum(r["instances"] for a, r in assets.items()
                                      if not r["is_fixture"]),
            "textures_on_disk": len(on_disk),
            "textures_reachable": len(textures),
            "textures_orphaned": len(orphaned),
        },
        "assets": assets,
        "textures": textures,
        "orphaned_textures": orphaned,
        "unresolved_assets": failed,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))

    c = payload["counts"]
    print(f"products={c['products']} fixtures={c['fixtures']} "
          f"item_prims={c['product_item_prims']}")
    print(f"textures: {c['textures_on_disk']} on disk, {c['textures_reachable']} reachable, "
          f"{c['textures_orphaned']} orphaned")
    if failed:
        # Fixtures authored without the instanceable-mesh wrapper pattern. They
        # carry no product branding and ship through the build unchanged.
        print(f"note: {len(failed)} fixtures use a non-wrapper layout and are "
              f"passed through as-is: {failed}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
