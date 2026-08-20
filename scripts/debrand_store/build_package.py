#!/usr/bin/env python3
"""Assemble the de-branded store package from ver2 + the generated labels.

Copies only what ``store.usd`` actually reaches, renames every branded wrapper
file and root prim to its generic id, repoints texture bindings at the generated
atlases, and re-points the shape-IP item prims at their bbox-matched
replacements.  Geometry, colliders, masses and all 2228 authored item poses are
carried across untouched -- masses live on the wrapper prim, so a swapped item
picks up its replacement's mass automatically.

Usage:
    python scripts/debrand_store/build_package.py \
        conf/debrand/inventory.json conf/debrand/rename_map.yaml \
        build/debrand_textures /home/soshin/repo/small_store_aij_ver5_debranded
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import yaml
from pxr import Sdf, UsdUtils

README = """# RoboBenchMart store package - de-branded

Self-contained store scene for Isaac Sim / Isaac Lab. Same room, same shelf
layout, same {items} pickable products in the same authored poses as the package it
was derived from - but every product's packaging design is synthetic, so nothing
in this scene depicts a real-world product or carries a real-world trademark.

See `DEBRAND_AUDIT.md` for the machine-checked evidence behind that statement.

## Use

- Quick look (any Isaac Sim/Lab install, from this folder):
  `python view_exported_store.py .` - opens the store with live physics,
  fly with right mouse button + WASD/QE.
- Or open `store.usd` in Isaac Sim and press Play (a `/physicsScene` is
  included for standalone use).
- For data collection, reference it into YOUR environment alongside
  your robot: `sim_utils.UsdFileCfg(usd_path=".../store.usd")` under an
  `AssetBaseCfg` (defaultPrim is `/World`; the bundled `/physicsScene` sits
  outside it, so your env's own physics scene governs).

## Physics (unchanged from the source package)

- **{items} products are dynamic rigid bodies** (UsdPhysics; convex
  decomposition colliders) - they can be picked. Shelves, walls, floor
  and the checkout are static colliders.
- Product physics is estimated from real size and packaging category: masses
  roughly 0.03-4.6 kg (drinks near liquid density, snacks light), and each
  product carries a bound physics material (glass slick ~0.55 static friction,
  plastic/cans ~0.7, cardboard grippy ~0.85, slight restitution on cans). Tune
  per-prim `physics:mass` / the bound material if a specific item matters.
- **Finger-based grasping / hand teleop**: bind a high-friction physics material
  to the fingertips (static = dynamic ~ 2.0) with `friction_combine_mode="max"`
  - grasps then hold regardless of each product's own friction (this benchmark's
  parallel gripper uses exactly that), and raise the hand articulation's solver
  position iteration count (8-16) for stable contacts.
- Item poses are authored at their resting positions; expect a millimeter-scale
  settle on the first Play.
- First load cooks the convex collision meshes - expect a slow first Play; it is
  cached afterwards.

## Materials

- Room and fixture materials are MDL (OmniPBR / NVIDIA Base / gltf) - they
  render in Isaac Sim / Omniverse only, not in generic USD viewers. These MDL
  modules resolve through Isaac's material search paths:
  `['OmniPBR.mdl', 'pbr.mdl']`
- Room material picks: floor=`Tiles015`, wall=`Plaster001`, ceiling=`Ceiling_Tiles`.

## How the packaging was replaced

Product meshes, UVs, colliders, masses and poses are carried over untouched.
Only the surface appearance changed:

{method_note}
- **{dropped} products were removed rather than repainted**, because their mesh
  silhouette was itself the recognisable element. Every one of their item prims
  was re-pointed at a bounding-box-matched product already on the shelves, so
  the item count and shelf density are unchanged. `DEBRAND_AUDIT.md` lists them.
- Product asset files, root prims, materials, shaders and GeomSubsets were all
  renamed to generic ids (`energy_can_02`, `mat_00`, `sub_01`).

Product ids are `<category>_<form>_<NN>`; categories are `cola, soda, energy,
juice, dairy, water, canned, sauce, condiment, haircare, skincare, deodorant,
shaving, detergent, cleaner, soap`. {products} distinct products are on the shelves.

Regenerate from the RoboBenchMart-Isaac-Private repo:

    python scripts/debrand_store/inventory.py <source-package> --out conf/debrand/inventory.json
    python scripts/debrand_store/plan_names.py conf/debrand/inventory.json \\
        --out conf/debrand/rename_map.yaml --blocklist conf/debrand/brand_blocklist.txt
    python scripts/debrand_store/gen_labels.py conf/debrand/inventory.json \\
        conf/debrand/rename_map.yaml --out-dir build/debrand_textures
    python scripts/debrand_store/build_package.py conf/debrand/inventory.json \\
        conf/debrand/rename_map.yaml build/debrand_textures <this-folder> --force
    python scripts/debrand_store/audit.py <this-folder> --against <source-package> \\
        --rename-map conf/debrand/rename_map.yaml \\
        --labels build/debrand_textures/label_manifest.json \\
        --inventory conf/debrand/inventory.json \\
        --blocklist conf/debrand/brand_blocklist.txt

## Known limitation

De-branding addresses trademark and trade-dress exposure in what this package
depicts. It does not establish the provenance or licence of the underlying 3D
meshes, for which no metadata exists in the source pipeline. If that matters for
your use, it needs to be answered separately.
"""


METHOD_NOTES = {
    "synthetic": """- **Albedo atlases are painted from scratch.** For each material the mesh's own
  `primvars:st` UVs are rasterised to find the label panels, and each panel is
  filled with a generated design: an invented wordmark, a category palette, a
  synthetic barcode and a nutrition block. Text orientation and handedness are
  derived from the UV-to-world Jacobian, so labels read correctly on the
  product. No pixel comes from the source artwork.
- **Shading maps (ORM, normal, occlusion, emissive) are constants.** They could
  not be carried over: some encoded label relief or were photographs of the
  original product. Roughness/metallic keep a single median value per channel so
  materials still read as metal, glass or plastic.""",
    "mirror+blur": """- **Textures are the source atlases, mirrored and blurred.** Each UV panel is
  flipped horizontally, so label text reads backwards on the product, and then
  Gaussian-blurred. The mirror is per panel rather than per image, because a
  whole-image flip would move an atlas's artwork away from the texels its UVs
  sample. Normal maps get their tangent-space X channel inverted to match the
  flip.
- **These textures are derivatives of the original product artwork.** They are
  not synthetic. At any blur strength that keeps the scene looking like a shop,
  wordmarks and trade dress stay recognisable - measured on the source atlases,
  several major wordmarks remain legible at a 2.5% blur radius. Treat this
  package as legibility reduction, not as de-branding, and see
  `DEBRAND_AUDIT.md` before relying on it.""",
}

TEXTURE_ROLES = ("albedo", "orm", "normal", "occlusion", "emissive")

# MDL modules name their maps as plain quoted paths.  UsdUtils cannot resolve
# inside an MDL, so these have to be followed by hand or the room materials
# ship without their textures.
MDL_TEXTURE = re.compile(r'"([^"]+\.(?:png|jpg|jpeg|exr|hdr|tga))"', re.I)

LIST_FIELDS = ("explicitItems", "addedItems", "prependedItems",
               "appendedItems", "deletedItems", "orderedItems")


def rewrite_listop(proxy, func):
    """Apply func to every item of a list-editable proxy, in place."""
    for field in LIST_FIELDS:
        items = list(getattr(proxy, field, []))
        if not items:
            continue
        rewritten = [func(i) for i in items]
        if rewritten != items:
            setattr(proxy, field, rewritten)


def plan_renames(layer, old_root: str, new_root: str, blocklist) -> dict:
    """Prim renames for one product layer, as {original path: new name}.

    The root prim is not the only place a brand hides: glTF material names come
    straight from the source file, so ``.../Looks/fanta_naranja_2l_001`` and its
    same-named shader and GeomSubset all carry the mark.  Materials, shaders and
    subsets are therefore renamed positionally rather than screened by keyword.
    """
    renames = {Sdf.Path(f"/{old_root}"): new_root}

    looks = layer.GetPrimAtPath(Sdf.Path(f"/{old_root}/geometry/Looks"))
    if looks is not None:
        for i, material in enumerate(looks.nameChildren):
            renames[material.path] = f"mat_{i:02d}"
            for child in material.nameChildren:
                if child.name == material.name:
                    renames[child.path] = "shader"

    geometry = layer.GetPrimAtPath(Sdf.Path(f"/{old_root}/geometry"))
    if geometry is not None:
        for mesh in geometry.nameChildren:
            subsets = [c for c in mesh.nameChildren if c.typeName == "GeomSubset"]
            for i, subset in enumerate(subsets):
                renames[subset.path] = f"sub_{i:02d}"

    # Safety net for anything the structural rules do not reach.
    def sweep(prim, index=[0]):
        if prim.path not in renames and any(b in prim.name.lower() for b in blocklist):
            renames[prim.path] = f"prim_{index[0]:03d}"
            index[0] += 1
        for child in prim.nameChildren:
            sweep(child)

    for prim in layer.rootPrims:
        sweep(prim)
    return renames


def remap_layer(src: Path, dst: Path, old_root: str, new_root: str, textures: dict,
                keep_materials=None, blocklist=()):
    """Copy a USD layer with every branded prim name and texture path replaced.

    Edits an in-memory copy and Exports a fresh crate: copying the file and
    saving in place leaves the original names behind as stale entries in the
    crate string table, where ``strings`` still finds them.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.OpenAsAnonymous(str(src))
    if layer is None:
        raise RuntimeError(f"cannot open {src}")

    if keep_materials is not None:
        looks = layer.GetPrimAtPath(Sdf.Path(f"/{old_root}/geometry/Looks"))
        if looks is not None:
            for name in [c.name for c in looks.nameChildren]:
                if name not in keep_materials:
                    del looks.nameChildren[name]

    renames = plan_renames(layer, old_root, new_root, blocklist)

    def remap_path(path):
        """Rebuild a path with every planned ancestor rename applied."""
        if path.isEmpty or not path.IsAbsolutePath():
            return path
        prim_path = path.GetPrimPath()
        parts, cursor = [], prim_path
        while cursor != Sdf.Path.absoluteRootPath and not cursor.isEmpty:
            parts.append(renames.get(cursor, cursor.name))
            cursor = cursor.GetParentPath()
        if not parts:
            return path
        rebuilt = Sdf.Path("/" + "/".join(reversed(parts)))
        return rebuilt.AppendProperty(path.name) if path.IsPropertyPath() else rebuilt

    def remap_reference(ref):
        return Sdf.Reference(ref.assetPath, remap_path(ref.primPath) if ref.primPath else ref.primPath,
                             ref.layerOffset, ref.customData)

    def remap_payload(payload):
        return Sdf.Payload(payload.assetPath,
                           remap_path(payload.primPath) if payload.primPath else payload.primPath,
                           payload.layerOffset)

    def walk(prim):
        rewrite_listop(prim.referenceList, remap_reference)
        rewrite_listop(prim.payloadList, remap_payload)
        for prop in prim.properties:
            if isinstance(prop, Sdf.RelationshipSpec):
                rewrite_listop(prop.targetPathList, remap_path)
            elif isinstance(prop, Sdf.AttributeSpec):
                rewrite_listop(prop.connectionPathList, remap_path)
                if prop.name == "inputs:texture" and isinstance(prop.default, Sdf.AssetPath):
                    name = Path(prop.default.path).name
                    if name in textures:
                        prop.default = Sdf.AssetPath(
                            f"./textures/instanceable_meshes/{textures[name]}")
        for child in prim.nameChildren:
            walk(child)

    for prim in layer.rootPrims:
        walk(prim)

    # Deepest first, so an ancestor path stays valid until its own turn.
    for path in sorted(renames, key=lambda p: -len(p.pathString)):
        spec = layer.GetPrimAtPath(path)
        if spec is not None:
            spec.name = renames[path]

    if layer.defaultPrim == old_root:
        layer.defaultPrim = new_root
    layer.Export(str(dst))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inventory")
    ap.add_argument("rename_map")
    ap.add_argument("label_dir", help="Output of gen_labels.py")
    ap.add_argument("out", help="Destination package directory")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing destination")
    ap.add_argument("--blocklist", default="conf/debrand/brand_blocklist.txt")
    args = ap.parse_args()

    inv = json.loads(Path(args.inventory).read_text())
    plan = yaml.safe_load(Path(args.rename_map).read_text())["products"]
    label_dir = Path(args.label_dir)
    labels = json.loads((label_dir / "label_manifest.json").read_text())
    methods = {rec.get("method", "synthetic") for rec in labels.values()}
    method = methods.pop() if len(methods) == 1 else "mixed"
    source = Path(inv["package"])
    out = Path(args.out)
    blocklist = tuple(w.strip().lower() for w in Path(args.blocklist).read_text().splitlines()
                      if w.strip() and not w.startswith("#"))

    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} exists; pass --force to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    dropped = {a for a, p in plan.items() if p.get("shape_ip")}
    replacement = {a: plan[a]["replacement"] for a in dropped}

    # Every product texture is regenerated -- albedo repainted, shading maps
    # flattened -- so the shipped-name map is just the label manifest.  Nothing
    # is copied out of ver2: its ORM and emissive maps carry brand artwork.
    texture_map = {Path(original).name: rec["out"] for original, rec in labels.items()}

    # Ship exactly what store.usd reaches, nothing more.
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(source / "store.usd"))
    if unresolved:
        print(f"WARNING {len(unresolved)} unresolved dependencies, e.g. {unresolved[:3]}")
    reachable = {Path(str(l.identifier)) for l in layers} | {Path(a) for a in assets}

    dropped_layers, dropped_textures = set(), set()
    for asset in dropped:
        rec = inv["assets"][asset]
        dropped_layers |= {source / rec["wrapper"], source / "SubUSDs" / rec["mesh_layer"]}
        for mat in rec["materials"]:
            for role in TEXTURE_ROLES:
                if mat.get(role):
                    dropped_textures.add((source / "SubUSDs" / mat[role].lstrip("./")).resolve())

    product_layers = set()
    for asset, rec in inv["assets"].items():
        if rec["is_fixture"]:
            continue
        product_layers |= {(source / rec["wrapper"]).resolve(),
                           (source / "SubUSDs" / rec["mesh_layer"]).resolve()}

    copied = renamed = 0
    # 1. Everything reachable that is not a product layer or a product texture.
    for path in sorted(reachable):
        path = path.resolve()
        if not path.is_file() or path.name == "store.usd":
            continue
        if path in product_layers or path in dropped_textures:
            continue
        if path.resolve() in {p.resolve() for p in dropped_layers}:
            continue
        if path.parent.name == "instanceable_meshes" and (
                path.name in texture_map or path in dropped_textures):
            # Product textures are regenerated under new names in step 2, and a
            # dropped asset's textures ship nowhere.  Fixture textures
            # (door_yellow, square_light_led_panel, the shelf's Image_*.png)
            # carry no brand and keep their names, so they fall through here.
            continue
        rel = path.relative_to(source)
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, out / rel)
        copied += 1

    # 2. Product wrappers and mesh layers, renamed and repointed.
    tex_out = out / "SubUSDs" / "textures" / "instanceable_meshes"
    tex_out.mkdir(parents=True, exist_ok=True)
    for asset, rec in sorted(inv["assets"].items()):
        if rec["is_fixture"] or asset in dropped:
            continue
        new_id = plan[asset]["new_id"]
        remap_layer(source / rec["wrapper"], out / "SubUSDs" / f"{new_id}.usd",
                    rec["root_prim"], new_id, texture_map, blocklist=blocklist)
        bound = {m["material"] for m in rec["materials"]}
        remap_layer(source / "SubUSDs" / rec["mesh_layer"],
                    out / "SubUSDs" / rec["mesh_layer"],
                    rec["root_prim"], new_id, texture_map, keep_materials=bound,
                    blocklist=blocklist)
        renamed += 2


    for rec in labels.values():
        shutil.copyfile(label_dir / rec["out"], tex_out / rec["out"])

    # 2b. Textures that only an MDL module names.
    mdl_copied = 0
    for mdl in sorted((out / "SubUSDs" / "materials").glob("*.mdl")):
        for match in MDL_TEXTURE.findall(mdl.read_text(errors="ignore")):
            src = (source / "SubUSDs" / "materials" / match).resolve()
            if not src.is_file():
                continue
            rel = src.relative_to(source.resolve())
            dest = out / rel
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            mdl_copied += 1

    # 3. store.usd: repoint every wrapper reference; swap the shape-IP items.
    store = Sdf.Layer.OpenAsAnonymous(str(source / "store.usd"))
    swaps, product_items = {}, 0

    def new_asset_path(ref):
        nonlocal product_items
        path = ref.assetPath
        if not path.startswith("./SubUSDs/"):
            return ref
        asset = path[len("./SubUSDs/"):-len(".usd")]
        if asset in dropped:
            target = replacement[asset]
            swaps[asset] = swaps.get(asset, 0) + 1
        elif asset in plan:
            target = asset
        else:
            return ref  # fixture, keeps its name
        product_items += 1
        new_id = plan[target]["new_id"]
        return Sdf.Reference(f"./SubUSDs/{new_id}.usd", ref.primPath,
                             ref.layerOffset, ref.customData)

    def walk_store(prim):
        rewrite_listop(prim.referenceList, new_asset_path)
        for child in prim.nameChildren:
            walk_store(child)

    for prim in store.rootPrims:
        walk_store(prim)
    store.Export(str(out / "store.usd"))

    # 4. The package documents itself, so a rebuild never loses its README.
    viewer = source / "view_exported_store.py"
    if viewer.exists():
        shutil.copyfile(viewer, out / "view_exported_store.py")
    (out / "README.md").write_text(README.format(
        items=product_items, dropped=len(dropped), products=len(plan) - len(dropped),
        method_note=METHOD_NOTES.get(method, METHOD_NOTES["synthetic"])))

    print(f"copied {copied} pass-through files (+{mdl_copied} MDL-referenced textures), "
          f"rewrote {renamed} product layers")
    for asset, count in sorted(swaps.items()):
        print(f"  swapped {count:3d} items: {asset} -> {plan[replacement[asset]]['new_id']}")
    print(f"package: {out}")


if __name__ == "__main__":
    main()
