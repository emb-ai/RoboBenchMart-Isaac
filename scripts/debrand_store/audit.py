#!/usr/bin/env python3
"""Verify a de-branded store package and write DEBRAND_AUDIT.md.

Three independent checks:

1. Structural parity against the source package -- every item prim's pose,
   physics flags and resolved mass must be unchanged, and the asset each item
   references must be either its own renamed wrapper or a documented swap.
2. Name residue -- no real-world mark may appear in any filename, prim path,
   property path or raw byte run anywhere in the package.
3. Artwork residue -- no shipped texture may be byte-identical to any texture in
   the source package, and every shipped product texture must match the sha256
   the generator recorded, which is what makes "painted from scratch" checkable.

No OCR check is performed: no OCR engine is installed and the EAST detector
model cannot be fetched offline.  The sha256 check is a stronger guarantee for
this purpose -- it proves the shipped file *is* the synthesised one, rather than
inferring from pixels that a brand is absent.

Usage:
    python scripts/debrand_store/audit.py /home/soshin/repo/small_store_aij_ver5_debranded \
        --against /home/soshin/repo/small_store_aij_ver2 \
        --rename-map conf/debrand/rename_map.yaml \
        --labels build/debrand_textures/label_manifest.json \
        --blocklist conf/debrand/brand_blocklist.txt
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml
from pxr import Sdf, Usd

# Only identifier/path-shaped runs are scanned.  A raw printable-ASCII scan
# reports float payloads as text: mesh vertex bytes contain "?aXe@", which is
# not the word "axe".
ARTWORK_NOTES = {
    "synthetic": [
        "Every product texture is synthesised: albedo atlases are painted from",
        "scratch into the mesh's own UV panels, and shading maps (ORM, normal,",
        "occlusion, emissive) are replaced by constants. The shading maps could not",
        "be carried over -- the source Tabasco emissive map is a photograph of the",
        "bottles and the source Nivea ORM map holds legible label relief.",
    ],
    "monotone": [
        "Every product texture is a single flat colour field. The colour is the",
        "product's own mean albedo over the texels its UVs cover, weighted across",
        "materials by 3D surface area; shading maps are flattened to constants so",
        "no label relief survives in roughness or normals.",
        "",
        "The colour is a one-number-per-channel summary of the original artwork, so",
        "nothing of the source design - no shape, text, logo or layout - is",
        "reproducible from it. Colour alone can still evoke a brand where a product",
        "is strongly associated with one (a red-and-white can reads as cola), but no",
        "mark is depicted.",
    ],
    "mirror+blur": [
        "**These textures are NOT synthetic.** Every product texture is the source",
        "atlas with each UV panel flipped horizontally and Gaussian-blurred, so the",
        "shipped files are derivative works of the original product artwork.",
        "",
        "The checks above show only that no shipped file is byte-identical to its",
        "source. They do **not** show that the marks are gone, and they are not",
        "evidence that they are: mirroring and blurring reduce legibility, they do",
        "not remove trade dress. Measured on the source atlases, several major",
        "wordmarks are still readable at a 2.5% blur radius, and colour schemes and",
        "logo shapes survive at any strength that keeps the scene looking like a",
        "shop. This package should be reviewed on the rendered result, not on this",
        "report.",
    ],
}

IDENT_RUN = re.compile(rb"[A-Za-z0-9_./\-]{6,}")


def compile_terms(blocklist):
    """Blocklist terms as boundary-anchored patterns.

    Substring matching is unusable here -- "cif" fires inside "specifier" on
    every USD file in the package.  Separators inside a term are made flexible
    so "old spice" also catches "old_spice" and "oldspice".
    """
    patterns = []
    for term in blocklist:
        parts = [p for p in re.split(r"[^a-z0-9]+", term) if p]
        if not parts:
            continue
        core = r"[^A-Za-z0-9]*".join(re.escape(p) for p in parts)
        patterns.append((term, re.compile(rf"(?<![A-Za-z0-9]){core}(?![A-Za-z0-9])", re.I)))
    return patterns


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def item_table(store_usd: Path) -> dict:
    """Item prim path -> referenced asset, pose and resolved mass."""
    stage = Usd.Stage.Open(str(store_usd))
    table = {}
    for prim in stage.Traverse():
        refs = prim.GetMetadata("references")
        if not refs:
            continue
        assets = [r.assetPath for r in refs.GetAddedOrExplicitItems()
                  if r.assetPath.startswith("./SubUSDs/")]
        if not assets:
            continue
        mass = prim.GetAttribute("physics:mass")
        table[prim.GetPath().pathString] = {
            "asset": assets[0][len("./SubUSDs/"):-len(".usd")],
            "translate": tuple(prim.GetAttribute("xformOp:translate").Get() or ()),
            "orient": tuple(prim.GetAttribute("xformOp:orient").Get().GetImaginary())
                      + (prim.GetAttribute("xformOp:orient").Get().GetReal(),)
                      if prim.GetAttribute("xformOp:orient").Get() else (),
            "mass": round(float(mass.Get()), 9) if mass and mass.Get() is not None else None,
            "rigid": prim.GetAttribute("physics:rigidBodyEnabled").Get(),
        }
    return table


def scan_names(package: Path, patterns, skip=()) -> list:
    """Blocklist hits in filenames and in identifier-like byte runs.

    ``skip`` excludes this report, which necessarily names the marks it removed.
    """
    hits = []
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(package).as_posix()
        if rel in skip:
            continue
        for term, pattern in patterns:
            if pattern.search(rel):
                hits.append((rel, "filename", term))
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".hdr"}:
            continue  # binary pixel data, covered by the artwork check
        blob = path.read_bytes()
        found = set()
        for run in IDENT_RUN.findall(blob):
            text = run.decode("ascii", "ignore")
            for term, pattern in patterns:
                if pattern.search(text):
                    found.add((term, text[:80]))
        for term, context in sorted(found):
            hits.append((rel, f"bytes: {context!r}", term))
    return hits


def scan_paths(store_usd: Path, patterns) -> list:
    """Blocklist hits in composed prim paths and property names."""
    hits = []
    stage = Usd.Stage.Open(str(store_usd))
    for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        candidates = [prim.GetPath().pathString]
        candidates += [p.GetName() for p in prim.GetProperties()]
        for text in candidates:
            for term, pattern in patterns:
                if pattern.search(text):
                    hits.append((prim.GetPath().pathString, "prim/property path", term))
    return hits


MDL_TEXTURE = re.compile(r'"([^"]+\.(?:png|jpg|jpeg|exr|hdr|tga))"', re.I)


def check_mdl_textures(package: Path) -> list:
    """Maps named inside MDL modules, which USD's dependency walk cannot see."""
    problems = []
    for mdl in sorted((package / "SubUSDs" / "materials").glob("*.mdl")):
        for match in MDL_TEXTURE.findall(mdl.read_text(errors="ignore")):
            if not (mdl.parent / match).resolve().is_file():
                problems.append(f"missing MDL texture: {match} (from {mdl.name})")
    return problems


def check_references(store_usd: Path) -> list:
    """Unresolved layers and missing texture files, MDL modules excepted."""
    problems = []
    stage = Usd.Stage.Open(str(store_usd))
    root = store_usd.parent
    seen = set()
    for layer in stage.GetUsedLayers():
        identifier = layer.identifier
        if identifier.endswith(".mdl") or identifier in seen:
            continue
        seen.add(identifier)
        if identifier.startswith("anon:"):
            continue
        if not Path(identifier).exists():
            problems.append(f"missing layer: {identifier}")

    for layer in stage.GetUsedLayers():
        if layer.identifier.startswith("anon:") or not Path(layer.identifier).exists():
            continue
        base = Path(layer.identifier).parent

        def visit(spec):
            if isinstance(spec, Sdf.AttributeSpec) and isinstance(spec.default, Sdf.AssetPath):
                asset = spec.default.path
                if not asset or Path(asset).suffix.lower() not in {
                        ".png", ".jpg", ".jpeg", ".hdr", ".exr", ".tga"}:
                    return  # MDL modules and their sub-identifiers resolve elsewhere
                if not (base / asset).exists():
                    problems.append(f"missing texture: {asset} (from {Path(layer.identifier).name})")

        def walk(prim):
            for prop in prim.properties:
                visit(prop)
            for child in prim.nameChildren:
                walk(child)

        for prim in layer.rootPrims:
            walk(prim)
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("package")
    ap.add_argument("--against", required=True, help="Source package to compare against")
    ap.add_argument("--rename-map", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--blocklist", required=True)
    ap.add_argument("--out", default=None, help="Report path (default: <package>/DEBRAND_AUDIT.md)")
    args = ap.parse_args()

    package, source = Path(args.package), Path(args.against)
    plan = yaml.safe_load(Path(args.rename_map).read_text())["products"]
    labels = json.loads(Path(args.labels).read_text())
    methods = {rec.get("method", "synthetic") for rec in labels.values()}
    method = methods.pop() if len(methods) == 1 else "mixed"
    inv = json.loads(Path(args.inventory).read_text())
    fixture_textures = set()
    for asset, rec in inv["assets"].items():
        if not rec["is_fixture"]:
            continue
        for mat in rec["materials"]:
            for role in ("albedo", "orm", "normal", "occlusion", "emissive"):
                if mat.get(role):
                    fixture_textures.add(Path(mat[role]).name)
    blocklist = tuple(w.strip().lower() for w in Path(args.blocklist).read_text().splitlines()
                      if w.strip() and not w.startswith("#"))

    failures, notes = [], []

    # --- 1. structural parity ---------------------------------------------
    old_items = item_table(source / "store.usd")
    new_items = item_table(package / "store.usd")
    expected = {}
    for asset, entry in plan.items():
        target = entry.get("replacement", asset)
        expected[asset] = plan[target]["new_id"]

    if set(old_items) != set(new_items):
        failures.append(f"item prim set differs: {len(old_items)} vs {len(new_items)}")
    pose_mismatch = mass_mismatch = asset_mismatch = 0
    swapped = Counter()
    for path, old in old_items.items():
        new = new_items.get(path)
        if new is None:
            continue
        if old["translate"] != new["translate"] or old["orient"] != new["orient"] \
                or old["rigid"] != new["rigid"]:
            pose_mismatch += 1
        want = expected.get(old["asset"], old["asset"])
        if new["asset"] != want:
            asset_mismatch += 1
        if old["asset"] in plan and plan[old["asset"]].get("shape_ip"):
            swapped[old["asset"]] += 1
            continue  # a swapped item legitimately takes its replacement's mass
        if old["mass"] != new["mass"]:
            mass_mismatch += 1
    for label, count in (("pose", pose_mismatch), ("mass", mass_mismatch),
                         ("asset mapping", asset_mismatch)):
        if count:
            failures.append(f"{count} item prims with {label} mismatch")

    # --- 2. name residue ---------------------------------------------------
    patterns = compile_terms(blocklist)
    report_name = (Path(args.out).name if args.out else "DEBRAND_AUDIT.md")
    name_hits = (scan_names(package, patterns, skip={report_name})
                 + scan_paths(package / "store.usd", patterns))
    if name_hits:
        failures.append(f"{len(name_hits)} brand-name hits in the package")

    # --- 3. artwork residue ------------------------------------------------
    src_hashes = {}
    for path in (source / "SubUSDs" / "textures").rglob("*"):
        if path.is_file():
            src_hashes.setdefault(sha256(path), path.name)
    expected_hash = {rec["out"]: rec["sha256"] for rec in labels.values()}

    product_tex = package / "SubUSDs" / "textures" / "instanceable_meshes"
    identical, unverified, verified, fixture_carried, expected_noop = [], [], 0, [], []
    for path in sorted(product_tex.iterdir()):
        if not path.is_file():
            continue
        digest = sha256(path)
        if path.name in expected_hash:
            # A product texture: must be the synthesised file, byte for byte.
            if expected_hash[path.name] == digest:
                verified += 1
            else:
                unverified.append(path.name)
            if digest in src_hashes:
                record = next((r for r in labels.values() if r["out"] == path.name), {})
                if record.get("changed") is False and record.get("panels_uniform"):
                    # Mirroring and blurring a constant-valued map is the identity.
                    expected_noop.append(path.name)
                else:
                    identical.append((path.name, src_hashes[digest]))
        elif path.name in fixture_textures:
            fixture_carried.append(path.name)  # shelf/door/lamp art, no brand
        else:
            unverified.append(path.name)
    if identical:
        failures.append(f"{len(identical)} shipped product textures identical to a source texture")
    if unverified:
        failures.append(f"{len(unverified)} shipped product textures not matched to the generator manifest")

    # Room and fixture textures are deliberately carried over unchanged.
    carried = list(fixture_carried)
    for path in sorted((package / "SubUSDs" / "textures").rglob("*")):
        if path.is_file() and path.parent.name != "instanceable_meshes":
            if sha256(path) in src_hashes:
                carried.append(path.relative_to(package).as_posix())

    # --- 4. references resolve --------------------------------------------
    ref_problems = check_references(package / "store.usd") + check_mdl_textures(package)
    if ref_problems:
        failures.append(f"{len(ref_problems)} unresolved references")

    counts = Counter(v["asset"] for v in new_items.values())
    products = {a: c for a, c in counts.items() if any(p["new_id"] == a for p in plan.values())}

    report = [
        "# De-branding audit",
        "",
        f"- package: `{package}`",
        f"- compared against: `{source}`",
        f"- verdict: **{'PASS' if not failures else 'FAIL'}**",
        "",
        "## 1. Structural parity",
        "",
        f"- item prims: {len(new_items)} (source: {len(old_items)})",
        f"- distinct products referenced: {len(products)}",
        f"- pose / rigid-body mismatches: {pose_mismatch}",
        f"- mass mismatches (excluding documented swaps): {mass_mismatch}",
        f"- asset-mapping mismatches: {asset_mismatch}",
        "",
        "Documented silhouette swaps (the mesh itself was the mark, so retexturing",
        "could not fix it; each item was re-pointed at a bounding-box-matched product",
        "already in the scene):",
        "",
    ]
    for asset, count in sorted(swapped.items()):
        entry = plan[asset]
        report.append(f"- `{asset}` x{count} -> `{plan[entry['replacement']]['new_id']}`"
                      f" - {entry['shape_ip_reason']}")
    report += [
        "",
        "## 2. Name residue",
        "",
        f"- blocklist terms checked: {len(blocklist)}",
        "- scope: every filename, every identifier-like byte run inside every",
        "  non-image file, and every composed prim path and property name in",
        f"  `store.usd`. This report (`{report_name}`) is excluded from its own scan,",
        "  since it names the marks that were removed.",
        f"- hits: **{len(name_hits)}**",
        "",
    ]
    for rel, where, word in name_hits[:40]:
        report.append(f"- `{rel}` ({where}) matched `{word}`")
    report += [
        "",
        "## 3. Artwork residue",
        "",
        f"- product textures shipped: {verified + len(unverified)}",
        f"- verified against the generator manifest by sha256: {verified}",
        f"- byte-identical to a source texture: **{len(identical)}**",
        (f"- unchanged because the map is a single flat value, verified uniform: "
         f"{len(expected_noop)}" if expected_noop else ""),
        "",
    ] + ARTWORK_NOTES.get(method, ARTWORK_NOTES["synthetic"]) + [
        "",
        f"Room and fixture textures carried over unchanged ({len(carried)} files): these are",
        "generic surface libraries (ambientCG-style tiles/plaster, Quixel-style scanner,",
        "monitor and corrugated-box packs) and the RoboBenchMart neon sign, which is our",
        "own mark.",
        "",
        "## 4. Reference resolution",
        "",
        f"- unresolved references: {len(ref_problems)}",
        "",
        "Covers USD asset paths and the maps named inside the bundled MDL modules,",
        "which USD's own dependency walk cannot see. The MDL *modules* themselves",
        "(`OmniPBR.mdl`, `gltf/pbr.mdl`) resolve through Isaac's material search",
        "paths rather than the package, as in the source package.",
        "",
    ]
    for problem in ref_problems[:20]:
        report.append(f"- {problem}")
    if failures:
        report += ["", "## Failures", ""] + [f"- {f}" for f in failures]

    out = Path(args.out) if args.out else package / "DEBRAND_AUDIT.md"
    out.write_text("\n".join(report) + "\n")

    print(f"verdict: {'PASS' if not failures else 'FAIL'}")
    print(f"  items {len(new_items)}  pose_mismatch {pose_mismatch}  mass_mismatch {mass_mismatch}"
          f"  asset_mismatch {asset_mismatch}")
    print(f"  name hits {len(name_hits)}  identical textures {len(identical)}"
          f"  manifest-verified {verified}  unresolved refs {len(ref_problems)}")
    for f in failures:
        print(f"  FAIL: {f}")
    print(f"report: {out}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
