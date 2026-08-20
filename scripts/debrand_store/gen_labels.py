#!/usr/bin/env python3
"""Repaint every product albedo atlas with synthetic, category-appropriate packaging.

Nothing is derived from the original artwork.  For each material the mesh's
``primvars:st`` UVs are rasterised into a texel mask, the connected components of
that mask are the label panels, and each panel is painted from scratch: a
category palette, an invented wordmark, a synthetic barcode and a nutrition
block.  The original file is opened only twice, for facts that are not artwork:
its pixel dimensions and its alpha channel (a cut-out mask, not label design).

Panel text orientation is derived from the UV->world Jacobian: whichever of the
u/v axes maps onto world Z is "up" on the shelf, so the wordmark is rotated to
read horizontally on the physical product.

Usage:
    python scripts/debrand_store/gen_labels.py conf/debrand/inventory.json \
        conf/debrand/rename_map.yaml --out-dir build/debrand_textures
"""

import argparse
import colorsys
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pxr import Usd, UsdGeom

# USD `st` from these glTF imports is OpenGL-convention (origin bottom-left).
# Verified empirically: rasterising with v_up covers 100% of each original
# atlas's inked region, versus 0.6-47% for the v_down reading.
V_UP = True

PALETTES = {
    "cola":      dict(bg=(140, 20, 26),   band=(250, 245, 240), ink=(255, 255, 255), accent=(35, 32, 34)),
    "soda":      dict(bg=(0, 140, 90),    band=(245, 250, 240), ink=(255, 255, 255), accent=(250, 200, 40)),
    "energy":    dict(bg=(24, 28, 44),    band=(80, 220, 235),  ink=(240, 250, 255), accent=(180, 240, 60)),
    "juice":     dict(bg=(240, 150, 40),  band=(255, 246, 225), ink=(90, 45, 10),    accent=(200, 60, 40)),
    "dairy":     dict(bg=(240, 246, 252), band=(40, 90, 170),   ink=(30, 50, 90),    accent=(120, 180, 230)),
    "water":     dict(bg=(225, 240, 248), band=(30, 120, 180),  ink=(20, 70, 120),   accent=(150, 205, 235)),
    "canned":    dict(bg=(190, 150, 60),  band=(60, 80, 50),    ink=(250, 245, 230), accent=(150, 40, 35)),
    "sauce":     dict(bg=(120, 25, 25),   band=(240, 225, 200), ink=(250, 240, 230), accent=(210, 140, 40)),
    "condiment": dict(bg=(200, 60, 45),   band=(250, 245, 235), ink=(255, 255, 255), accent=(240, 190, 60)),
    "haircare":  dict(bg=(45, 130, 140),  band=(235, 248, 248), ink=(255, 255, 255), accent=(160, 215, 215)),
    "skincare":  dict(bg=(238, 228, 220), band=(180, 140, 120), ink=(80, 60, 52),    accent=(220, 190, 170)),
    "deodorant": dict(bg=(40, 48, 66),    band=(200, 210, 225), ink=(240, 244, 250), accent=(120, 150, 190)),
    "shaving":   dict(bg=(230, 232, 236), band=(40, 70, 120),   ink=(35, 45, 60),    accent=(200, 60, 50)),
    "detergent": dict(bg=(30, 70, 150),   band=(240, 246, 255), ink=(255, 255, 255), accent=(250, 200, 50)),
    "cleaner":   dict(bg=(70, 150, 80),   band=(245, 250, 245), ink=(255, 255, 255), accent=(250, 215, 60)),
    "soap":      dict(bg=(245, 240, 232), band=(150, 120, 160), ink=(70, 60, 75),    accent=(200, 175, 210)),
}

FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Black.ttf",
    "/usr/share/fonts/truetype/open-sans/OpenSans-Bold.ttf",
    "/usr/share/fonts/truetype/quicksand/Quicksand-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
SMALL_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"

VOLUMES = {
    "can": ["330 ML", "355 ML", "500 ML", "250 ML"],
    "bottle": ["500 ML", "1 L", "1.5 L", "750 ML"],
    "carton": ["1 L", "500 ML", "200 ML", "750 ML"],
    "tin": ["400 G", "415 G", "220 G", "340 G"],
    "jar": ["350 G", "500 G", "280 G"],
    "tube": ["50 ML", "75 ML", "100 ML"],
    "stick": ["50 ML", "75 ML"],
    "spray": ["400 ML", "500 ML", "300 ML"],
}
FOOD = {"cola", "soda", "energy", "juice", "dairy", "water", "canned", "sauce", "condiment"}

# Non-albedo maps cannot be passed through: the Tabasco emissive map is a
# photograph of the bottles, and the Nivea ORM map carries legible "POWER
# STRONG" relief.  Each is replaced by a constant.  For ORM/occlusion the
# constant is the median of the original inside its UV panels -- a single
# scalar per channel, so the material still reads as metal or plastic without
# carrying any of the original's expressive content.
FLAT_FILL = {"normal": (128, 128, 255), "emissive": (0, 0, 0)}
ROLES = ("albedo", "orm", "normal", "occlusion", "emissive")


def flatten_map(src, size, mode, role, rects):
    """A constant-valued stand-in for a shading map."""
    if role in FLAT_FILL:
        fill = FLAT_FILL[role]
    else:
        with Image.open(src) as handle:
            arr = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        selected = []
        for x, y, w, h, _ in rects[:8]:
            patch = arr[max(0, y):y + h, max(0, x):x + w]
            if patch.size:
                selected.append(patch.reshape(-1, 3))
        if selected:
            sample = np.concatenate(selected)
        else:
            flat = arr.reshape(-1, 3)
            lit = flat[flat.max(axis=1) > 8]
            sample = lit if len(lit) else flat
        fill = tuple(int(v) for v in np.median(sample, axis=0))
    image = Image.new("RGB", size, fill)
    return image.convert("L") if mode == "L" else image


def rng(seed_text: str, salt: str, modulo: int) -> int:
    d = hashlib.blake2b(f"{seed_text}:{salt}".encode(), digest_size=8).digest()
    return int.from_bytes(d, "little") % modulo


def contrast_ink(background):
    """Black or white, whichever reads on this background.

    The palettes cannot hardcode this: the wordmark sits on the *band*, and a
    hardcoded white ink rendered XANORA invisibly on a near-white cola band.
    """
    r, g, b = background[:3]
    return (18, 18, 22) if (0.2126 * r + 0.7152 * g + 0.0722 * b) > 140 else (250, 250, 252)


# How far a category's palette may drift in hue.  Fifteen energy drinks sharing
# one navy-and-cyan scheme read as a monochrome shelf; a milk carton that has
# drifted to orange reads as a mistake, so the staples move much less.
HUE_RANGE = {
    "energy": 70, "soda": 60, "juice": 45, "cleaner": 55, "detergent": 50,
    "haircare": 55, "skincare": 30, "soap": 45, "deodorant": 45, "sauce": 30,
    "condiment": 25, "canned": 35, "cola": 12, "dairy": 18, "water": 20,
    "shaving": 35,
}


def vary_palette(palette, seed_text, category):
    """Rotate a whole palette in hue by a per-asset amount, keeping it harmonious."""
    span = HUE_RANGE.get(category, 30)
    shift = (rng(seed_text, "hue", 2 * span + 1) - span) / 360.0
    sat = 0.85 + rng(seed_text, "sat", 41) / 100.0
    val = 0.88 + rng(seed_text, "val", 30) / 100.0
    out = {}
    for key, color in palette.items():
        h, s_, v = colorsys.rgb_to_hsv(*[c / 255.0 for c in color])
        h = (h + shift) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, min(1.0, s_ * sat), min(1.0, v * val))
        out[key] = tuple(int(round(c * 255)) for c in (r, g, b))
    return out


class MeshCache:
    """Triangulated points/UVs per mesh layer; the big layers are reused a lot."""

    def __init__(self, sub_usds: Path):
        self.sub_usds = sub_usds
        self._cache = {}

    def get(self, mesh_layer: str, mesh_path: str):
        key = (mesh_layer, mesh_path)
        if key in self._cache:
            return self._cache[key]
        stage = Usd.Stage.Open(str(self.sub_usds / mesh_layer))
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        fv_idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)

        primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar("st")
        uv = np.asarray(primvar.Get(), dtype=np.float64) if primvar else None
        if uv is not None:
            indices = primvar.GetIndices()
            if indices:
                uv = uv[np.asarray(indices, dtype=np.int64)]
            if primvar.GetInterpolation() == "vertex":
                uv = uv[fv_idx]  # lift to face-varying so corner lookup is uniform

        # Fan-triangulate, remembering which face each triangle came from so a
        # GeomSubset (which indexes faces) can select its own triangles.
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        if counts.size and (counts == 3).all():
            base = starts[:, None]
            tri_corners = base + np.array([0, 1, 2])
            tri_face = np.arange(counts.size, dtype=np.int64)
        else:
            corners, faces = [], []
            for face, (start, count) in enumerate(zip(starts, counts)):
                for k in range(1, count - 1):
                    corners.append((start, start + k, start + k + 1))
                    faces.append(face)
            tri_corners = np.asarray(corners, dtype=np.int64).reshape(-1, 3)
            tri_face = np.asarray(faces, dtype=np.int64)
        value = (points, fv_idx, np.asarray(tri_corners, dtype=np.int64),
                 np.asarray(tri_face, dtype=np.int64), uv)
        if len(self._cache) > 2:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value
        return value


def panels_and_orientation(mesh_data, face_ids, size):
    """UV texel mask -> label panels, plus the text orientation for this atlas.

    Returns ``(rects, rotate, mirrored)``.  ``mirrored`` matters: these atlases
    were authored mirrored (the source Coca-Cola wrap reads backwards as a flat
    image), so text painted the obvious way comes out reversed on the product.

    Writing image x as px = u*(W-1) and image y as py = (1-v)*(H-1), the label
    reads correctly from outside the surface when
    ``dot(cross(dP/du, -dP/dv), N_out) < 0``.  Since
    ``cross(dP/du, dP/dv) = cross(e1, e2) / det`` and ``cross(e1, e2)`` is the
    outward face normal under USD's default winding, that reduces to ``det > 0``.
    So an area-weighted negative ``det`` means the atlas is mirrored.
    """
    points, fv_idx, tri_corners, tri_face, uv = mesh_data
    width, height = size
    if uv is None or len(tri_corners) == 0:
        return [], 0, False

    tris = tri_corners
    if face_ids is not None:
        keep = np.isin(tri_face, face_ids)
        tris = tri_corners[keep]
    if len(tris) == 0:
        return [], 0, False

    corner_uv = uv[tris]                       # (T, 3, 2)
    corner_xyz = points[fv_idx[tris]]          # (T, 3, 3)

    # UV -> world Jacobian: whichever axis carries world Z is "up" on the shelf.
    duv1 = corner_uv[:, 1] - corner_uv[:, 0]
    duv2 = corner_uv[:, 2] - corner_uv[:, 0]
    dp1 = corner_xyz[:, 1] - corner_xyz[:, 0]
    dp2 = corner_xyz[:, 2] - corner_xyz[:, 0]
    det = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
    ok = np.abs(det) > 1e-12
    rotate, mirrored = 0, False
    if ok.any():
        weight = np.abs(det[ok])
        mirrored = float(np.average(np.sign(det[ok]), weights=weight)) < 0
    if ok.any():
        inv = 1.0 / det[ok]
        dpdu = (duv2[ok, 1, None] * dp1[ok] - duv1[ok, 1, None] * dp2[ok]) * inv[:, None]
        dpdv = (-duv2[ok, 0, None] * dp1[ok] + duv1[ok, 0, None] * dp2[ok]) * inv[:, None]
        up_u = np.average(np.abs(dpdu[:, 2]) / (np.linalg.norm(dpdu, axis=1) + 1e-12), weights=weight)
        up_v = np.average(np.abs(dpdv[:, 2]) / (np.linalg.norm(dpdv, axis=1) + 1e-12), weights=weight)
        rotate = 90 if up_u > up_v else 0

    # Rasterise at reduced resolution: the mask only drives panel layout.
    scale = min(1.0, 768.0 / max(width, height))
    mw, mh = max(2, int(width * scale)), max(2, int(height * scale))
    px = corner_uv[:, :, 0] * (mw - 1)
    py = (1.0 - corner_uv[:, :, 1]) * (mh - 1) if V_UP else corner_uv[:, :, 1] * (mh - 1)
    poly = np.stack([px, py], axis=2).astype(np.int32)
    mask = np.zeros((mh, mw), np.uint8)
    cv2.fillPoly(mask, list(poly), 1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    rects = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 0.0015 * mw * mh:
            continue
        rects.append((int(x / scale), int(y / scale), int(w / scale), int(h / scale), area / (mw * mh)))
    rects.sort(key=lambda r: -r[4])
    return rects, rotate, mirrored


def fit_font(text, box_w, box_h, paths, seed_text):
    path = paths[rng(seed_text, "font", len(paths))]
    size = max(8, int(box_h))
    while size > 8:
        font = ImageFont.truetype(path, size)
        left, top, right, bottom = font.getbbox(text)
        if right - left <= box_w and bottom - top <= box_h:
            return font
        size = int(size * 0.92)
    return ImageFont.truetype(path, 8)


def draw_barcode(draw, x, y, w, h, seed_text):
    draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255))
    pos, i = x + max(2, w // 24), 0
    while pos < x + w - max(2, w // 24):
        bar = 1 + rng(seed_text, f"bar{i}", 3)
        if i % 2 == 0:
            draw.rectangle([pos, y + h * 0.12, pos + bar, y + h * 0.78], fill=(15, 15, 15))
        pos += bar + 1
        i += 1
    digits = "".join(str(rng(seed_text, f"d{k}", 10)) for k in range(12))
    try:
        font = ImageFont.truetype(SMALL_FONT, max(6, int(h * 0.18)))
        draw.text((x + w / 2, y + h * 0.88), digits, fill=(15, 15, 15), font=font, anchor="mm")
    except Exception:
        pass


def draw_nutrition(draw, x, y, w, h, seed_text):
    draw.rectangle([x, y, x + w, y + h], fill=(252, 252, 250), outline=(30, 30, 30), width=max(1, h // 60))
    rows = 6
    for r in range(rows):
        ry = y + h * (0.18 + 0.13 * r)
        draw.line([x + w * 0.06, ry, x + w * 0.94, ry], fill=(60, 60, 60), width=max(1, h // 120))
        val = rng(seed_text, f"nut{r}", 90) + 1
        try:
            font = ImageFont.truetype(SMALL_FONT, max(6, int(h * 0.09)))
            draw.text((x + w * 0.90, ry - h * 0.055), f"{val}", fill=(40, 40, 40), font=font, anchor="rm")
        except Exception:
            pass


def paint_panel(canvas, rect, rotate, mirrored, palette, plan, seed_text, detailed):
    x, y, w, h, _ = rect
    if w < 8 or h < 8:
        return
    pw, ph = (h, w) if rotate == 90 else (w, h)
    if min(pw, ph) < 56:
        ImageDraw.Draw(canvas).rectangle([x, y, x + w, y + h], fill=palette["band"])
        return
    tile = Image.new("RGB", (pw, ph), palette["bg"])
    draw = ImageDraw.Draw(tile)

    band_h = int(ph * 0.44)
    band_y = int(ph * 0.28)
    draw.rectangle([0, band_y, pw, band_y + band_h], fill=palette["band"])
    draw.rectangle([0, band_y - max(1, ph // 90), pw, band_y], fill=palette["accent"])
    draw.rectangle([0, band_y + band_h, pw, band_y + band_h + max(1, ph // 90)], fill=palette["accent"])

    brand = plan["brand"]
    on_band = contrast_ink(palette["band"])
    on_bg = contrast_ink(palette["bg"])
    font = fit_font(brand, pw * 0.86, band_h * 0.62, FONTS, seed_text)
    draw.text((pw / 2, band_y + band_h * 0.42), brand, fill=on_band, font=font, anchor="mm")

    if detailed and ph > 120:
        sub = plan["display"].split(" ", 1)[1].upper()
        sub_font = fit_font(sub, pw * 0.7, band_h * 0.22, [SMALL_FONT], seed_text)
        draw.text((pw / 2, band_y + band_h * 0.80), sub, fill=on_band, font=sub_font, anchor="mm")

        vols = VOLUMES.get(plan["form"], ["500 ML"])
        vol = vols[rng(seed_text, "vol", len(vols))]
        vol_font = fit_font(vol, pw * 0.25, ph * 0.09, [SMALL_FONT], seed_text)
        draw.text((pw * 0.94, ph * 0.10), vol, fill=on_bg, font=vol_font, anchor="rm")

        bw, bh = int(pw * 0.26), int(ph * 0.15)
        draw_barcode(draw, int(pw * 0.05), int(ph * 0.80), bw, bh, seed_text)
        if plan["category"] in FOOD:
            nw, nh = int(pw * 0.24), int(ph * 0.19)
            draw_nutrition(draw, int(pw * 0.70), int(ph * 0.76), nw, nh, seed_text)

    if rotate == 90:
        tile = tile.rotate(-90, expand=True)
    if mirrored:
        # Flip within the panel rect, so the panel stays aligned with its UV island.
        tile = ImageOps.mirror(tile)
    canvas.paste(tile, (x, y))


def generate(size, mode, rects, rotate, mirrored, plan, seed_text):
    palette = vary_palette(PALETTES[plan["category"]], seed_text, plan["category"])
    canvas = Image.new("RGB", size, palette["bg"])
    if not rects:
        return canvas.convert(mode) if mode == "L" else canvas
    for i, rect in enumerate(rects[:4]):
        paint_panel(canvas, rect, rotate, mirrored, palette, plan, f"{seed_text}:{i}",
                    detailed=(i == 0))
    for rect in rects[4:]:
        x, y, w, h, _ = rect
        ImageDraw.Draw(canvas).rectangle([x, y, x + w, y + h], fill=palette["accent"])
    return canvas.convert("L") if mode == "L" else canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inventory")
    ap.add_argument("rename_map")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="Limit to these original asset names.")
    args = ap.parse_args()

    inv = json.loads(Path(args.inventory).read_text())
    plan = yaml.safe_load(Path(args.rename_map).read_text())["products"]
    package = Path(inv["package"])
    sub_usds = package / "SubUSDs"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = MeshCache(sub_usds)
    manifest, written, skipped = {}, 0, 0

    targets = [a for a in sorted(plan) if not plan[a].get("shape_ip")]
    if args.only:
        targets = [a for a in targets if a in set(args.only)]

    for asset in targets:
        rec = inv["assets"][asset]
        entry = plan[asset]
        mesh_data = cache.get(rec["mesh_layer"], rec["mesh_path"])

        # Group by texture file: several materials can share one, and a file
        # must be painted once from the union of their faces.
        by_texture = {}
        for mat in rec["materials"]:
            for role in ROLES:
                path = mat.get(role)
                if not path:
                    continue
                info = by_texture.setdefault(path, {"roles": set(), "mats": []})
                info["roles"].add(role)
                info["mats"].append(mat)

        for path, info in by_texture.items():
            src = sub_usds / path.lstrip("./")
            if not src.exists():
                skipped += 1
                continue
            with Image.open(src) as original:
                size, mode = original.size, original.mode
                alpha = original.getchannel("A") if mode in ("RGBA", "LA") else None

            face_ids, whole_mesh = [], False
            stage = Usd.Stage.Open(str(sub_usds / rec["mesh_layer"]))
            for mat in info["mats"]:
                if not mat["subset"]:
                    whole_mesh = True
                    break
                subset_prim = f"{rec['mesh_path']}/{mat['subset']}"
                indices = UsdGeom.Subset(stage.GetPrimAtPath(subset_prim)).GetIndicesAttr().Get()
                if indices is not None:
                    face_ids.append(np.asarray(indices, dtype=np.int64))
            faces = None if whole_mesh or not face_ids else np.unique(np.concatenate(face_ids))

            rects, rotate, mirrored = panels_and_orientation(mesh_data, faces, size)
            role = "albedo" if "albedo" in info["roles"] else sorted(info["roles"])[0]
            seed_text = f"{asset}:{info['mats'][0]['material']}"
            if role == "albedo":
                image = generate(size, mode, rects, rotate, mirrored, entry, seed_text)
            else:
                image = flatten_map(src, size, mode, role, rects)
            if alpha is not None:
                image = image.convert("RGBA")
                image.putalpha(alpha)

            suffix = Path(path).stem.split("_texture")[-1]
            new_name = f"{entry['new_id']}_texture{suffix}.png"
            image.save(out_dir / new_name)
            manifest[path] = {
                "asset": asset, "new_id": entry["new_id"], "out": new_name, "role": role,
                "roles": sorted(info["roles"]),
                "materials": [m["material"] for m in info["mats"]],
                "size": list(size), "mode": mode, "panels": len(rects),
                "rotate": rotate, "mirrored": mirrored,
                "sha256": hashlib.sha256((out_dir / new_name).read_bytes()).hexdigest(),
            }
            written += 1

    (out_dir / "label_manifest.json").write_text(json.dumps(manifest, indent=2))
    roles = {}
    for rec in manifest.values():
        roles[rec["role"]] = roles.get(rec["role"], 0) + 1
    print(f"wrote {written} textures for {len(targets)} products ({skipped} skipped): "
          + ", ".join(f"{v} {k}" for k, v in sorted(roles.items())) + f" -> {out_dir}")


if __name__ == "__main__":
    main()
