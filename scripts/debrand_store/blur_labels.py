#!/usr/bin/env python3
"""Mirror-and-blur variant of the texture stage: degrade the ORIGINAL artwork.

Alternative to ``gen_labels.py``.  Where that paints new packaging from scratch,
this keeps the source atlases and degrades them: each UV panel is flipped
horizontally (so text reads backwards on the product) and Gaussian-blurred.

The mirror is applied **per UV panel**, not to the whole image.  A global flip
would move an atlas's artwork away from the texels its UVs sample -- the
Coca-Cola wrap occupies only the left half of its atlas, so a whole-image flip
puts the label where nothing reads it and shows bare background on the can.

Caveat, recorded here because it is the point of the exercise: this output is a
derivative of the original artwork, and at any blur strength that keeps the
scene looking like a shop, wordmarks and trade dress stay recognisable.
Measured on the source atlases, "Coca-Cola zero", "Red Bull" and "Tide" all
remain legible at a 2.5% blur radius.

Usage:
    python scripts/debrand_store/blur_labels.py conf/debrand/inventory.json \
        conf/debrand/rename_map.yaml --out-dir build/blur_textures --blur-frac 0.015
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from pxr import Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_labels import ROLES, MeshCache, panels_and_orientation  # noqa: E402


def degrade_panel(arr, rect, blur_frac, is_normal):
    """Flip one UV panel horizontally and blur it, in place."""
    x, y, w, h, _ = rect
    height, width = arr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    patch = arr[y0:y1, x0:x1]

    patch = patch[:, ::-1]
    if is_normal and patch.shape[2] >= 3:
        # Tangent-space X points the other way once the panel is mirrored;
        # leaving it alone would light the surface from the wrong side.
        patch = patch.copy()
        patch[..., 0] = 255 - patch[..., 0]

    kernel = int(max(patch.shape[0], patch.shape[1]) * blur_frac)
    kernel = max(3, kernel | 1)
    # Replicate at the edges so the black atlas background is not smeared into
    # the label border, which would show as a dark halo around the product.
    patch = cv2.GaussianBlur(patch, (kernel, kernel), 0, borderType=cv2.BORDER_REPLICATE)
    arr[y0:y1, x0:x1] = patch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inventory")
    ap.add_argument("rename_map")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--blur-frac", type=float, default=0.015,
                    help="Gaussian radius as a fraction of the panel's larger side.")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    inv = json.loads(Path(args.inventory).read_text())
    plan = yaml.safe_load(Path(args.rename_map).read_text())["products"]
    sub_usds = Path(inv["package"]) / "SubUSDs"
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
                rgb = np.asarray(original.convert("RGB"), dtype=np.uint8).copy()

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
            before = rgb.copy()
            if not rects:
                # No UV island resolved: degrade the whole image rather than
                # shipping it untouched.
                rects = [(0, 0, size[0], size[1], 1.0)]
            for rect in rects:
                degrade_panel(rgb, rect, args.blur_frac, role == "normal")

            # A panel of one flat colour is unchanged by mirroring and blurring.
            # Recorded so the audit can tell an expected no-op from a texture
            # that silently failed to be processed.
            changed = bool(np.any(before != rgb))
            uniform = all(
                len(np.unique(before[max(0, y):y + h, max(0, x):x + w].reshape(-1, 3), axis=0)) == 1
                for x, y, w, h, _ in rects)

            image = Image.fromarray(rgb)
            if mode == "L":
                image = image.convert("L")
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
                "method": "mirror+blur", "blur_frac": args.blur_frac,
                "changed": changed, "panels_uniform": uniform,
                "sha256": hashlib.sha256((out_dir / new_name).read_bytes()).hexdigest(),
            }
            written += 1

    (out_dir / "label_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"mirrored+blurred {written} textures for {len(targets)} products "
          f"({skipped} skipped) at blur_frac={args.blur_frac} -> {out_dir}")


if __name__ == "__main__":
    main()
