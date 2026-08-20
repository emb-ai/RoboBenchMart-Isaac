#!/usr/bin/env python3
"""Headless RGB previews of an exported store package, for visual A/B checks.

The bundled ``view_exported_store.py`` is GUI-only; this renders fixed
viewpoints to PNG so a de-branded package can be compared against its source
from exactly the same camera.

Usage:
    python scripts/debrand_store/render_preview.py /path/to/package \
        --out-dir build/preview --tag ver5
"""

import argparse
import functools
from pathlib import Path

print = functools.partial(print, flush=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("package", help="Package folder containing store.usd")
parser.add_argument("--out-dir", required=True)
parser.add_argument("--tag", required=True, help="Filename prefix, e.g. ver2 / ver5")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--settle", type=int, default=90,
                    help="Render steps before capture (textures stream in asynchronously).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

# Same optics as the replay third-person camera: ~72 deg horizontal FOV.
FOCAL_LENGTH = 14.4
HORIZONTAL_APERTURE = 20.955


def shelf_viewpoints(stage):
    """An aisle overview plus two shelf close-ups that fill frame with products."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    world = cache.ComputeWorldBound(stage.GetPrimAtPath("/World")).ComputeAlignedRange()
    lo, hi = world.GetMin(), world.GetMax()
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2

    views = [("overview", (cx, lo[1] + 1.2, 2.0), (cx, cy, 1.2))]

    # Close-ups: aim at real item prims so the label art fills the frame.
    items = []
    for prim in stage.Traverse():
        if "visual_items" in prim.GetPath().pathString and prim.GetTypeName() == "Xform":
            attr = prim.GetAttribute("xformOp:translate")
            if attr and attr.Get() is not None:
                pose = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()).ExtractTranslation()
                items.append((prim.GetPath().pathString, np.array([pose[0], pose[1], pose[2]])))
    items.sort(key=lambda kv: kv[0])
    for index, label in ((len(items) // 4, "shelf_a"), (3 * len(items) // 4, "shelf_b")):
        if not items:
            break
        target = items[index][1]
        for sign in (-1.0, 1.0):
            eye = target + np.array([0.0, sign * 1.1, 0.25])
            if lo[1] < eye[1] < hi[1]:
                views.append((label, tuple(float(v) for v in eye),
                              tuple(float(v) for v in target)))
                break
    return views


def main():
    package = Path(args.package)
    root = package / "store.usd"
    if not root.exists():
        raise SystemExit(f"{root} not found")

    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(root.resolve())):
        raise SystemExit(f"could not open {root}")
    stage = ctx.get_stage()

    views = shelf_viewpoints(stage)
    print(f"[preview] {len(views)} viewpoints: {[v[0] for v in views]}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=0.01))
    camera = Camera(CameraCfg(
        prim_path="/World/preview_camera",
        update_period=0.0,
        height=args.height,
        width=args.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=FOCAL_LENGTH,
            horizontal_aperture=HORIZONTAL_APERTURE,
            clipping_range=(0.05, 1.0e6),
        ),
    ))
    sim.reset()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, eye, target in views:
        camera.set_world_poses_from_view(
            eyes=torch.tensor([eye], dtype=torch.float32, device=sim.device),
            targets=torch.tensor([target], dtype=torch.float32, device=sim.device))
        for _ in range(args.settle):
            sim.step()
            camera.update(sim.get_physics_dt())
        rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
        path = out_dir / f"{args.tag}_{name}.png"
        Image.fromarray(rgb[..., :3].astype(np.uint8)).save(path)
        print(f"[preview] wrote {path}")

    sim_utils.SimulationContext.clear_instance()
    simulation_app.close()


if __name__ == "__main__":
    main()
