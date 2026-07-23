"""Bake Sberchel MJCF → static USD with materials already repaired.

Runs green_challenge ``spawn_from_mjcf`` (convert + equality/mimic patches +
instance explode + material rebind on the live prim), then exports that live
prim as a single flattened USD for ``SBERCHEL_CONVERTED_*`` configs.

Usage::

    # Closed kinematics (default)
    /home/soshin/miniconda3/envs/env_isaaclab/bin/python scripts/bake_sberchel_usd.py --headless

    # Open / serial kinematics (wrist_pitch, no crank/rod) for XR teleop
    /home/soshin/miniconda3/envs/env_isaaclab/bin/python scripts/bake_sberchel_usd.py --open --headless
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Bake Sberchel MJCF to a static USD package.")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Destination directory (default: dsynth_isaac/assets/robots/sberchel_converted).",
)
parser.add_argument(
    "--force",
    action="store_true",
    default=True,
    help="Force MJCF→USD reconversion before baking (default: True).",
)
parser.add_argument(
    "--open",
    action="store_true",
    default=False,
    help=(
        "Bake the open/serial MJCF (v3_0_alpha_locomanipulator_open.xml) instead of "
        "the closed model. Writes v3_0_alpha_locomanipulator_open.usd alongside the closed USD."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- green_challenge on path (MJCF spawn + ConfigManager) --------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_GC = (_REPO_ROOT.parent / "green_challenge").resolve()
_GC = Path(os.environ.get("GREEN_CHALLENGE_PATH", str(_DEFAULT_GC))).resolve()
_gc_parent = str(_GC.parent)
if _gc_parent not in sys.path:
    sys.path.insert(0, _gc_parent)

from green_challenge.assets.robot_v3_0a.interface.utils.mjcf import spawn_from_mjcf  # noqa: E402
from green_challenge.assets.robot_v3_0a.utils.config_manager import ConfigManager  # noqa: E402


_BAKE_PRIM_PATH = "/World/BakeSberchel"
_OUT_USD_NAME_CLOSED = "v3_0_alpha_locomanipulator.usd"
_OUT_USD_NAME_OPEN = "v3_0_alpha_locomanipulator_open.usd"


def _export_repaired_prim(source_prim_path: str, dest_usd: Path) -> None:
    """Export the live repaired robot as a single self-contained USD.

    Uses ``Usd.Stage.Flatten()`` (composes references/payloads), not
    ``FlattenLayerStack`` (root layer stack only — that produced a 17KB stub).
    """
    import omni.usd
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    src_prim = stage.GetPrimAtPath(source_prim_path)
    if not src_prim or not src_prim.IsValid():
        raise RuntimeError(f"Bake prim missing after spawn: {source_prim_path}")

    bound = unbound = 0
    for prim in Usd.PrimRange(src_prim):
        if prim.GetTypeName() != "Mesh":
            continue
        if "/collisions/" in prim.GetPath().pathString:
            continue
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if mat:
            bound += 1
        else:
            unbound += 1
    print(f"[bake_sberchel_usd] live visual meshes: bound={bound} unbound={unbound}")
    if bound == 0:
        raise RuntimeError("No material bindings on live bake prim; refusing to export white robot.")

    dest_usd = Path(dest_usd)
    dest_usd.parent.mkdir(parents=True, exist_ok=True)
    if dest_usd.exists():
        dest_usd.unlink()

    # Compose references/payloads/instance overrides into one layer.
    flat_layer = stage.Flatten(False)

    out_layer = Sdf.Layer.CreateNew(str(dest_usd))
    out_stage = Usd.Stage.Open(out_layer)
    UsdGeom.SetStageUpAxis(out_stage, UsdGeom.GetStageUpAxis(stage))
    UsdGeom.SetStageMetersPerUnit(out_stage, UsdGeom.GetStageMetersPerUnit(stage))

    target_prim_path = Sdf.Path("/Robot")
    Sdf.CreatePrimInLayer(out_layer, target_prim_path)
    Sdf.CopySpec(flat_layer, Sdf.Path(source_prim_path), out_layer, target_prim_path)
    out_layer.defaultPrim = target_prim_path.name
    omni.usd.resolve_paths(flat_layer.identifier, out_layer.identifier)
    out_layer.Save()

    size = dest_usd.stat().st_size
    print(f"[bake_sberchel_usd] exported flattened USD: {dest_usd} ({size} bytes)")
    if size < 1_000_000:
        raise RuntimeError(
            f"Exported USD is unexpectedly small ({size} bytes); geometry/materials likely missing."
        )


def main() -> None:
    cfg = ConfigManager.instance().load_config("sberchel_v3_0_alpha_locomanipulation_full")
    spawn_cfg = cfg.spawn
    if args_cli.force:
        spawn_cfg.force_usd_conversion = True

    closed_mjcf = Path(spawn_cfg.asset_path)
    if not closed_mjcf.is_file():
        raise FileNotFoundError(f"Closed Sberchel MJCF not found: {closed_mjcf}")

    if args_cli.open:
        mjcf_path = closed_mjcf.with_name("v3_0_alpha_locomanipulator_open.xml")
        out_name = _OUT_USD_NAME_OPEN
        if not mjcf_path.is_file():
            raise FileNotFoundError(f"Open Sberchel MJCF not found: {mjcf_path}")
        spawn_cfg.asset_path = str(mjcf_path)
    else:
        mjcf_path = closed_mjcf
        out_name = _OUT_USD_NAME_CLOSED

    spawn_cfg.usd_dir = str(mjcf_path.parent / "usd_cache")
    spawn_cfg.usd_file_name = f"{mjcf_path.stem}.usd"

    print(f"[bake_sberchel_usd] mode={'open' if args_cli.open else 'closed'}")
    print(f"[bake_sberchel_usd] MJCF: {mjcf_path}")
    print(f"[bake_sberchel_usd] usd_dir: {spawn_cfg.usd_dir}")
    print("[bake_sberchel_usd] Running spawn_from_mjcf (convert + patch + material repair)...")
    spawn_from_mjcf(prim_path=_BAKE_PRIM_PATH, cfg=spawn_cfg)

    dest_dir = Path(args_cli.output_dir) if args_cli.output_dir else (
        _REPO_ROOT / "dsynth_isaac" / "assets" / "robots" / "sberchel_converted"
    )
    # Only replace the target USD file — keep sibling closed/open assets.
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_entry = dest_dir / out_name

    _export_repaired_prim(_BAKE_PRIM_PATH, dest_entry)
    print(f"[bake_sberchel_usd] Wrote package to {dest_dir}")
    print(f"BAKE_USD_PATH={dest_entry.resolve()}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
