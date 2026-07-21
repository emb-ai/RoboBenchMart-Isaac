"""Provide Sberchel hand URDFs / DexPilot configs for dex-retargeting.

Uses the hand-tuned assets under ``dsynth_isaac/assets/sberchel_xr`` (same files
as the working ``sberchel_teleop`` IsaacTeleop path). Auto-exporting finger
chains from the full robot URDF produced tip frames that drove DexPilot to
joint limits (permanently clenched hands).
"""

from __future__ import annotations

import shutil
from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets" / "sberchel_xr"

_LEFT_HAND_URDF = "sberchel_left_hand.urdf"
_RIGHT_HAND_URDF = "sberchel_right_hand.urdf"
_LEFT_YAML = "sberchel_hand_left_dexpilot.yml"
_RIGHT_YAML = "sberchel_hand_right_dexpilot.yml"


def _ensure_asset(name: str) -> Path:
    path = _ASSETS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing Sberchel XR hand asset: {path}")
    return path


def prepare_sberchel_dex_hand_urdfs(
    full_robot_urdf_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[str, str]:
    """Return left/right hand URDF paths (copied into ``cache_dir`` when given)."""
    left_src = _ensure_asset(_LEFT_HAND_URDF)
    right_src = _ensure_asset(_RIGHT_HAND_URDF)
    if cache_dir is None:
        return str(left_src), str(right_src)

    dex_dir = Path(cache_dir) / "dex_retargeting"
    dex_dir.mkdir(parents=True, exist_ok=True)
    left_dst = dex_dir / _LEFT_HAND_URDF
    right_dst = dex_dir / _RIGHT_HAND_URDF
    shutil.copy2(left_src, left_dst)
    shutil.copy2(right_src, right_dst)
    return str(left_dst), str(right_dst)


def prepare_sberchel_dex_retargeting_configs(
    cache_dir: str | Path,
    full_robot_urdf_path: str | Path | None = None,
) -> tuple[str, str]:
    """Copy DexPilot YAML configs into the cache and point them at local hand URDFs."""
    cache_dir = Path(cache_dir)
    left_urdf, right_urdf = prepare_sberchel_dex_hand_urdfs(full_robot_urdf_path, cache_dir)
    dex_dir = cache_dir / "dex_retargeting"
    left_cfg = dex_dir / _LEFT_YAML
    right_cfg = dex_dir / _RIGHT_YAML
    shutil.copy2(_ensure_asset(_LEFT_YAML), left_cfg)
    shutil.copy2(_ensure_asset(_RIGHT_YAML), right_cfg)

    # Rewrite relative urdf_path entries to the absolute cached hand URDFs.
    for cfg_path, urdf_path in ((left_cfg, left_urdf), (right_cfg, right_urdf)):
        text = cfg_path.read_text()
        lines = []
        for line in text.splitlines():
            if line.strip().startswith("urdf_path:"):
                lines.append(f"  urdf_path: {urdf_path}")
            else:
                lines.append(line)
        cfg_path.write_text("\n".join(lines) + "\n")

    return str(left_cfg), str(right_cfg)
