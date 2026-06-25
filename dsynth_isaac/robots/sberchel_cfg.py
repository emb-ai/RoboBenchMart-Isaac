"""Sberchel v3.0a locomanipulator robot configuration.

This reuses the robot definition that already lives in the ``green_challenge``
repository instead of duplicating it here. ``green_challenge`` stores the full
Isaac Lab ``ArticulationCfg`` as a serialized JSON ("interface config") and
rebuilds it through its ``ConfigManager``. The rebuilt object is exactly the
same type as the other robots in this repo (e.g. ``FETCH_CONFIG``), so it can be
dropped straight into ``self.scene.robot``.

The config spawns the robot from a MuJoCo ``.xml`` (converted to USD on first
load) and uses green_challenge's custom frictional actuators.
"""

import os
import sys

# --- Make the sibling ``green_challenge`` package importable -----------------
# It lives next to this repo (../green_challenge). The interface JSON references
# modules as ``green_challenge.assets.robot_v3_0a...``, so the *parent* of the
# green_challenge folder must be on sys.path. Override with GREEN_CHALLENGE_PATH.
# This file is <repo>/dsynth_isaac/robots/sberchel_cfg.py -> 3 dirnames to repo root.
_DEFAULT_GC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..", "green_challenge"))
_GREEN_CHALLENGE_ROOT = os.environ.get("GREEN_CHALLENGE_PATH", _DEFAULT_GC)
_parent = os.path.dirname(os.path.abspath(_GREEN_CHALLENGE_ROOT))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# NOTE: importing this enables the ``isaacsim.asset.importer.mjcf`` extension,
# so it must happen *after* the simulation app has launched. Because this module
# is only imported lazily by ``gym.make`` (via the env cfg entry point), that
# ordering is satisfied.
from green_challenge.assets.robot_v3_0a.utils.config_manager import ConfigManager

# Rebuild the full ArticulationCfg (MJCF spawn + frictional actuators) from the
# serialized "interface" config shipped in green_challenge.
_cm = ConfigManager.instance()
SBERCHEL_CONFIG = _cm.load_config("sberchel_v3_0_alpha_locomanipulation_full")
