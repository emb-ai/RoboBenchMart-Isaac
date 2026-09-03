"""Single place that resolves the ``green_challenge`` checkout.

Override the location with the ``GREEN_CHALLENGE_PATH`` environment variable.
A checkout directory may be named with a hyphen (``green-challenge``), so
putting its parent on ``sys.path`` -- what every script used to do with the old
``../green_challenge`` sibling -- cannot import it: ``green-challenge`` is not a
legal module name. Bind the directory to the ``green_challenge`` package name
explicitly instead, which works for any checkout location and name.

The default is a *materialized* checkout, not upstream's working copy. Upstream
(``/home/krapukhin/projects/green_challenge``, branch ``green-challenge-sim``)
now ships the robot package as ``assets/robot_v3_0a.tar.bz2`` with ``/assets/``
gitignored, so it has no ``assets/robot_v3_0a/`` tree, no ``assets/__init__.py``
and no ``assets/robot_selector.py`` -- ``green_challenge.assets.robot_v3_0a`` is
not importable from it and the settings/mask/MJCF paths below do not resolve.
It is also another user's read-only directory, so it cannot be unpacked in
place. ``scripts_bash/setup_green_challenge_sim.sh`` builds the unpacked copy
this default points at; re-run it whenever upstream changes.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

DEFAULT_GREEN_CHALLENGE_PATH = "/home/soshin/repo/green_challenge_sim"

# Upstream working copy that ``setup_green_challenge_sim.sh`` mirrors.
UPSTREAM_GREEN_CHALLENGE_PATH = "/home/krapukhin/projects/green_challenge"

_PACKAGE = "green_challenge"

# Robot asset package everything below is addressed inside.
ROBOT_PACKAGE = "robot_v3_0a"


def green_challenge_root() -> str:
    """Absolute path of the green_challenge checkout (assets, meshes, masks, policy)."""
    return os.path.abspath(
        os.environ.get("GREEN_CHALLENGE_PATH", DEFAULT_GREEN_CHALLENGE_PATH)
    )


def green_challenge_settings_dir() -> Path:
    """``<checkout>/assets/robot_v3_0a/dependencies/settings``.

    Holds ``camera_parameters.json``, ``camera_mask/``, ``joint_groups.json``
    and ``fingertip_collision.json`` -- everything ``ConfigManager.load_settings``
    reads.
    """
    return Path(green_challenge_root()) / "assets" / ROBOT_PACKAGE / "dependencies" / "settings"


def ensure_green_challenge_importable() -> str:
    """Make ``import green_challenge`` resolve to the checkout; return its root."""
    root = green_challenge_root()
    if _PACKAGE in sys.modules:
        return root
    init_py = os.path.join(root, "__init__.py")
    if not os.path.isfile(init_py):
        raise FileNotFoundError(
            f"green_challenge checkout not found at {root!r} "
            "(set GREEN_CHALLENGE_PATH to the checkout directory)."
        )
    spec = importlib.util.spec_from_file_location(
        _PACKAGE, init_py, submodule_search_locations=[root]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[_PACKAGE]
        raise
    return root
