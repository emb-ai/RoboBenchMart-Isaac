"""Single place that resolves the ``green_challenge`` checkout.

The checkout lives at ``/home/krapukhin/projects/green-challenge`` (override with
the ``GREEN_CHALLENGE_PATH`` environment variable). Its directory name is
hyphenated, so putting the parent directory on ``sys.path`` -- what every script
used to do with the old ``../green_challenge`` sibling -- cannot import it:
``green-challenge`` is not a legal module name. Bind the directory to the
``green_challenge`` package name explicitly instead, which works for any
checkout location and name.
"""

from __future__ import annotations

import importlib.util
import os
import sys

DEFAULT_GREEN_CHALLENGE_PATH = "/home/krapukhin/projects/green_challenge"

_PACKAGE = "green_challenge"


def green_challenge_root() -> str:
    """Absolute path of the green_challenge checkout (assets, meshes, masks, policy)."""
    return os.path.abspath(
        os.environ.get("GREEN_CHALLENGE_PATH", DEFAULT_GREEN_CHALLENGE_PATH)
    )


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
