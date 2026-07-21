"""Compatibility helpers for Pink IK / qpsolvers / DAQP version skew."""

from __future__ import annotations

import inspect
import logging

logger = logging.getLogger(__name__)


def ensure_pink_ik_solver_compat() -> None:
    """Ensure DAQP is compatible with qpsolvers 4.12+ warm-start API.

    qpsolvers>=4.12 passes ``primal_start`` into ``daqp.solve()``. Older daqp
    (0.7.x) rejects that keyword, Pink IK catches the exception, and arm joints
    never update (fingers still move via direct joint targets).
    """
    import daqp

    if "primal_start" in inspect.signature(daqp.solve).parameters:
        return

    if getattr(daqp, "_dsynth_warmstart_patched", False):
        return

    _patch_daqp_solve_signature()


def _patch_daqp_solve_signature() -> None:
    """Ignore unsupported ``primal_start`` when an old DAQP build is installed."""
    import daqp

    original_solve = daqp.solve

    def solve_with_optional_warmstart(*args, primal_start=None, **kwargs):
        return original_solve(*args, **kwargs)

    daqp.solve = solve_with_optional_warmstart
    daqp._dsynth_warmstart_patched = True
    logger.warning(
        "Patched legacy daqp.solve() to ignore primal_start. "
        "For best teleop performance run: pip install 'daqp>=0.8.5'"
    )
