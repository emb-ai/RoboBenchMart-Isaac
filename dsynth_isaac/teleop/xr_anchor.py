"""Runtime XR anchor teleport: snap the headset frame back onto the moving robot.

``XrCfg.anchor_pos``/``anchor_rot`` pin the XR anchor to a fixed world pose, so
walking the robot away (WASD locomotion) leaves the operator standing at the
spawn point. This module latches the robot pose at start, expresses the anchor
in the robot's yaw frame, and re-applies that offset at the robot's current pose
on demand — the operator ends up exactly where they stood relative to the robot
at episode start.
"""

from __future__ import annotations

import logging
import math

from dsynth_isaac.teleop.sberchel_head_follow import robot_root_state

logger = logging.getLogger(__name__)

XR_ANCHOR_PRIM_PATH = "/World/XRAnchor"


def quat_mul_wxyz(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Hamilton product of two wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def yaw_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    """Rotation of ``yaw`` radians about world Z as a wxyz quaternion."""
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def xr_display_active() -> bool:
    """Whether an XR session is actually rendering to a headset right now.

    Anchor rotation written before the session is live does not survive session
    start (position does), so callers should re-assert the anchor pose on the
    ``False -> True`` edge of this signal.
    """
    try:
        from omni.kit.xr.core import XRCore

        core = XRCore.get_singleton()
        if hasattr(core, "is_xr_display_enabled"):
            return bool(core.is_xr_display_enabled())
        return bool(core.is_xr_enabled())
    except Exception:
        return False


def anchor_prim_path_from_cfg(xr_cfg) -> str:
    """Anchor prim path ``OpenXRDevice`` creates for this XR config."""
    parent = getattr(xr_cfg, "anchor_prim_path", None)
    if parent is None:
        return XR_ANCHOR_PRIM_PATH
    return f"{parent.rstrip('/')}/XRAnchor"


class XrAnchorTeleporter:
    """Move the XR anchor prim so the operator keeps their start-time robot-relative pose.

    Args:
        xr_cfg: The env ``XrCfg`` whose ``anchor_pos``/``anchor_rot`` were used to
            place the anchor prim at startup.
        robot: Articulation whose root pose defines the reference frame. Read once
            at construction (call right after the initial ``env.reset()``).
        prim_path: Anchor prim created by ``OpenXRDevice``. Defaults to the path
            that device derives from ``xr_cfg``.
        fixed_height: Keep the anchor at its configured Z (floor level) instead of
            following the pelvis bob.
        ref_pose: Optional ``((x, y, z), yaw)`` robot pose the configured anchor
            pose was designed for (the nominal ``init_state`` pose). Use this when
            the spawn pose is randomized: latching against the robot's live pose
            would bake the random spawn offset into every later teleport.
    """

    def __init__(
        self,
        xr_cfg,
        robot,
        *,
        prim_path: str | None = None,
        fixed_height: bool = True,
        ref_pose: tuple[tuple[float, float, float], float] | None = None,
    ) -> None:
        self._prim_path = prim_path or anchor_prim_path_from_cfg(xr_cfg)
        self._fixed_height = fixed_height
        self._anchor_pos = tuple(float(v) for v in xr_cfg.anchor_pos)
        self._anchor_quat = tuple(float(v) for v in xr_cfg.anchor_rot)
        self._layer_identifier = self._resolve_layer_identifier()
        if ref_pose is not None:
            self.capture_reference_pose(*ref_pose)
        else:
            self.capture_reference(robot)

    def capture_reference(self, robot) -> None:
        """Latch the robot pose that the configured anchor pose belongs to."""
        pos, yaw = robot_root_state(robot)
        self.capture_reference_pose(pos, yaw)

    def capture_reference_pose(self, pos: tuple[float, float, float], yaw: float) -> None:
        """Latch an explicit robot pose that the configured anchor pose belongs to."""
        rx, ry, rz = (float(v) for v in pos)
        self._ref_yaw = float(yaw)
        self._ref_z = rz
        dx = self._anchor_pos[0] - rx
        dy = self._anchor_pos[1] - ry
        c, s = math.cos(self._ref_yaw), math.sin(self._ref_yaw)
        # Anchor offset expressed in the robot's yaw frame.
        self._offset_x = c * dx + s * dy
        self._offset_y = -s * dx + c * dy
        self._offset_z = self._anchor_pos[2] - rz

    def anchor_pose_for_robot(
        self, robot
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Anchor world pose reproducing the reference offset at the robot's pose now."""
        (rx, ry, rz), ryaw = robot_root_state(robot)
        c, s = math.cos(ryaw), math.sin(ryaw)
        pos = (
            rx + c * self._offset_x - s * self._offset_y,
            ry + s * self._offset_x + c * self._offset_y,
            self._anchor_pos[2] if self._fixed_height else rz + self._offset_z,
        )
        quat = quat_mul_wxyz(yaw_quat_wxyz(ryaw - self._ref_yaw), self._anchor_quat)
        return pos, quat

    def teleport_to_robot(self, robot) -> tuple[tuple[float, float, float], float] | None:
        """Write the anchor prim so the operator lands back on the robot.

        Returns ``((x, y, z), yaw)`` of the new anchor pose, or ``None`` if the
        anchor prim could not be written (e.g. running without XR).
        """
        pos, quat = self.anchor_pose_for_robot(robot)
        if not self._write_anchor(pos, quat):
            return None
        yaw = math.atan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]),
        )
        return pos, yaw

    def _resolve_layer_identifier(self) -> str | None:
        """USD layer holding the anchor prim; ``None`` falls back to a Fabric write."""
        try:
            from isaacsim.core.utils.stage import get_current_stage

            prim = get_current_stage().GetPrimAtPath(self._prim_path)
            prim_stack = prim.GetPrimStack() if prim is not None else None
            return prim_stack[0].layer.identifier if prim_stack else None
        except Exception:  # stage not ready / no XR anchor prim
            return None

    def _write_anchor(
        self,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float],
    ) -> bool:
        w, x, y, z = quat
        try:
            from omni.kit.xr.core import XRCore
            from pxr import Gf

            matrix = Gf.Matrix4d()
            matrix.SetRotateOnly(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
            matrix.SetTranslateOnly(Gf.Vec3d(*pos))
            XRCore.get_singleton().set_world_transform_matrix(
                self._prim_path, matrix, self._layer_identifier
            )
            return True
        except Exception as exc:
            logger.debug("XR anchor write via XRCore failed: %s", exc)
        try:
            from isaacsim.core.prims import SingleXFormPrim

            SingleXFormPrim(self._prim_path).set_world_pose(position=pos, orientation=quat)
            return True
        except Exception as exc:
            logger.warning("XR anchor teleport failed: %s", exc)
            return False
