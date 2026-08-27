"""Runtime XR anchor teleport: snap the headset frame back onto the moving robot.

``XrCfg.anchor_pos``/``anchor_rot`` pin the XR anchor to a fixed world pose, so
walking the robot away (WASD locomotion) leaves the operator standing at the
spawn point. This module latches the robot pose at start, expresses the anchor
in the robot's yaw frame, and re-applies that offset at the robot's current pose
on demand — the operator ends up exactly where they stood relative to the robot
at episode start.

Moving the *anchor* only rebases the play space: the operator's own head keeps
whatever yaw their physical body has, so after turning around in the room a
teleport drops them next to the robot facing somewhere else entirely. Pass a
``head_pose_fn`` to align the *view* instead — the anchor is then solved so the
headset itself lands on the target spot looking along the robot's forward axis.
"""

from __future__ import annotations

import logging
import math

from dsynth_isaac.teleop.sberchel_head_follow import (
    head_yaw_from_pose,
    robot_root_state,
    wrap_angle,
)

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


def end_xr_display(simulation_app=None, max_updates: int = 120) -> None:
    """End the headset session (disable the active XR profile) and wait for it.

    Kit's quit path wedges inside ``xrWaitFrame`` when the app shuts down while
    a CloudXR/OpenXR session is still presenting (window X / File->Exit hangs
    and spams ``XR_ERROR_VALIDATION_FAILURE``), so callers must end the session
    before ``env.close()``/``SimulationApp.close()``. Disabling is only a
    request; pass ``simulation_app`` so updates can be pumped until the session
    actually ends (bounded by ``max_updates``).
    """
    try:
        from omni.kit.xr.core import XRCore

        core = XRCore.get_singleton()
        if not core.is_xr_enabled():
            return
        core.request_disable_profile()
        for _ in range(max_updates):
            if simulation_app is not None:
                simulation_app.update()
            if not core.is_xr_enabled():
                break
    except Exception as exc:
        logger.warning("XR session stop failed: %s", exc)


def head_pose_tracked(head_pose) -> bool:
    """Whether a 7D headset pose is real tracking data.

    ``OpenXRDevice`` hands back a zero-position identity pose until the runtime
    reports a head transform. Aligning to that placeholder would face the
    operator along world +X rather than wherever they are actually looking.
    """
    if head_pose is None:
        return False
    return any(abs(float(v)) > 1e-6 for v in head_pose[:3])


def pose_from_matrix(matrix) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Split a ``Gf.Matrix4d`` into ``((x, y, z), (w, x, y, z))``."""
    translation = matrix.ExtractTranslation()
    quat = matrix.ExtractRotationQuat()
    imaginary = quat.GetImaginary()
    return (
        (float(translation[0]), float(translation[1]), float(translation[2])),
        (
            float(quat.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ),
    )


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
        head_pose_fn: Optional callable returning the headset's live 7D world pose
            (xyz + wxyz), or ``None`` while the session is not tracking. When it
            yields a tracked pose, teleports solve for the anchor that puts the
            *headset* on the target spot facing along the robot's forward axis,
            instead of only rebasing the play space. Pass ``None`` to keep the
            plain anchor-offset behaviour.
        view_yaw_offset: Extra heading (rad) added to the robot yaw the view is
            aligned to. A manual trim for headsets whose reported forward axis is
            off; ``0`` looks exactly where the robot's body +X points.
        forward_offset: Metres along the robot's forward axis to stand ahead of
            the latched spot. The configured anchor puts the operator on the
            robot's own root, i.e. inside its head — a small nudge clears that
            geometry. Costs the arms the same distance of forward reach, since
            wrist targets are absolute XR world poses.
    """

    def __init__(
        self,
        xr_cfg,
        robot,
        *,
        prim_path: str | None = None,
        fixed_height: bool = True,
        ref_pose: tuple[tuple[float, float, float], float] | None = None,
        head_pose_fn=None,
        view_yaw_offset: float = 0.0,
        forward_offset: float = 0.0,
    ) -> None:
        self._prim_path = prim_path or anchor_prim_path_from_cfg(xr_cfg)
        self._fixed_height = fixed_height
        self._anchor_pos = tuple(float(v) for v in xr_cfg.anchor_pos)
        self._anchor_quat = tuple(float(v) for v in xr_cfg.anchor_rot)
        self._layer_identifier = self._resolve_layer_identifier()
        self._head_pose_fn = head_pose_fn
        self._view_yaw_offset = float(view_yaw_offset)
        self._forward_offset = float(forward_offset)
        # Last pose this object wrote — the fallback when the anchor's live
        # transform cannot be read back.
        self._written_pos = self._anchor_pos
        self._written_quat = self._anchor_quat
        # Whether the most recent teleport used the headset pose. Callers use it
        # to retry once tracking comes up (the anchor is re-asserted the moment
        # the session goes live, when the head pose is often still untracked).
        self.last_teleport_aligned = False
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

    def target_xy_for_robot(self, robot) -> tuple[float, float, float, float]:
        """``(x, y, z, robot_yaw)`` the operator should occupy at the robot's pose now.

        ``forward_offset`` is added here rather than latched into the reference:
        it is a standing-position preference layered on the calibrated anchor
        offset, not part of what the anchor config was tuned to.
        """
        (rx, ry, rz), ryaw = robot_root_state(robot)
        c, s = math.cos(ryaw), math.sin(ryaw)
        offset_x = self._offset_x + self._forward_offset
        return (
            rx + c * offset_x - s * self._offset_y,
            ry + s * offset_x + c * self._offset_y,
            self._anchor_pos[2] if self._fixed_height else rz + self._offset_z,
            ryaw,
        )

    def anchor_pose_for_robot(
        self, robot, head_pose=None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Anchor world pose reproducing the reference offset at the robot's pose now.

        With a tracked ``head_pose`` the target spot is claimed by the headset
        rather than by the anchor origin: the play space is rigidly rotated by
        ``robot_yaw - head_yaw`` (so the operator ends up looking along the
        robot's forward axis) and then translated so the head lands on target.
        Physical drift away from the room origin no longer leaks into where the
        teleport drops the operator.
        """
        tx, ty, tz, ryaw = self.target_xy_for_robot(robot)
        if not head_pose_tracked(head_pose):
            quat = quat_mul_wxyz(yaw_quat_wxyz(ryaw - self._ref_yaw), self._anchor_quat)
            return (tx, ty, tz), quat

        (ax, ay, _az), aquat = self.current_anchor_pose()
        hx, hy = float(head_pose[0]), float(head_pose[1])
        delta_yaw = wrap_angle(ryaw + self._view_yaw_offset - head_yaw_from_pose(head_pose))
        c, s = math.cos(delta_yaw), math.sin(delta_yaw)
        # Rotating the play space about the world origin carries the anchor with
        # it; the translation that lands the head on target moves the anchor by
        # the same amount, i.e. the anchor keeps its (rotated) head-relative offset.
        dx, dy = ax - hx, ay - hy
        pos = (tx + c * dx - s * dy, ty + s * dx + c * dy, tz)
        return pos, quat_mul_wxyz(yaw_quat_wxyz(delta_yaw), aquat)

    def teleport_to_robot(
        self, robot, *, align_view: bool = True
    ) -> tuple[tuple[float, float, float], float] | None:
        """Write the anchor prim so the operator lands back on the robot.

        Unless ``align_view`` is off, the headset pose from ``head_pose_fn`` also
        turns the view to face along the robot's heading.

        Returns ``((x, y, z), yaw)`` of the new anchor pose, or ``None`` if the
        anchor prim could not be written (e.g. running without XR).
        """
        head_pose = None
        if align_view and self._head_pose_fn is not None:
            try:
                head_pose = self._head_pose_fn()
            except Exception as exc:  # tracking not up yet / device gone
                logger.debug("XR head pose unavailable for view alignment: %s", exc)
        aligned = head_pose_tracked(head_pose)
        pos, quat = self.anchor_pose_for_robot(robot, head_pose=head_pose)
        if not self._write_anchor(pos, quat):
            self.last_teleport_aligned = False
            return None
        self.last_teleport_aligned = aligned
        yaw = math.atan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]),
        )
        return pos, yaw

    def current_anchor_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """The anchor's live world pose, falling back to the last pose written here.

        Read back rather than cached because the session start drops the anchor
        *rotation* written before it (position survives), so the cache can be a
        half-turn out exactly when the first alignment runs.
        """
        try:
            from omni.kit.xr.core import XRCore

            core = XRCore.get_singleton()
            try:
                matrix = core.get_world_transform_matrix(self._prim_path)
            except TypeError:
                matrix = core.get_world_transform_matrix(self._prim_path, self._layer_identifier)
            if matrix is not None:
                return pose_from_matrix(matrix)
        except Exception as exc:
            logger.debug("XR anchor read via XRCore failed: %s", exc)
        try:
            import omni.usd
            from isaacsim.core.utils.stage import get_current_stage

            prim = get_current_stage().GetPrimAtPath(self._prim_path)
            if prim is not None and prim.IsValid():
                return pose_from_matrix(omni.usd.get_world_transform_matrix(prim))
        except Exception as exc:
            logger.debug("XR anchor read via USD failed: %s", exc)
        return self._written_pos, self._written_quat

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
            self._written_pos, self._written_quat = pos, quat
            return True
        except Exception as exc:
            logger.debug("XR anchor write via XRCore failed: %s", exc)
        try:
            from isaacsim.core.prims import SingleXFormPrim

            SingleXFormPrim(self._prim_path).set_world_pose(position=pos, orientation=quat)
            self._written_pos, self._written_quat = pos, quat
            return True
        except Exception as exc:
            logger.warning("XR anchor teleport failed: %s", exc)
            return False
