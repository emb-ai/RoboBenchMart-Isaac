"""Headset-only task-instruction panel for XR demo recording.

Thin wrapper around ``isaaclab.ui.xr_widgets.show_instruction`` — the same
camera-facing XR overlay Isaac Lab's own XR teleop uses (``XRSceneView`` UI
compositing, NOT a stage prim, so robot camera sensors and viewport render
products can never capture it; recorded RGB stays clean by construction).

The panel is anchored at an offset (right, up, forward) from the operator's
head pose, snapshotted at the moment ``refresh()`` is called with pitch/roll
dropped — "up" is always world-up, so the panel sits above eye height even
when the operator is looking down at the table. It then stays world-fixed
(comfortable in VR; a fully head-locked HUD judders).
Every headset-triggered event (recording start, reset, T-teleport, static-mode
entry) re-snapshots it back into view.

The XR camera prim only exists while an XR session is running, so ``refresh()``
is safe to call any time: before the session connects it just reports that it
is waiting and the next event retries.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

XR_CAMERA_PRIM_PATH = "/_xr/stage/xrCamera"
TARGET_WIDGET_PRIM_PATH = "/target_task_instruction"


class XrTargetHud:
    """Per-trajectory task text in the headset; ``show()`` picks, ``refresh()`` re-anchors.

    Args:
        texts: Display strings, one per selectable target.
        offset: (right, up, forward) placement in the yaw-levelled head frame,
            metres — up is world-up, forward is the horizontal heading.
        width: Maximum panel width in metres — text wraps to fit.
        font_size: Text height in metres.
    """

    def __init__(
        self,
        texts: list[str],
        *,
        offset: tuple[float, float, float] = (0.9, 0.45, 1.8),
        width: float = 1.6,
        font_size: float = 0.08,
    ) -> None:
        self.texts = list(texts)
        self._offset = tuple(float(v) for v in offset)
        self._max_width = float(width)
        self._font_size = float(font_size)
        self._current: int | None = None
        self._failed = False
        self._shown = False
        self._waiting_logged = False
        self._pending: int | None = None
        self._auto_retry_countdown = 0

    @property
    def current_item(self) -> str | None:
        return self.texts[self._current] if self._current is not None else None

    def show(self, index: int) -> str | None:
        """Make text ``index`` the panel content; returns it (None if HUD disabled)."""
        self._current = index
        self.refresh_soon()
        return None if self._failed else self.texts[index]

    def refresh_soon(self, delay_frames: int = 8) -> None:
        """Schedule a ``refresh()`` a few ``tick()`` calls from now.

        Anchor teleports move the operator, but the xrCamera prim only reflects
        the new head pose a couple of XR frames later — an immediate snapshot
        would pin the panel at the pre-teleport view.
        """
        if self._failed or self._current is None:
            return
        self._pending = max(1, int(delay_frames))

    def tick(self) -> None:
        """Call once per main-loop iteration; runs scheduled and auto refreshes.

        While the panel has text but is not on screen (script started before the
        XR session connected), keep retrying every few dozen iterations so it
        appears as soon as the session is up — no button press required.
        """
        if self._pending is not None:
            self._pending -= 1
            if self._pending <= 0:
                self._pending = None
                self.refresh()
            return
        if self._shown or self._failed or self._current is None:
            return
        self._auto_retry_countdown -= 1
        if self._auto_retry_countdown <= 0:
            self._auto_retry_countdown = 30
            self.refresh()

    def _reanchor_existing(self, text: str) -> bool:
        """Move the live widget's anchor prim to the current head pose.

        Returns False when the widget does not exist yet or shows different
        text — the caller then falls back to the full hide+show rebuild.
        """
        try:
            from isaaclab.ui.xr_widgets import instruction_widget

            entry = instruction_widget.camera_facing_widget_container.get(TARGET_WIDGET_PRIM_PATH)
            if entry is None or entry[1] != text:
                return False
            return self._level_anchor()
        except Exception:
            return False

    def _level_anchor(self) -> bool:
        """Set the anchor prim to the head's yaw-only pose (pitch/roll dropped).

        With the raw head frame, looking down at the table sends the offset's
        "up + forward" into the table and the panel disappears inside it.
        Levelling the frame keeps "up" as world-up, so the panel always sits
        above eye height regardless of where the operator is looking.
        """
        try:
            import omni.kit.commands
            from isaacsim.core.utils.stage import get_current_stage
            from pxr import Gf, Usd, UsdGeom

            stage = get_current_stage()
            cam = stage.GetPrimAtPath(XR_CAMERA_PRIM_PATH)
            anchor = stage.GetPrimAtPath(TARGET_WIDGET_PRIM_PATH)
            if not cam.IsValid() or not anchor.IsValid():
                return False
            cam_xf = UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            pos = cam_xf.ExtractTranslation()
            # Camera forward is -Z; flatten it onto the ground plane for yaw.
            fwd_row = cam_xf.GetRow(2)
            forward = Gf.Vec3d(-fwd_row[0], -fwd_row[1], 0.0)
            if forward.GetLength() < 1e-3:
                # Looking straight up/down — the screen-up axis (+Y) projects
                # onto the horizontal heading instead.
                up_row = cam_xf.GetRow(1)
                forward = Gf.Vec3d(up_row[0], up_row[1], 0.0)
            if forward.GetLength() < 1e-3:
                return False
            forward = forward.GetNormalized()
            right_axis = Gf.Cross(forward, Gf.Vec3d(0.0, 0.0, 1.0))
            xform = Gf.Matrix4d(1.0)
            xform.SetRow(0, Gf.Vec4d(right_axis[0], right_axis[1], right_axis[2], 0.0))
            xform.SetRow(1, Gf.Vec4d(0.0, 0.0, 1.0, 0.0))
            xform.SetRow(2, Gf.Vec4d(-forward[0], -forward[1], -forward[2], 0.0))
            xform.SetRow(3, Gf.Vec4d(pos[0], pos[1], pos[2], 1.0))
            omni.kit.commands.execute(
                "TransformPrim",
                path=TARGET_WIDGET_PRIM_PATH,
                new_transform_matrix=xform,
            )
            return True
        except Exception:
            return False

    def refresh(self) -> None:
        """(Re)display the panel at the current view; retries until XR is up."""
        if self._failed or self._current is None:
            return
        try:
            from isaaclab.ui.xr_widgets import hide_instruction, show_instruction
            from isaacsim.core.utils.stage import get_current_stage
            from pxr import Gf
        except Exception as exc:  # no XR UI stack (e.g. keyboard-only debug run)
            self._failed = True
            print(f"[target-hud] disabled — XR widget stack unavailable: {exc!r}", flush=True)
            return
        try:
            if not get_current_stage().GetPrimAtPath(XR_CAMERA_PRIM_PATH).IsValid():
                self._shown = False
                if not self._waiting_logged:
                    print(
                        "[target-hud] XR session not up yet (no xrCamera prim); "
                        "panel will appear at the next recording start / reset / teleport.",
                        flush=True,
                    )
                    self._waiting_logged = True
                return
            # Re-anchor in place whenever the widget already shows this text.
            # hide+show destroys and recreates the widget's render-to-texture
            # resources; doing that on every reset trips a Kit refcount bug
            # (gpu.foundation "Refcounting underflow while releasing
            # AssetTextureData" -> heap corruption -> abort). Only the anchor
            # prim (a textureless camera copy) needs to move.
            if self._reanchor_existing(self.texts[self._current]):
                self._shown = True
                self._waiting_logged = False
                return
            right, up, forward = self._offset
            # Text changed (or first show): drop + re-show so the panel
            # re-snapshots at the CURRENT head pose (show_instruction
            # early-returns when the text is unchanged).
            hide_instruction(TARGET_WIDGET_PRIM_PATH)
            show_instruction(
                self.texts[self._current],
                prim_path_source=XR_CAMERA_PRIM_PATH,
                translation=Gf.Vec3d(right, up, -forward),
                display_duration=None,
                max_width=self._max_width,
                min_width=0.6,
                font_size=self._font_size,
                target_prim_path=TARGET_WIDGET_PRIM_PATH,
            )
            # show_instruction snapshots the raw (possibly pitched) head pose;
            # immediately level it so the panel starts above eye height too.
            self._level_anchor()
            if not self._shown:
                print(f"[target-hud] panel shown: {self.texts[self._current]}", flush=True)
            self._shown = True
            self._waiting_logged = False
        except Exception as exc:
            self._failed = True
            print(f"[target-hud] disabled — widget creation failed: {exc!r}", flush=True)
