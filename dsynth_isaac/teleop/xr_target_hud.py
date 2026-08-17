"""Headset-only task-instruction panel for XR demo recording.

Thin wrapper around ``isaaclab.ui.xr_widgets.show_instruction`` — the same
camera-facing XR overlay Isaac Lab's own XR teleop uses (``XRSceneView`` UI
compositing, NOT a stage prim, so robot camera sensors and viewport render
products can never capture it; recorded RGB stays clean by construction).

The panel is anchored to the XR camera prim (``/_xr/stage/xrCamera``) at a
view-space offset (right, up, forward), snapshotted at the moment ``refresh()``
is called — it lands at the top-right of wherever the operator is looking and
then stays world-fixed (comfortable in VR; a fully head-locked HUD judders).
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
        offset: (right, up, forward) placement in the XR camera frame, metres.
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
            right, up, forward = self._offset
            # Drop + re-show so the panel re-snapshots at the CURRENT head pose
            # (show_instruction early-returns when the text is unchanged).
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
            if not self._shown:
                print(f"[target-hud] panel shown: {self.texts[self._current]}", flush=True)
            self._shown = True
            self._waiting_logged = False
        except Exception as exc:
            self._failed = True
            print(f"[target-hud] disabled — widget creation failed: {exc!r}", flush=True)
