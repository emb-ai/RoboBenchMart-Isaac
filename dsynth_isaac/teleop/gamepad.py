"""PS5 (DualSense) gamepad input for teleop scripts, read straight from ``/dev/input/jsN``.

Why the raw joystick device instead of carb/pygame: OpenXR steals window focus, so
carb input (and anything else needing the Isaac window) drops events while the
headset is on — the same reason ``sberchel_kb_record_demos.py`` polls X11 globally
for WASD. ``/dev/input/js*`` is focus-independent, world-readable on this box, and
needs no third-party package (the kernel ``hid-playstation`` driver exposes the
DualSense there as soon as it is plugged in / paired).

Default DualSense mapping (kernel ``hid-playstation`` js layout, 8 axes / 13 buttons)::

    Left stick        forward/back + turn     (WASD)
    Right stick X     analog strafe
    Create            toggle locomotion       (C)
    L1                teleport XR view        (T)
    R1                restore checkpoint      (V / Left arrow)
    D-pad up          save episode + reset, new start pose      (R / Up arrow)
    D-pad down        discard episode + reset, SAME start pose  (X / Down arrow)
    D-pad right       discard episode + reset, new start pose   (Right arrow)
    D-pad left        restore checkpoint      (V / Left arrow)
    Options           start / stop recording  (XR START / STOP)
    D-pad             digital fwd/back + turn fallback (only with sticks disabled;
                      then the D-pad drives locomotion instead of save/discard/restore)

Run standalone to check the mapping of a connected pad without launching Isaac::

    python -m dsynth_isaac.teleop.gamepad          # live button/axis dump
"""

from __future__ import annotations

import fcntl
import glob
import os
import struct
import time
from collections.abc import Callable

# struct js_event { __u32 time; __s16 value; __u8 type; __u8 number; }
_JS_EVENT_FMT = "IhBB"
_JS_EVENT_SIZE = struct.calcsize(_JS_EVENT_FMT)
_JS_EVENT_BUTTON = 0x01
_JS_EVENT_AXIS = 0x02
_JS_EVENT_INIT = 0x80

_AXIS_MAX = 32767.0

# DualSense button indices (js API orders buttons by ascending evdev keycode).
BTN_CROSS = 0
BTN_CIRCLE = 1
BTN_TRIANGLE = 2
BTN_SQUARE = 3
BTN_L1 = 4
BTN_R1 = 5
BTN_L2 = 6
BTN_R2 = 7
BTN_CREATE = 8
BTN_OPTIONS = 9
BTN_PS = 10
BTN_L3 = 11
BTN_R3 = 12

# DualSense axis indices.
AX_LEFT_X = 0
AX_LEFT_Y = 1
AX_L2 = 2
AX_RIGHT_X = 3
AX_RIGHT_Y = 4
AX_R2 = 5
AX_HAT_X = 6
AX_HAT_Y = 7

BUTTON_NAMES = {
    BTN_CROSS: "CROSS",
    BTN_CIRCLE: "CIRCLE",
    BTN_TRIANGLE: "TRIANGLE",
    BTN_SQUARE: "SQUARE",
    BTN_L1: "L1",
    BTN_R1: "R1",
    BTN_L2: "L2",
    BTN_R2: "R2",
    BTN_CREATE: "CREATE",
    BTN_OPTIONS: "OPTIONS",
    BTN_PS: "PS",
    BTN_L3: "L3",
    BTN_R3: "R3",
}


def _ioc(direction: int, type_: int, nr: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (type_ << 8) | nr


def _jsiocgname(fd: int) -> str:
    buf = bytearray(128)
    fcntl.ioctl(fd, _ioc(2, ord("j"), 0x13, len(buf)), buf)
    return bytes(buf).split(b"\x00")[0].decode(errors="replace")


def _jsiocg_counts(fd: int) -> tuple[int, int]:
    axes = bytearray(1)
    buttons = bytearray(1)
    fcntl.ioctl(fd, _ioc(2, ord("j"), 0x11, 1), axes)  # JSIOCGAXES
    fcntl.ioctl(fd, _ioc(2, ord("j"), 0x12, 1), buttons)  # JSIOCGBUTTONS
    return axes[0], buttons[0]


def list_joysticks() -> list[tuple[str, str, int, int]]:
    """Return ``(path, name, num_axes, num_buttons)`` for readable ``/dev/input/js*``."""
    found: list[tuple[str, str, int, int]] = []
    for path in sorted(glob.glob("/dev/input/js*")):
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            continue
        try:
            found.append((path, _jsiocgname(fd), *_jsiocg_counts(fd)))
        except OSError:
            pass
        finally:
            os.close(fd)
    return found


def find_gamepad(preferred: str = "") -> str | None:
    """Pick a gamepad device path, preferring a DualSense over other joysticks.

    The DualSense also registers a "Motion Sensors" joystick node (accelerometer /
    gyro, zero buttons) — that one is never a control device, so it is skipped.
    """
    if preferred:
        return preferred if os.path.exists(preferred) else None
    candidates = [d for d in list_joysticks() if d[3] >= 8 and "motion sensor" not in d[1].lower()]
    if not candidates:
        return None
    for path, name, _, _ in candidates:
        if "dualsense" in name.lower() or "wireless controller" in name.lower():
            return path
    return candidates[0][0]


class LinuxJoystick:
    """Non-blocking reader for a Linux ``/dev/input/jsN`` device, with auto-reconnect.

    State is level-triggered: :meth:`poll` drains all pending events into
    :attr:`buttons` / :attr:`axes`, so callers see current stick and button state
    regardless of loop rate. Axes are normalized to ``[-1, 1]``.
    """

    _RECONNECT_PERIOD_S = 2.0

    def __init__(self, path: str, *, on_event: Callable[[str, int, float], None] | None = None) -> None:
        self.path = path
        self.name = ""
        self.buttons: dict[int, bool] = {}
        self.axes: dict[int, float] = {}
        self.connected = False
        self._fd: int | None = None
        self._on_event = on_event
        self._last_open_attempt = 0.0
        self._open()

    def _open(self) -> bool:
        self._last_open_attempt = time.time()
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return False
        self._fd = fd
        self.connected = True
        try:
            self.name = _jsiocgname(fd)
        except OSError:
            self.name = "unknown"
        return True

    def _drop(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
        self._fd = None
        self.connected = False
        self.buttons.clear()
        self.axes.clear()

    def poll(self) -> bool:
        """Drain pending events. Returns True while the device is connected."""
        if self._fd is None:
            if time.time() - self._last_open_attempt >= self._RECONNECT_PERIOD_S:
                if self._open():
                    print(f"[PAD] reconnected: {self.name} ({self.path})", flush=True)
            return self.connected
        while True:
            try:
                data = os.read(self._fd, _JS_EVENT_SIZE)
            except BlockingIOError:
                break
            except OSError:  # unplugged / powered off
                print(f"[PAD] disconnected: {self.path}", flush=True)
                self._drop()
                return False
            if len(data) < _JS_EVENT_SIZE:
                break
            _, value, ev_type, number = struct.unpack(_JS_EVENT_FMT, data)
            kind = ev_type & ~_JS_EVENT_INIT
            if kind == _JS_EVENT_BUTTON:
                self.buttons[number] = bool(value)
                if self._on_event is not None and not ev_type & _JS_EVENT_INIT:
                    self._on_event("button", number, float(value))
            elif kind == _JS_EVENT_AXIS:
                self.axes[number] = max(-1.0, min(1.0, value / _AXIS_MAX))
                if self._on_event is not None and not ev_type & _JS_EVENT_INIT:
                    self._on_event("axis", number, self.axes[number])
        return True

    def button(self, index: int) -> bool:
        return self.buttons.get(index, False)

    def axis(self, index: int) -> float:
        return self.axes.get(index, 0.0)

    def close(self) -> None:
        self._drop()


def _deadzone(value: float, dz: float) -> float:
    """Rescale ``value`` outside the deadzone so travel stays smooth from 0."""
    if abs(value) <= dz:
        return 0.0
    return (abs(value) - dz) / (1.0 - dz) * (1.0 if value > 0.0 else -1.0)


class RecordGamepad:
    """DualSense drop-in for ``RecordKeyboard``: hold-to-move plus edge-triggered actions.

    Exposes the same ``poll`` / ``command`` / ``pressed_keys`` / ``close`` surface as
    the keyboard helper, so a script can run either or both at once.
    """

    def __init__(
        self,
        speed: float,
        turn: float,
        strafe: float,
        *,
        device: str = "",
        deadzone: float = 0.15,
        use_sticks: bool = True,
        debug: bool = False,
        on_toggle_mode: Callable[[], None] | None = None,
        on_save_reset: Callable[[], None] | None = None,
        on_discard_reset: Callable[[], None] | None = None,
        on_next_pose_reset: Callable[[], None] | None = None,
        on_teleport_view: Callable[[], None] | None = None,
        on_toggle_recording: Callable[[], None] | None = None,
        on_restore_checkpoint: Callable[[], None] | None = None,
    ) -> None:
        self._speed = speed
        self._turn = turn
        self._strafe = strafe
        self._deadzone = deadzone
        self._use_sticks = use_sticks
        self._on_toggle_mode = on_toggle_mode
        self._on_save_reset = on_save_reset
        self._on_discard_reset = on_discard_reset
        self._on_next_pose_reset = on_next_pose_reset
        self._on_teleport_view = on_teleport_view
        self._on_toggle_recording = on_toggle_recording
        self._on_restore_checkpoint = on_restore_checkpoint
        self._edge_prev: set[str] = set()
        self._seed_edges = True
        self._js: LinuxJoystick | None = None

        path = find_gamepad(device)
        if path is None:
            available = list_joysticks()
            detail = (
                ", ".join(f"{p} ({n})" for p, n, _, _ in available) if available else "none readable"
            )
            print(
                f"[PAD] No gamepad found (looked for /dev/input/js*: {detail}). "
                f"Pair/plug the DualSense, or pass --gamepad_dev /dev/input/jsN.",
                flush=True,
            )
            return
        cb = self._debug_event if debug else None
        self._js = LinuxJoystick(path, on_event=cb)
        if self._js.connected:
            print(f"[PAD] Using gamepad: {self._js.name} ({path})", flush=True)
        else:
            print(
                f"[PAD] Could not open {path} (permissions?). "
                f"Check `ls -l {path}` — it must be readable by your user.",
                flush=True,
            )

    @property
    def connected(self) -> bool:
        return self._js is not None and self._js.connected

    @staticmethod
    def _debug_event(kind: str, number: int, value: float) -> None:
        label = BUTTON_NAMES.get(number, "?") if kind == "button" else ""
        print(f"[PAD-DEBUG] {kind} {number} {label} = {value:+.2f}", flush=True)

    def squeeze(self) -> float:
        """L2 analog travel as a 0..1 grasp squeeze command.

        The DualSense triggers rest at -1 and reach +1 fully pressed, so remap to
        0..1. Falls back to the digital button (some drivers expose L2 only as a
        button) which reads as full squeeze.
        """
        if self._js is None:
            return 0.0
        analog = (self._js.axis(AX_L2) + 1.0) * 0.5
        if analog > 0.02:
            return min(1.0, analog)
        return 1.0 if self._js.button(BTN_L2) else 0.0

    def _edges_now(self) -> set[str]:
        js = self._js
        if js is None:
            return set()
        edges = set()
        if js.button(BTN_CREATE):
            edges.add("TOGGLE_MODE")
        if self._use_sticks:
            # With sticks disabled the D-pad drives locomotion instead.
            hat_x = js.axis(AX_HAT_X)
            hat_y = js.axis(AX_HAT_Y)
            if hat_y < -0.5:  # D-pad up
                edges.add("SAVE")
            elif hat_y > 0.5:  # D-pad down
                edges.add("DISCARD")
            if hat_x < -0.5:  # D-pad left
                edges.add("RESTORE")
            elif hat_x > 0.5:  # D-pad right
                edges.add("NEXT")
        if js.button(BTN_OPTIONS):
            edges.add("TOGGLE_RECORDING")
        if js.button(BTN_L1):
            edges.add("TELEPORT")
        if js.button(BTN_R1):
            edges.add("RESTORE")
        return edges

    def poll(self) -> None:
        """Process edge-triggered buttons; call every loop iteration."""
        if self._js is None:
            return
        was_connected = self._js.connected
        self._js.poll()
        if self._js.connected and not was_connected:
            self._seed_edges = True  # ignore buttons held across a reconnect
        edge_now = self._edges_now()
        if self._seed_edges:
            # Ignore anything already held when the pad connected.
            self._edge_prev = edge_now
            self._seed_edges = False
            return
        newly = edge_now - self._edge_prev
        self._edge_prev = edge_now
        callbacks = {
            "TOGGLE_MODE": self._on_toggle_mode,
            "SAVE": self._on_save_reset,
            "DISCARD": self._on_discard_reset,
            "NEXT": self._on_next_pose_reset,
            "TELEPORT": self._on_teleport_view,
            "TOGGLE_RECORDING": self._on_toggle_recording,
            "RESTORE": self._on_restore_checkpoint,
        }
        for action in newly:
            callback = callbacks.get(action)
            if callback is not None:
                callback()

    def command(self) -> tuple[float, float, float]:
        """Return (lin_vel_x, lin_vel_y, ang_vel_z) from the left and right sticks."""
        js = self._js
        if js is None or not js.connected:
            return 0.0, 0.0, 0.0
        vy = 0.0
        if self._use_sticks:
            # WASD layout on the left stick: Y = forward/back (W/S), X = turn (A/D).
            vx = -_deadzone(js.axis(AX_LEFT_Y), self._deadzone) * self._speed
            wz = -_deadzone(js.axis(AX_LEFT_X), self._deadzone) * self._turn
            vy = -_deadzone(js.axis(AX_RIGHT_X), self._deadzone) * self._strafe
        else:
            hat_x, hat_y = js.axis(AX_HAT_X), js.axis(AX_HAT_Y)
            # D-pad up is -1 on HAT_Y, left is -1 on HAT_X.
            vx = self._speed if hat_y < -0.5 else (-self._speed if hat_y > 0.5 else 0.0)
            wz = self._turn if hat_x < -0.5 else (-self._turn if hat_x > 0.5 else 0.0)
        # ``+ 0.0`` keeps a negated zero from printing as "-0.00" in the debug line.
        return vx + 0.0, vy + 0.0, wz + 0.0

    def pressed_keys(self) -> str:
        """Compact debug string of active movement inputs."""
        js = self._js
        if js is None or not js.connected:
            return "-"
        parts = []
        if self._use_sticks:
            lx = _deadzone(js.axis(AX_LEFT_X), self._deadzone)
            ly = _deadzone(js.axis(AX_LEFT_Y), self._deadzone)
            rx = _deadzone(js.axis(AX_RIGHT_X), self._deadzone)
            if lx or ly:
                parts.append(f"LS({lx:+.1f},{ly:+.1f})")
            if rx:
                parts.append(f"RS({rx:+.1f})")
        else:
            hat_x, hat_y = js.axis(AX_HAT_X), js.axis(AX_HAT_Y)
            if hat_y < -0.5:
                parts.append("UP")
            elif hat_y > 0.5:
                parts.append("DOWN")
            if hat_x < -0.5:
                parts.append("LEFT")
            elif hat_x > 0.5:
                parts.append("RIGHT")
        return "".join(p + " " for p in parts).strip() or "-"

    def close(self) -> None:
        if self._js is not None:
            self._js.close()
            self._js = None


def _main() -> None:
    """Standalone mapping check: prints every button/axis change."""
    devices = list_joysticks()
    print("Readable joystick devices:")
    for path, name, axes, buttons in devices:
        print(f"  {path}: {name} ({axes} axes, {buttons} buttons)")
    pad = RecordGamepad(
        1.0,
        1.5,
        0.6,
        debug=True,
        on_toggle_mode=lambda: print(">>> CREATE: toggle locomotion", flush=True),
        on_save_reset=lambda: print(">>> D-PAD UP: save episode + reset", flush=True),
        on_discard_reset=lambda: print(">>> D-PAD DOWN: discard episode + reset", flush=True),
        on_teleport_view=lambda: print(">>> L1: teleport XR view", flush=True),
        on_toggle_recording=lambda: print(">>> OPTIONS: toggle recording", flush=True),
        on_restore_checkpoint=lambda: print(">>> R1 / D-PAD LEFT: restore checkpoint", flush=True),
    )
    if not pad.connected:
        return
    print("Press buttons (Ctrl+C to exit)...", flush=True)
    last = ""
    try:
        while True:
            pad.poll()
            line = f"cmd={pad.command()} active={pad.pressed_keys()}"
            if line != last:
                print(line, flush=True)
                last = line
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        pad.close()


if __name__ == "__main__":
    _main()
