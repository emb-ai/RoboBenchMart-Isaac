"""Drive the Sberchel humanoid around a darkstore with WASD, like a video game.

The ONNX RL policy ("zest") balances the robot and walks toward a velocity
command; this script turns held keys into that command:

    W : walk forward          S : walk backward
    A : turn left             D : turn right
    Q : strafe left           E : strafe right
    (release all keys -> stop and stand)

Hold-to-move: the command is active only while the key is held. Commands are in
the robot's body frame, so "forward" always follows where the robot is facing.

Requires the Isaac Sim GUI window (carb keyboard) — run WITHOUT --headless, and
click the Isaac window once so it captures your key presses.

    python scripts/demo_sberchel_kb.py --config_dir_path generated_envs/isaac_sample
"""

import os
import sys
import argparse
from isaaclab.app import AppLauncher

# Make the sibling green_challenge package importable (for policy.network.Network).
# Default: green_challenge sits next to this repo (../green_challenge). Override with GREEN_CHALLENGE_PATH.
_DEFAULT_GC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "green_challenge"))
_GC = os.environ.get("GREEN_CHALLENGE_PATH", _DEFAULT_GC)
_gc_parent = os.path.dirname(os.path.abspath(_GC))
if _gc_parent not in sys.path:
    sys.path.insert(0, _gc_parent)

parser = argparse.ArgumentParser(description="WASD-drive the Sberchel humanoid in a darkstore.")
parser.add_argument("--config_dir_path", type=str, help="Path to the config directory.")
parser.add_argument("--task", type=str, default="Isaac-Dsynth-Sberchel-Base-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed to use.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--episode_len", type=int, default=-1, help="Steps to run (-1 = until the window is closed).")
# Empirical sweet spot for this "vel_med" policy: ~1.0 m/s and ~1.5 rad/s give
# the fastest actual motion. Commanding higher makes it shuffle in place, not
# move faster (the policy was trained for medium velocities).
parser.add_argument("--speed", type=float, default=1.0, help="Forward/back speed for W/S (m/s). ~1.0 is the practical max.")
parser.add_argument("--turn", type=float, default=1.5, help="Turn rate for A/D (rad/s). ~1.5 is the practical max.")
parser.add_argument("--strafe", type=float, default=0.6, help="Strafe speed for Q/E (m/s).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

from isaaclab_tasks.utils import parse_env_cfg

sys.path.append('.')
import dsynth_isaac.tasks.manager_based


class WasdKeyboard:
    """Track currently-held WASDQE keys via the Isaac Sim (carb) keyboard."""

    def __init__(self, speed: float, turn: float, strafe: float) -> None:
        import carb
        import omni
        self._carb = carb
        self._speed = speed
        self._turn = turn
        self._strafe = strafe
        self._pressed = set()
        self._ok = False
        try:
            self._appwindow = omni.appwindow.get_default_app_window()
            self._input = carb.input.acquire_input_interface()
            self._keyboard = self._appwindow.get_keyboard()
            self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)
            self._ok = True
        except Exception as exc:  # headless / no window -> no keyboard
            print(f"[WASD] keyboard unavailable ({exc!r}); robot will just stand. "
                  f"Run WITHOUT --headless to drive it.")

    def _on_event(self, event, *args, **kwargs):
        # Only press/release carry a key we care about. Other event types (e.g.
        # CHAR) deliver event.input as a plain string, so we must check the type
        # BEFORE touching event.input, and accept either an enum or a string.
        carb = self._carb
        try:
            et = event.type
            if et not in (
                carb.input.KeyboardEventType.KEY_PRESS,
                carb.input.KeyboardEventType.KEY_RELEASE,
            ):
                return True
            raw = event.input
            name = raw if isinstance(raw, str) else getattr(raw, "name", "")
            name = name.rsplit(".", 1)[-1].upper()  # "KeyboardInput.W" -> "W"
            if et == carb.input.KeyboardEventType.KEY_PRESS:
                self._pressed.add(name)
            else:
                self._pressed.discard(name)
        except Exception:
            pass
        return True

    def command(self):
        """Return (lin_vel_x, lin_vel_y, ang_vel_z) from the held keys."""
        p = self._pressed
        vx = (self._speed if "W" in p else 0.0) - (self._speed if "S" in p else 0.0)
        wz = (self._turn if "A" in p else 0.0) - (self._turn if "D" in p else 0.0)
        vy = (self._strafe if "Q" in p else 0.0) - (self._strafe if "E" in p else 0.0)
        return vx, vy, wz

    def close(self):
        if self._ok:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            except Exception:
                pass


_HELP = """
==================== WASD CONTROL ====================
  W forward    S backward    A turn left    D turn right
  Q strafe L   E strafe R    (release = stop & stand)
  >>> Click the Isaac Sim window so it captures your keys <<<
=====================================================
"""


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.scene.scene_config_dir_path = args_cli.config_dir_path
    env_cfg.target_item_name_regex = ".*Nestle Honey.*"

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env.reset(seed=args_cli.seed)

    from green_challenge.policy.network import Network
    net = Network(env, type="zest", device="cpu", closed_kinematics=False)
    net.reset()

    kb = WasdKeyboard(args_cli.speed, args_cli.turn, args_cli.strafe)
    print(_HELP, flush=True)

    t = 0
    while simulation_app.is_running() and (args_cli.episode_len < 0 or t < args_cli.episode_len):
        with torch.inference_mode():
            vx, vy, wz = kb.command()
            net.set_command_detailed(vx, vy, wz)
            net.set_joint_overrides({
                "torso_joint_pos": [0.0] * 13,
                "wrist_joint_pos": [0.0] * 4,
                "finger_joint_pos": [0.0] * 12,
            })
            loco = net(net.obs())
            legs = list(loco["joint_pos"][:12])
            act_list = legs + [0.0] * 13 + [0.0] * 4 + [0.0] * 12
            act_tensor = torch.as_tensor(
                act_list, dtype=torch.float32, device=env.unwrapped.device
            ).unsqueeze(0).repeat(args_cli.num_envs, 1)
            _ = env.step(act_tensor)
        t += 1

    kb.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
