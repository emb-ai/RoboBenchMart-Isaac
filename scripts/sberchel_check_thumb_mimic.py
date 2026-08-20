"""Headless check: can the thumb pitch joints physically reach their commands?

The thumb pitch drives two mimic-coupled slave joints (intermediate = 1.334x,
range 0..0.8; distal = 0.898x, range 0..0.4), created by the MJCF->USD import
pass. Two failure modes look identical in teleop (a saturated 8.5 Nm error on
thumb_proximal_pitch_joint) but need different fixes:

* **Slave-limit ceiling** -- the distal slave hits its 0.4 limit at
  0.4/0.898 = 0.445 rad of master travel, so the master stalls there. Expected
  and handled by capping commands (_THUMB_PITCH_MAX_REACHABLE in the
  retargeter).
* **Mimic gearing sign fighting the slave limits** -- the importer logs
  gearing=-1.334 where the MJCF equality is +1.334. If the sign convention
  does not match the slave's 0..0.8 range after import, the slave pins to its
  ZERO limit immediately and the master stalls near 0 in free air (observed
  live: right thumb pitch pinned at err +0.48 rad / 8.5 Nm for a whole
  session, including while walking with an empty hand).

This script ramps each thumb pitch target 0 -> 0.5 rad with every other joint
held at default, gravity off, nothing to collide with, and prints where the
joint actually lands together with its mimic slaves.

Run:
    python scripts/sberchel_check_thumb_mimic.py
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from dsynth_isaac.green_challenge_path import ensure_green_challenge_importable  # noqa: E402
from dsynth_isaac.sim_defaults import apply_sberchel_cli_defaults  # noqa: E402

ensure_green_challenge_importable()

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
# Sberchel runs on green_challenge's CPU physics by default; --device overrides.
apply_sberchel_cli_defaults(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from dsynth_isaac.robots.sberchel_cfg import SBERCHEL_XR_LOCOMOTION_NECK_ROBOT  # noqa: E402

RAMP_STEPS = 150  # 1 s at 150 Hz
HOLD_STEPS = 150  # settle time at full command
TARGET_MAX = 0.5  # the joint's upper limit -- what teleop was commanding
DT = 1.0 / 150.0


def main() -> None:
    # Gravity off: the full humanoid would otherwise crumple without its
    # locomotion policy, and free-air behaviour is exactly what we are testing.
    sim = SimulationContext(SimulationCfg(dt=DT, device="cpu", gravity=(0.0, 0.0, 0.0)))
    robot = Articulation(SBERCHEL_XR_LOCOMOTION_NECK_ROBOT.replace(prim_path="/World/Robot"))
    sim.reset()
    robot.update(DT)

    names = list(robot.data.joint_names)
    default = robot.data.default_joint_pos.clone()

    def settle(steps: int, targets: torch.Tensor) -> None:
        for _ in range(steps):
            robot.set_joint_position_target(targets)
            robot.write_data_to_sim()
            sim.step()
            robot.update(DT)

    verdicts: dict[str, str] = {}
    for side in ("left", "right"):
        master = f"{side}_thumb_proximal_pitch_joint"
        slaves = [f"{side}_thumb_intermediate_joint", f"{side}_thumb_distal_joint"]
        jid = names.index(master)
        slave_ids = [names.index(s) for s in slaves if s in names]

        print(
            f"\n[{master}] ramp 0 -> {TARGET_MAX} rad over {RAMP_STEPS} steps, free air:",
            flush=True,
        )
        targets = default.clone()
        settle(50, targets)  # relax into default pose first

        for step in range(RAMP_STEPS + HOLD_STEPS):
            frac = min(1.0, step / RAMP_STEPS)
            targets[:, jid] = frac * TARGET_MAX
            robot.set_joint_position_target(targets)
            robot.write_data_to_sim()
            sim.step()
            robot.update(DT)
            if step % 30 == 0 or step == RAMP_STEPS + HOLD_STEPS - 1:
                cmd = float(targets[0, jid])
                meas = float(robot.data.joint_pos[0, jid])
                slave_meas = " ".join(
                    f"{names[s].split('thumb_')[1]}={float(robot.data.joint_pos[0, s]):+.3f}"
                    for s in slave_ids
                )
                print(
                    f"  cmd={cmd:+.3f} measured={meas:+.3f} err={cmd - meas:+.3f}  | {slave_meas}",
                    flush=True,
                )

        final_meas = float(robot.data.joint_pos[0, jid])
        final_err = TARGET_MAX - final_meas
        if final_meas < 0.10:
            verdict = (
                "BLOCKED near zero in free air -> imported mimic gearing sign fights the "
                "slave 0..0.8 limits; fix the [mimic] import pass"
            )
        elif final_meas < 0.40:
            verdict = (
                f"stalls at {final_meas:.3f} rad, well below the 0.445 mimic ceiling -> "
                "self-collision obstruction: the thumb assembly (with its mimic-driven "
                "intermediate/distal segments) presses into the palm/fingers "
                "(enabled_self_collisions=true in the green_challenge interface cfg; "
                "convex-decomposition hulls may collide earlier than the source meshes)"
            )
        elif final_err > 0.08:
            verdict = (
                f"stalls at {final_meas:.3f} rad -> mimic slave limit ceiling "
                "(expected ~0.445; the retargeter cap at 0.44 handles this)"
            )
        else:
            verdict = "OK -- reaches the command; live jams must be collision/object contact"
        print(f"  VERDICT: {verdict}")
        verdicts[master] = verdict

        targets[:, jid] = default[:, jid]
        settle(75, targets)  # back to open before the other hand

    print("\n=== summary ===", flush=True)
    for joint, verdict in verdicts.items():
        print(f"  {joint}: {verdict}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
