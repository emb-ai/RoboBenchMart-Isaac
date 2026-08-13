"""Validate the Sberchel DexPilot hand configs without launching Isaac Sim.

Two checks, both offline:

* **Curl tracking.** Sweep the hand model from open to fist, feed its own
  fingertip geometry back in as if it were operator landmarks, and read out the
  joints the robot can actually command. Those must increase monotonically. A
  column that reverses is the signature of the failure where the robot's fingers
  moved OPPOSITE to the operator's.
* **Fingertip workspace.** How far each tip travels between open and fist. The
  model has to move like a hand: a human sweeps roughly 18 -> 7 cm. When the
  fingers were modelled as a single joint with the tip rigidly attached this was
  only ~4 cm, and no frame rotation could reconcile open with fist (32.7 deg
  residual). Modelling the full chain brought it to ~10 cm and 12 deg.

Note the thumb is exempt from the span check: its tip distance from the wrist
barely changes because it opposes ACROSS the palm rather than curling toward it.
Its direction swings ~40 deg, which is the meaningful quantity.

Note ``low_pass_alpha`` in the config: each ``retarget()`` call moves only part
of the way to the solution, so the sweep must settle before reading. Sampling
after a single call measures the filter, not the geometry.

WHAT THIS CANNOT CATCH: it feeds model-derived landmarks, so every target is
reachable by construction. It says nothing about whether the operator->robot
frame rotation (``_OPERATOR2MANO``) is right -- that needs a real capture and
``scripts/sberchel_fit_dex_frame.py``. Passing here means "the hand model moves
like a hand", not "dex retargeting works".

Usage:
    python scripts/sberchel_build_dex_hand_urdf.py     # regenerate the models
    python scripts/sberchel_check_dex_retargeting.py   # then verify them
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ASSETS = Path(__file__).resolve().parents[1] / "dsynth_isaac" / "assets" / "sberchel_xr"
SETTLE = 60  # iterations for the config's low-pass filter to converge
DIGITS = ("thumb", "index", "middle", "ring", "pinky")
TOL = 0.05  # rad

HANDS = {
    "left": ("sberchel_hand_left_dexpilot.yml", "sberchel_left_hand.urdf", "left_hand_base_link"),
    "right": ("sberchel_hand_right_dexpilot.yml", "sberchel_right_hand.urdf", "right_hand_base_link"),
}


def build(work: Path, cfg_name: str, urdf_name: str, scaling: float | None):
    from dex_retargeting.retargeting_config import RetargetingConfig

    cfg_path = work / cfg_name
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["retargeting"]["urdf_path"] = str((work / urdf_name).resolve())
    if scaling is not None:
        cfg["retargeting"]["scaling_factor"] = scaling
    cfg_path.write_text(yaml.dump(cfg))
    return RetargetingConfig.load_from_file(str(cfg_path)).build()


def curl_tracking(rt, side: str, base_link: str, hand_joints: list[str]) -> list[tuple[float, dict]]:
    """Sweep the model from open to fist and read back the commandable joints.

    This replaced an exact round trip, which stopped being meaningful once the
    hand model gained the intermediate joints: 11 DOFs against 5 fingertip
    targets is underdetermined -- proximal and intermediate rotate about parallel
    axes, so their SUM is pinned but the split is not. Demanding exact recovery
    of each joint therefore fails on a perfectly good model.

    What actually predicts teleop behaviour is monotonicity: as the operator
    curls, the joints the robot can command must curl too, and never reverse.
    A non-monotonic column here is the signature of the inversion that made an
    earlier build move the fingers backwards.
    """
    opt, robot = rt.optimizer, rt.optimizer.robot
    tips = [robot.get_link_index(f"{side}_{d}_tip") for d in DIGITS]
    base_id = robot.get_link_index(base_link)
    idx = np.asarray(opt.target_link_human_indices)
    names = robot.dof_joint_names
    lo = np.array([l for l, _ in robot.joint_limits])
    hi = np.array([h for _, h in robot.joint_limits])

    out = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        robot.compute_forward_kinematics(lo + frac * (hi - lo))
        origin = robot.get_link_pose(base_id)[:3, 3]
        pts = np.zeros((21, 3))
        for slot, lid in zip((4, 8, 12, 16, 20), tips):
            pts[slot] = robot.get_link_pose(lid)[:3, 3] - origin
        ref = pts[idx, :] if opt.retargeting_type == "POSITION" else pts[idx[1, :], :] - pts[idx[0, :], :]
        for _ in range(SETTLE):
            q_hat = rt.retarget(ref)
        out.append((frac, {n: float(v) for n, v in zip(names, q_hat) if n in hand_joints}))
    return out


def fingertip_workspace(rt, side: str, base_link: str) -> list[tuple[str, float, float]]:
    """Wrist-to-tip distance at full extension and full curl, per digit (cm).

    A human finger sweeps roughly 18 -> 7 cm. Anything much narrower than that
    means most of the operator's curl range is unreachable, so DexPilot will
    saturate on real hand input no matter what the round trip says.
    """
    opt, robot = rt.optimizer, rt.optimizer.robot
    digits = ("thumb", "index", "middle", "ring", "pinky")
    tip_ids = [robot.get_link_index(f"{side}_{d}_tip") for d in digits]
    base_id = robot.get_link_index(base_link)
    lo = np.array([l for l, _ in robot.joint_limits])
    hi = np.array([h for _, h in robot.joint_limits])

    def dists(q):
        robot.compute_forward_kinematics(np.asarray(q, float))
        origin = robot.get_link_pose(base_id)[:3, 3]
        return [float(np.linalg.norm(robot.get_link_pose(i)[:3, 3] - origin)) * 100 for i in tip_ids]

    return list(zip(digits, dists(lo), dists(hi)))


def main() -> int:
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for f in ASSETS.iterdir():
            shutil.copy2(f, work / f.name)

        for side, (cfg_name, urdf_name, base_link) in HANDS.items():
            configured = yaml.safe_load((work / cfg_name).read_text())["retargeting"]
            rt = build(work, cfg_name, urdf_name, 1.0)
            commandable = [f"{side}_thumb_proximal_yaw_joint", f"{side}_thumb_proximal_pitch_joint"] + [
                f"{side}_{d}_proximal_joint" for d in DIGITS if d != "thumb"
            ]
            sweep = curl_tracking(rt, side, base_link, commandable)

            names = sorted(sweep[0][1])
            bad = [n for n in names
                   if any(s[1][n] < prev[1][n] - 1e-3 for prev, s in zip(sweep, sweep[1:]))]
            ws = fingertip_workspace(rt, side, base_link)
            narrow = [d for d, o, c in ws if d != "thumb" and (o - c) < 6.0]
            ok = not bad and not narrow
            failed += not ok

            print(f"{'PASS' if ok else 'FAIL'} {side:5s} hand  "
                  f"(scaling_factor={configured.get('scaling_factor')}, "
                  f"low_pass_alpha={configured.get('low_pass_alpha')})")
            print("       curl tracking, commanded joint vs operator curl:")
            print("         curl   " + "  ".join(f"{n.replace(side + '_', '')[:14]:>14s}" for n in names))
            for frac, q in sweep:
                print(f"         {frac:4.2f}   " + "  ".join(f"{q[n]:14.2f}" for n in names))
            if bad:
                print(f"         NON-MONOTONIC: {', '.join(bad)}  <-- fingers would move backwards")

            print("       fingertip travel, wrist-to-tip cm (human ~18 -> ~7):")
            for digit, o, c in ws:
                note = "  (direction, not distance, is what matters here)" if digit == "thumb" else (
                    "  <-- too narrow" if (o - c) < 6.0 else "")
                print(f"         {digit:7s} {o:5.1f} -> {c:5.1f}  (span {o - c:5.1f}){note}")
            print()

    if failed:
        print(f"RESULT: {failed} hand(s) FAILED -- see the flagged rows above.")
        print("Regenerate with scripts/sberchel_build_dex_hand_urdf.py and re-check.")
        return 1
    print("RESULT: PASS -- both hands curl monotonically with a human-scale fingertip sweep.")
    print("This does NOT validate _OPERATOR2MANO; capture a hand and run")
    print("scripts/sberchel_fit_dex_frame.py for that.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
